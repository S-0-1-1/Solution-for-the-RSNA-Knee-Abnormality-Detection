import os
import json
from collections import Counter
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi

THREADS = int(os.environ.get('KNEE_THREADS', '1'))
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(THREADS)
BONES = {1: 'femur', 2: 'patella', 3: 'fibula', 4: 'tibia'}
ROIS = ['medial_TF', 'lateral_TF', 'notch', 'PF', 'MCL', 'joint', 'popliteal', 'bones']


def load_volume(d):
    dss = [pydicom.dcmread(os.path.join(d, f)) for f in os.listdir(d) if f.endswith('.dcm')]
    shape0 = Counter((int(x.Rows), int(x.Columns)) for x in dss).most_common(1)[0][0]
    dss = [x for x in dss if (int(x.Rows), int(x.Columns)) == shape0]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    nrm = np.cross(iop[:3], iop[3:])
    proj = np.array([np.dot(np.array(x.ImagePositionPatient, float), nrm) for x in dss])
    o = np.argsort(proj)
    dss = [dss[i] for i in o]
    proj = proj[o]
    keep = np.r_[True, np.diff(np.round(proj, 2)) > 0]
    dss = [x for x, k in zip(dss, keep) if k]
    proj = proj[keep]
    px = [x.pixel_array for x in dss]
    if px[0].ndim != 2:
        raise ValueError(f'multi-frame pixel data {px[0].shape}')
    vol = np.stack(px).astype(np.float32)
    vol = vol * float(getattr(dss[0], 'RescaleSlope', 1) or 1) + float(getattr(dss[0], 'RescaleIntercept', 0) or 0)
    if getattr(dss[0], 'PhotometricInterpretation', '') == 'MONOCHROME1':
        vol = vol.max() - vol
    ps = [float(v) for v in dss[0].PixelSpacing]
    dz = float(np.median(np.diff(proj))) if len(proj) > 1 else float(getattr(dss[0], 'SliceThickness', 1.0) or 1.0)
    return vol, (dz, ps[0], ps[1]), iop, np.array(dss[0].ImagePositionPatient, float), dss[0]


def series_image(d):
    vol, (dz, dy, dx), iop, origin, hdr = load_volume(d)
    img = sitk.GetImageFromArray(vol)
    X, Y = iop[:3], iop[3:]
    img.SetDirection(np.stack([X, Y, np.cross(X, Y)], axis=1).flatten().tolist())
    img.SetSpacing((dx, dy, dz))
    img.SetOrigin(origin.tolist())
    return img, str(getattr(hdr, 'SeriesDescription', ''))


def to_las(img):
    return sitk.DICOMOrient(img, 'LAS')


def resample_to(img, new_sp, interp):
    old_sp, old_sz = np.array(img.GetSpacing()), np.array(img.GetSize())
    new_sz = np.maximum(1, np.round(old_sz * old_sp / np.array(new_sp))).astype(int).tolist()
    return sitk.Resample(img, new_sz, sitk.Transform(), interp, img.GetOrigin(), tuple(float(s) for s in new_sp),
                         img.GetDirection(), 0.0, img.GetPixelID())


def clean_mask(m):
    vox_ml = float(np.prod(m.GetSpacing())) / 1000.0
    arr = sitk.GetArrayFromImage(m)
    arr[arr > 4] = 0
    out, st = np.zeros_like(arr, dtype=np.uint8), {}
    min_vox = max(1, int(round(0.5 / vox_ml)))
    for lab, name in BONES.items():
        b = sitk.GetImageFromArray((arr == lab).astype(np.uint8))
        b.CopyInformation(m)
        cc = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(b), minimumObjectSize=min_vox))
        n_all, n_keep = int((arr == lab).sum()), int((cc > 0).sum())
        st[f'{name}_ml'] = round(n_keep * vox_ml, 2)
        st[f'{name}_ncc'] = int(cc.max())
        st[f'{name}_lcc'] = round(float((cc == 1).sum()) / max(n_keep, 1), 3)
        st[f'{name}_specks_ml'] = round((n_all - n_keep) * vox_ml, 2)
        out[cc > 0] = lab
    o = sitk.GetImageFromArray(out)
    o.CopyInformation(m)
    return o, st


def lps_points(arr, img, lab, cap=120000):
    idx = np.argwhere(arr == lab)
    if len(idx) == 0:
        return None
    if len(idx) > cap:
        idx = idx[np.random.default_rng(0).choice(len(idx), cap, replace=False)]
    D = np.array(img.GetDirection()).reshape(3, 3)
    s = np.array(img.GetSpacing())
    o = np.array(img.GetOrigin())
    return o + idx[:, ::-1].astype(np.float64) * s @ D.T


def pc(a, q):
    return float(np.percentile(a, q))


def box(xa, xb, ya, yb, za, zb):
    return [min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb), min(za, zb), max(za, zb)]


def fov_center(img):
    return [float(v) for v in img.TransformContinuousIndexToPhysicalPoint(((np.array(img.GetSize()) - 1) / 2).tolist())]


def quality(q):
    n = len([w for w in str(q.get('flag_reasons') or '').split('|') if w])
    lcc = float(np.mean([q.get(f'{b}_lcc', 0.0) for b in ('femur', 'tibia', 'patella')]))
    return int(q.get('qc_ok', 0)), int(q.get('geom_ok', 0)), -n, lcc


def better(a, b):
    return quality(a) > quality(b)


def jdump(obj, path):
    json.dump(obj, open(path, 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def bone_regions(mask, st, tag):
    arr = sitk.GetArrayFromImage(mask)
    P = {n: lps_points(arr, mask, lab) for lab, n in BONES.items()}
    fem, pat, fib, tib = P['femur'], P['patella'], P['fibula'], P['tibia']
    C = {n: None if v is None else v.mean(0) for n, v in P.items()}
    q, why = {}, []
    q['femur_above_tibia'] = bool(fem is not None and tib is not None and C['femur'][2] > C['tibia'][2])
    q['patella_ant_femur'] = bool(pat is not None and fem is not None and C['patella'][1] < C['femur'][1])
    q['fibula_post_tibia'] = bool(fib is not None and tib is not None and C['fibula'][1] > C['tibia'][1])
    lf = None if fib is None or tib is None else 'L' if C['fibula'][0] > C['tibia'][0] else 'R'
    q['lat_fibula'], q['lat_tag'] = lf, tag
    q['lat_agree'] = None if lf is None or tag is None else bool(lf == tag)
    if lf and (tag is None or lf == tag):
        q['lat_final'], q['lat_source'] = lf, 'fibula'
    elif tag:
        q['lat_final'], q['lat_source'] = tag, 'tag'
    else:
        q['lat_final'], q['lat_source'] = None, 'unknown'
        why.append('lat_unknown')
    if st['femur_ml'] < 40:
        why.append('femur_small')
    if st['tibia_ml'] < 30:
        why.append('tibia_small')
    if st['patella_ml'] < 2:
        why.append('patella_small')
    if st['patella_ml'] > 45:
        why.append('patella_big')
    if st['fibula_ml'] < 0.5:
        why.append('fibula_missing')
    for b in ('femur', 'tibia', 'patella'):
        if st[f'{b}_ml'] > 0 and st[f'{b}_lcc'] < 0.9:
            why.append(f'{b}_fragmented')
    if not q['femur_above_tibia']:
        why.append('geom_femur_tibia')
    if not q['patella_ant_femur']:
        why.append('geom_patella')
    if not q['fibula_post_tibia']:
        why.append('geom_fibula')
    if q['lat_agree'] is False:
        why.append('lat_mismatch')
    q['flag_reasons'], q['qc_ok'] = '|'.join(why), int(not why)
    q['geom_ok'] = int(fem is not None and tib is not None and q['femur_above_tibia'])
    if not q['geom_ok']:
        return q
    s = 1.0 if q['lat_final'] == 'L' else -1.0
    m = -s
    zf, zt = pc(fem[:, 2], 5), pc(tib[:, 2], 95)
    zj = 0.5 * (zf + zt)
    top = tib[tib[:, 2] > zt - 15]
    top = top if len(top) > 200 else tib
    con = fem[fem[:, 2] < zf + 25]
    con = con if len(con) > 200 else fem
    x0, x1, y0, y1 = pc(top[:, 0], 2), pc(top[:, 0], 98), pc(top[:, 1], 2), pc(top[:, 1], 98)
    cx0, cx1, cy0, cy1 = pc(con[:, 0], 2), pc(con[:, 0], 98), pc(con[:, 1], 2), pc(con[:, 1], 98)
    if x1 - x0 > cx1 - cx0 + 15:
        x0, x1 = cx0, cx1
    xc, hw, ym = 0.5 * (x0 + x1), 0.5 * (x1 - x0), 0.5 * (y0 + y1)
    xe = xc + m * hw
    R = {'medial_TF': box(xc, xc + m * (hw + 8), y0 - 5, y1 + 5, zj - 18, zj + 22),
         'lateral_TF': box(xc, xc + s * (hw + 8), y0 - 5, y1 + 5, zj - 18, zj + 22),
         'notch': box(xc - 12, xc + 12, y0, y1, zj - 8, zj + 35),
         'MCL': box(xe - m * 10, xe + m * 18, ym - 22, ym + 18, zj - 55, zj + 45),
         'popliteal': box(xc + s * 10, xc + m * (hw + 20), cy1 - 15, cy1 + 45, zj - 50, zj + 40)}
    ya = min(y0, cy0) - 5
    if pat is not None and st['patella_ml'] >= 2:
        px0, px1, py0, py1, pz0, pz1 = [pc(pat[:, i], k) for i in range(3) for k in (2, 98)]
        R['PF'] = box(px0 - 8, px1 + 8, py0 - 6, py1 + 18, pz0 - 8, pz1 + 8)
        R['joint'] = box(min(x0, cx0) - 12, max(x1, cx1) + 12, min(py0, y0) - 10, cy1 + 12, zj - 25, max(pz1 + 35, zj + 60))
        R['bones'] = box(min(x0, cx0, px0) - 10, max(x1, cx1, px1) + 10, min(ya, py0 - 5), max(y1, cy1) + 5, zj - 60,
                         max(zj + 70, pz1 + 10))
    else:
        R['joint'] = box(min(x0, cx0) - 12, max(x1, cx1) + 12, cy0 - 25, cy1 + 12, zj - 25, zj + 70)
        R['bones'] = box(min(x0, cx0) - 10, max(x1, cx1) + 10, ya, max(y1, cy1) + 5, zj - 60, zj + 70)
    q['K'], q['rois'], q['knee_width_mm'] = [xc, ym, zj], R, round(2 * hw, 1)
    q['femur_win_ml'] = round(float(np.mean((fem[:, 2] >= zj) & (fem[:, 2] <= zj + 60))) * st['femur_ml'], 2)
    q['tibia_win_ml'] = round(float(np.mean((tib[:, 2] <= zj) & (tib[:, 2] >= zj - 60))) * st['tibia_ml'], 2)
    return q


def fix_labels(m, patella_max_ml=45.0, fibula_max_ml=30.0):
    arr = sitk.GetArrayFromImage(m).astype(np.uint8)
    arr[arr > 4] = 0
    spz = np.array(m.GetSpacing())[::-1]
    D = np.array(m.GetDirection()).reshape(3, 3)
    org = np.array(m.GetOrigin())
    spx = np.array(m.GetSpacing())
    vox_ml = float(np.prod(spx)) / 1000.0

    def phys(idx):
        return org + np.asarray(idx, float)[..., ::-1] * spx @ D.T
    nS = arr.shape[0]
    info = dict(bridged=0, patella_found=0, fibula_found=0, femur_top=0, tibia_bottom=0, shaft_added_ml=0.0)
    obj, nobj = ndi.label(arr > 0)
    out = np.zeros_like(arr)
    if nobj == 0:
        o = sitk.GetImageFromArray(out)
        o.CopyInformation(m)
        return o, info
    votes = np.bincount(obj.ravel().astype(np.int64) * 5 + arr.ravel(), minlength=(nobj + 1) * 5).reshape(nobj + 1, 5)
    votes[0] = 0
    f_obj = int(np.argmax(votes[:, 1])) if votes[:, 1].max() > 0 else 0
    tv = votes[:, 4].copy()
    t_in_f = int(tv[f_obj]) if f_obj else 0
    tv[f_obj] = 0
    t_obj = int(np.argmax(tv)) if tv.max() > 0 else 0
    F = obj == f_obj if f_obj else np.zeros(arr.shape, bool)
    if f_obj and t_in_f > 2 * (tv[t_obj] if t_obj else 0) and t_in_f > 0:
        info['bridged'] = 1
        sl = ndi.find_objects(F.astype(np.uint8))[0]
        Fb, Ab = F[sl], arr[sl]
        d_f = ndi.distance_transform_edt(~((Ab == 1) & Fb), sampling=spz)
        d_t = ndi.distance_transform_edt(~((Ab == 4) & Fb), sampling=spz)
        T = np.zeros(arr.shape, bool)
        T[sl] = Fb & (d_t < d_f)
        F = F & ~T
    else:
        T = obj == t_obj if t_obj else np.zeros(arr.shape, bool)

    def pts(mask, cap=60000):
        idx = np.argwhere(mask)
        if len(idx) > cap:
            idx = idx[np.random.default_rng(0).choice(len(idx), cap, replace=False)]
        return phys(idx) if len(idx) else None
    Pf, Pt = pts(F), pts(T)
    if Pf is not None:
        zf = float(np.percentile(Pf[:, 2], 5))
        cond = Pf[Pf[:, 2] < zf + 25]
        cond_y = float(np.median(cond[:, 1])) if len(cond) else float(np.median(Pf[:, 1]))
        info['femur_top'] = int(np.where(F.any(axis=(1, 2)))[0].max() >= 0.8 * (nS - 1))
    if Pt is not None:
        zt = float(np.percentile(Pt[:, 2], 95))
        prox = Pt[Pt[:, 2] > zt - 40]
        tcx, tcy = float(np.mean(prox[:, 0])), float(np.mean(prox[:, 1]))
        info['tibia_bottom'] = int(np.where(T.any(axis=(1, 2)))[0].min() <= 0.2 * (nS - 1))

    def best_piece(lab, rule, max_ml, min_ml):
        pcs, n = ndi.label(arr == lab)
        if n == 0:
            return None
        size = np.bincount(pcs.ravel())[1:] * vox_ml
        com = ndi.center_of_mass(np.ones(arr.shape, np.uint8), pcs, range(1, n + 1))
        boxes = ndi.find_objects(pcs)
        best = None
        for k in range(n):
            if not min_ml <= size[k] <= max_ml:
                continue
            c = phys(np.array(com[k]))
            zr = phys(np.array([[boxes[k][0].start, 0, 0], [boxes[k][0].stop - 1, 0, 0]]))[:, 2]
            if rule(c, float(zr.min()), float(zr.max())) and (best is None or size[k] > size[best]):
                best = k
        return None if best is None else pcs == best + 1
    PAT = FIB = None
    if Pf is not None:
        PAT = best_piece(2, lambda c, z0, z1: c[1] < cond_y - 10 and zf - 15 <= c[2] <= zf + 90, patella_max_ml, 1.0)
    if Pt is not None and Pf is not None:
        FIB = best_piece(3, lambda c, z0, z1: z1 < zf + 5 and c[2] < zt - 5 and c[1] > tcy - 5 and abs(c[0] - tcx) > 12,
                         fibula_max_ml, 0.3)

    def promote(piece, lab):
        o_id = int(obj[tuple(np.argwhere(piece)[0])])
        if o_id not in (f_obj, t_obj) and votes[o_id, lab] >= 0.5 * votes[o_id].sum():
            return obj == o_id
        return piece
    out[F] = 1
    out[T] = 4
    if PAT is not None:
        out[promote(PAT, 2)] = 2
        info['patella_found'] = 1
    if FIB is not None:
        out[promote(FIB, 3)] = 3
        info['fibula_found'] = 1
    used = {int(v) for v in np.unique(obj[out > 0]) if v}
    msp = spz[1:]

    def footprint(mask, upper):
        zs = np.where(mask.any(axis=(1, 2)))[0]
        if not len(zs):
            return None
        zmed = int(np.median(zs))
        part = mask[zmed:] if upper else mask[:zmed + 1]
        a, b = np.where(part.any(axis=0))
        return zmed, a.min(), a.max(), b.min(), b.max()
    fpF, fpT = footprint(out == 1, True), footprint(out == 4, False)
    for k, sl in enumerate(ndi.find_objects(obj), start=1):
        if sl is None or k in used:
            continue
        ca, cl = (sl[1].start + sl[1].stop - 1) / 2, (sl[2].start + sl[2].stop - 1) / 2
        if fpF and sl[0].start > fpF[0]:
            ea, el = 5 / msp[0], 5 / msp[1]
            if fpF[1] - ea <= ca <= fpF[2] + ea and fpF[3] - el <= cl <= fpF[4] + el:
                out[obj == k] = 1
                info['shaft_added_ml'] = info.get('shaft_added_ml', 0) + float((obj == k).sum()) * vox_ml
                continue
        if fpT and sl[0].stop - 1 < fpT[0]:
            sa, sb = 3 / msp[0], 3 / msp[1]
            if fpT[1] + sa <= ca <= fpT[2] - sa and fpT[3] + sb <= cl <= fpT[4] - sb:
                out[obj == k] = 4
                info['shaft_added_ml'] = info.get('shaft_added_ml', 0) + float((obj == k).sum()) * vox_ml
    for lab in (1, 4, 2, 3):
        for x in range(out.shape[2]):
            sl2 = out[:, :, x]
            if (sl2 == lab).any():
                fill = ndi.binary_fill_holes(sl2 == lab) & (sl2 == 0)
                sl2[fill] = lab
    zs1, zs4 = np.where((out == 1).any(axis=(1, 2)))[0], np.where((out == 4).any(axis=(1, 2)))[0]
    info['femur_top'] = int(len(zs1) > 0 and zs1.max() >= 0.8 * (nS - 1))
    info['tibia_bottom'] = int(len(zs4) > 0 and zs4.min() <= 0.2 * (nS - 1))
    info['shaft_added_ml'] = round(info['shaft_added_ml'], 2)
    changed = (arr > 0) & (out != arr)
    info['changed_ml'] = round(float(changed.sum()) * vox_ml, 2)
    info['deleted_ml'] = round(float(((arr > 0) & (out == 0)).sum()) * vox_ml, 2)
    o = sitk.GetImageFromArray(out)
    o.CopyInformation(m)
    return o, info
