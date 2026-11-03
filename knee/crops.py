import os
import io
import json
import zlib
import warnings
from types import SimpleNamespace
from collections import Counter
import numpy as np
import pandas as pd
import pydicom
import cv2
from scipy import ndimage as ndi
from PIL import Image, features

VERSION = 2
OUT_PX, MAX_SLICES, MIN_SLICES = 128, 24, 3
MAX_LARGE = 16
LARGE_ROIS = ('joint', 'bones', 'popliteal', 'MCL')
MARGIN_MM, SLAB_MM, THIN_MM = 4.0, 3.0, 2.0
MIN_SERIES_IMAGES = 8
ROIS = ['medial_TF', 'lateral_TF', 'notch', 'PF', 'MCL', 'joint', 'popliteal', 'bones']
SLOTS = ['sag_nf', 'sag_fs', 'cor_fs', 'cor_nf', 'ax_fs']
VIEW = {'sag_nf': 'sag', 'sag_fs': 'sag', 'cor_fs': 'cor', 'cor_nf': 'cor', 'ax_fs': 'ax'}
CROPS = [('medial_TF', 'sag_nf'), ('medial_TF', 'sag_fs'), ('medial_TF', 'cor_fs'),
         ('lateral_TF', 'sag_nf'), ('lateral_TF', 'sag_fs'), ('lateral_TF', 'cor_fs'),
         ('notch', 'sag_nf'), ('notch', 'sag_fs'), ('notch', 'cor_fs'),
         ('MCL', 'cor_fs'), ('MCL', 'ax_fs'),
         ('PF', 'ax_fs'), ('PF', 'sag_fs'), ('PF', 'sag_nf'),
         ('popliteal', 'ax_fs'), ('popliteal', 'sag_fs'),
         ('joint', 'sag_fs'), ('joint', 'ax_fs'),
         ('bones', 'cor_fs'), ('bones', 'sag_fs'), ('bones', 'cor_nf')]
CROP_NAMES = [f'{r}__{s}' for r, s in CROPS]
ANCHOR = {'sag_nf': 'medial_TF', 'sag_fs': 'bones', 'cor_fs': 'bones', 'cor_nf': 'bones', 'ax_fs': 'joint'}
GRE_WORDS = ('FFE', 'GRE', 'SPGR', 'FLASH', 'MPRAGE', 'DESS', 'FIESTA', 'VIBE', 'WATS', 'MEDIC', 'TRUFI', 'MERGE',
             'COSMIC', 'T2*', 'T2_STAR', 'HEMO')
DERIVED_WORDS = ('MPR', 'SECONDARY', 'PROJECTION', 'REFORMAT', 'MIP', 'SUBTRACT')
WEBP = bool(features.check('webp'))


def _col(df, name, default=''):
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


def rank_series(S):
    plane = _col(S, 'Anatomical_Plane').fillna('').astype(str).str.strip().str[:3].str.lower()
    fs = pd.to_numeric(_col(S, 'Fat_Suppression', 0), errors='coerce').fillna(0).astype(int)
    txt = (_col(S, 'image_type').fillna('').astype(str) + ' ' + _col(S, 'series_desc').fillna('').astype(str)).str.upper()
    seq = _col(S, 'scanning_sequence').fillna('').astype(str).str.upper()
    tr = pd.to_numeric(_col(S, 'TR', np.nan), errors='coerce')
    te = pd.to_numeric(_col(S, 'TE', np.nan), errors='coerce')
    n = pd.to_numeric(_col(S, 'n_slices', 0), errors='coerce').fillna(0)
    acq = _col(S, 'mr_acq').fillna('').astype(str).str.upper()
    err = _col(S, 'error').fillna('').astype(str).str.strip().str.len() > 0
    gre = (seq.str.contains('GR') & ~seq.str.contains('SE')) | txt.apply(lambda t: any(w in t for w in GRE_WORDS))
    der = txt.apply(lambda t: any(w in t for w in DERIVED_WORDS))
    is3d = (acq == '3D') | (n >= 60)
    t1 = (tr < 1000).values
    pdw = ((tr >= 1000) & (te <= 60)).values
    unk = tr.isna().values
    stir = seq.str.contains('IR').values
    out = {}
    for slot in SLOTS:
        want_plane = {'sag': 'sag', 'cor': 'cor', 'ax': 'axi'}[VIEW[slot]]
        m = ((plane == want_plane) & (fs == int(slot.endswith('_fs')))).values
        if not m.any():
            continue
        if slot == 'sag_nf':
            pref = np.where(pdw, 0, np.where(t1 | unk, 1, 2))
        elif slot == 'cor_nf':
            pref = np.where(t1, 0, np.where(pdw | unk, 1, 2))
        else:
            pref = np.where(stir, 1, 0)
        K = pd.DataFrame({'uid': S.SeriesInstanceUID.astype(str).values, 'err': err.values, 'gre': gre.values,
                          'der': der.values, 'is3d': is3d.values, 'pref': pref, 'few': (n < 10).values,
                          'neg_n': -n.values})[m]
        out[slot] = K.sort_values(['err', 'gre', 'der', 'is3d', 'pref', 'few', 'neg_n'], kind='stable').uid.tolist()
    return out


def _f(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _seq0(ds, name):
    try:
        v = ds.get(name)
        return v[0] if v is not None and len(v) else None
    except Exception:
        return None


def _decode(ds, path):
    try:
        return ds.pixel_array, False
    except Exception:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(path)), True


def _frame_geom(ds, k, nf):
    if nf == 1 and 'ImagePositionPatient' in ds:
        ipp, iop, ps = ds.get('ImagePositionPatient'), ds.get('ImageOrientationPatient'), ds.get('PixelSpacing')
        slope, inter = _f(ds.get('RescaleSlope', 1), 1.0), _f(ds.get('RescaleIntercept', 0), 0.0)
    else:
        per, sh = ds.get('PerFrameFunctionalGroupsSequence'), _seq0(ds, 'SharedFunctionalGroupsSequence')
        g = per[k] if per is not None and k < len(per) else None

        def grp(name):
            for src in (g, sh):
                v = _seq0(src, name) if src is not None else None
                if v is not None:
                    return v
            return None
        pp, po = grp('PlanePositionSequence'), grp('PlaneOrientationSequence')
        pm, pv = grp('PixelMeasuresSequence'), grp('PixelValueTransformationSequence')
        ipp = pp.get('ImagePositionPatient') if pp is not None else None
        iop = po.get('ImageOrientationPatient') if po is not None else ds.get('ImageOrientationPatient')
        ps = pm.get('PixelSpacing') if pm is not None else ds.get('PixelSpacing')
        src = pv if pv is not None else ds
        slope, inter = _f(src.get('RescaleSlope', 1), 1.0), _f(src.get('RescaleIntercept', 0), 0.0)
    if ipp is None or iop is None or ps is None:
        return None
    try:
        ipp, iop, ps = (np.array([float(v) for v in a]) for a in (ipp, iop, ps))
    except Exception:
        return None
    if ipp.shape != (3,) or iop.shape != (6,) or ps.shape != (2,) or ps.min() <= 0:
        return None
    return ipp, iop, ps, (slope or 1.0), (inter or 0.0)


def load_series(d):
    files = sorted(f for f in os.listdir(d) if f.lower().endswith('.dcm'))
    if not files:
        raise FileNotFoundError(f'no .dcm files in {d}')
    frames, tsx = [], Counter()
    info = dict(files=len(files), fallback=0, multiframe=0, skipped=0, mono1=0)
    for f in files:
        p = os.path.join(d, f)
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            info['skipped'] += 1
            continue
        try:
            tsx[str(ds.file_meta.TransferSyntaxUID.name)] += 1
        except Exception:
            pass
        try:
            a, rescaled = _decode(ds, p)
        except Exception:
            info['skipped'] += 1
            continue
        a = np.asarray(a)
        if a.ndim == 2:
            a = a[None]
        if a.ndim != 3:
            info['skipped'] += 1
            continue
        info['fallback'] += int(rescaled)
        info['multiframe'] += int(a.shape[0] > 1)
        mono1 = str(ds.get('PhotometricInterpretation', '')).upper() == 'MONOCHROME1'
        echo, inst = _f(ds.get('EchoTime', None), 0.0) or 0.0, _f(ds.get('InstanceNumber', None), 0.0) or 0.0
        thk = _f(ds.get('SliceThickness', None), None)
        for k in range(a.shape[0]):
            g = _frame_geom(ds, k, a.shape[0])
            if g is None:
                info['skipped'] += 1
                continue
            ipp, iop, ps, slope, inter = g
            if rescaled:
                slope, inter = 1.0, 0.0
            frames.append(dict(px=a[k], ipp=ipp, iop=iop, ps=ps, slope=slope, inter=inter, mono1=mono1, echo=echo,
                               inst=inst * 10000 + k, thk=thk))
    if not frames:
        raise ValueError('no readable image in the series')

    def key(x):
        return x['px'].shape, tuple(np.round(x['iop'], 2))
    best, count = Counter(key(x) for x in frames).most_common(1)[0]
    if count < MIN_SERIES_IMAGES:
        raise ValueError(f'only {count} images with one geometry (localizer?)')
    frames = [x for x in frames if key(x) == best]
    iop = frames[0]['iop']
    r, c = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    n = np.cross(r, c)
    n = n / np.linalg.norm(n)
    pos = np.round(np.array([x['ipp'] @ n for x in frames]), 1)
    if len(np.unique(pos)) < len(pos):
        E = {}
        for x, q in zip(frames, pos):
            E.setdefault(x['echo'], set()).add(q)
        if len(E) > 1:
            e0 = sorted(E, key=lambda e: (-len(E[e]), e))[0]
            keep = [i for i, x in enumerate(frames) if x['echo'] == e0]
            frames, pos = [frames[i] for i in keep], pos[keep]
        seen, keep = set(), []
        for i in sorted(range(len(frames)), key=lambda i: (pos[i], frames[i]['inst'])):
            if pos[i] not in seen:
                seen.add(pos[i])
                keep.append(i)
        info['duplicates_dropped'] = len(frames) - len(keep)
        frames = [frames[i] for i in keep]
    w = np.array([x['ipp'] @ n for x in frames])
    o = np.argsort(w, kind='stable')
    frames, w = [frames[i] for i in o], w[o]
    vol = np.stack([x['px'].astype(np.float32) * np.float32(x['slope']) + np.float32(x['inter']) for x in frames])
    if sum(x['mono1'] for x in frames) * 2 > len(frames):
        vol = vol.max() - vol
    info.update(used=len(frames), tsx='|'.join(sorted(tsx)), shape=list(vol.shape))
    return SimpleNamespace(vol=vol, ipp=np.array([x['ipp'] for x in frames]), r=r, c=c, n=n, w=w,
                           dy=float(frames[0]['ps'][0]), dx=float(frames[0]['ps'][1]),
                           dz=float(np.median(np.diff(w))) if len(w) > 1 else float(frames[0]['thk'] or 3.0),
                           info=info, uid=os.path.basename(os.path.normpath(d)))


def load_mask(path):
    import SimpleITK as sitk
    m = sitk.ReadImage(path)
    return dict(arr=sitk.GetArrayFromImage(m).astype(np.uint8), D=np.array(m.GetDirection(), float).reshape(3, 3),
                o=np.array(m.GetOrigin(), float), sp=np.array(m.GetSpacing(), float))


def mask_at(M, pts):
    q = ((pts - M['o']) @ M['D']) / M['sp']
    idx = np.rint(q).astype(np.int64)
    size = np.array(M['arr'].shape[::-1])
    ok = np.all((idx >= 0) & (idx < size), axis=-1)
    out = np.zeros(pts.shape[:-1], np.uint8)
    ii = idx[ok]
    out[ok] = M['arr'][ii[:, 2], ii[:, 1], ii[:, 0]]
    return out


class PlaneError(ValueError):
    pass


def canon_axes(view, s):
    med, inf, post = np.array([-s, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
    return {'sag': (inf, post, med), 'cor': (inf, med, post), 'ax': (post, med, inf)}[view]


def frame_for(V, view, s):
    rt, ct, tt = canon_axes(view, s)
    cos = float(V.n @ tt)
    if abs(cos) < 0.7:
        raise PlaneError(f'series plane is not {view} (normal . expected = {cos:.2f})')
    nc = V.n * np.sign(cos)
    er = rt - (rt @ V.n) * V.n
    er = er / np.linalg.norm(er)
    ec = ct - (ct @ V.n) * V.n
    ec = ec - (ec @ er) * er
    ec = ec / np.linalg.norm(ec)
    return er, ec, nc


def corners(b):
    return np.array([[x, y, z] for x in b[0:2] for y in b[2:4] for z in b[4:6]], float)


def _sample(V, k, C, er, ec, g, sig):
    P = V.ipp[k]
    base = (C - ((C - P) @ V.n) * V.n) - P
    jj = (base @ V.r + g[:, None] * (er @ V.r) + g[None, :] * (ec @ V.r)) / V.dx
    ii = (base @ V.c + g[:, None] * (er @ V.c) + g[None, :] * (ec @ V.c)) / V.dy
    img = V.vol[k]
    H, W = img.shape
    inside = (ii >= -0.5) & (ii <= H - 0.5) & (jj >= -0.5) & (jj <= W - 0.5)
    if not inside.any():
        return np.full(ii.shape, np.nan, np.float32)
    if sig is not None:
        pad = int(4 * max(sig)) + 2
        i0, i1 = max(int(np.floor(ii[inside].min())) - pad, 0), min(int(np.ceil(ii[inside].max())) + pad + 1, H)
        j0, j1 = max(int(np.floor(jj[inside].min())) - pad, 0), min(int(np.ceil(jj[inside].max())) + pad + 1, W)
        sub = ndi.gaussian_filter(img[i0:i1, j0:j1], sig, mode='nearest')
        v = ndi.map_coordinates(sub, [np.clip(ii - i0, 0, i1 - i0 - 1), np.clip(jj - j0, 0, j1 - j0 - 1)], order=1, mode='nearest')
    else:
        v = ndi.map_coordinates(img, [np.clip(ii, 0, H - 1), np.clip(jj, 0, W - 1)], order=1, mode='nearest')
    return np.where(inside, v, np.nan).astype(np.float32)


def crop(V, box, view, s, M=None, out=OUT_PX, margin=MARGIN_MM, max_slices=None, aa=True):
    max_slices = max_slices or MAX_SLICES
    er, ec, nc = frame_for(V, view, s)
    Q = corners(box)
    C = Q.mean(0)
    half = max(np.abs((Q - C) @ er).max(), np.abs((Q - C) @ ec).max()) + margin
    d = 2.0 * half / out
    g = (np.arange(out) - (out - 1) / 2.0) * d
    t = Q @ nc
    t0, t1, tc = t.min() - margin, t.max() + margin, float(C @ nc)
    wc = V.ipp @ nc
    order = np.argsort(wc, kind='stable')
    ws = wc[order]
    cover = float(np.clip((min(t1, ws[-1]) - max(t0, ws[0])) / (t1 - t0), 0, 1))
    n_in = int(((ws >= t0) & (ws <= t1)).sum())
    n_full = max(n_in, min(MIN_SLICES, len(wc))) if V.dz >= THIN_MM else max(1, int(round((t1 - t0) / SLAB_MM)))
    sig = None
    if aa:
        fi, fj = d / V.dy, d / V.dx
        si, sj = (0.5 * np.sqrt(f * f - 1) if f > 1.2 else 0.0 for f in (fi, fj))
        if si > 0 or sj > 0:
            sig = (si, sj)
    if V.dz >= THIN_MM:
        sel = order[(ws >= t0) & (ws <= t1)]
        if len(sel) < MIN_SLICES:
            dist = np.maximum(np.maximum(t0 - wc, wc - t1), 0)
            sel = np.argsort(dist, kind='stable')[:min(MIN_SLICES, len(wc))]
            sel = sel[np.argsort(wc[sel], kind='stable')]
        if len(sel) > max_slices:
            sel = sel[np.round(np.linspace(0, len(sel) - 1, max_slices)).astype(int)]
        groups, centers, mode = [[int(k)] for k in sel], wc[sel], '2d'
    else:
        K = max(1, int(round((t1 - t0) / SLAB_MM)))
        K = min(K, max_slices)
        step = (t1 - t0) / K
        centers, groups, mode = t0 + step * (np.arange(K) + 0.5), [], 'slab'
        for cp in centers:
            mem = np.where(np.abs(wc - cp) <= step / 2)[0]
            if not len(mem):
                k = int(np.argmin(np.abs(wc - cp)))
                mem = np.array([k]) if abs(wc[k] - cp) <= step else np.array([], int)
            groups.append([int(k) for k in mem])
        keep = [i for i, gr in enumerate(groups) if gr]
        if not keep:
            k = int(np.argmin(np.abs(wc - tc)))
            groups, centers, keep = [[k]], np.array([wc[k]]), [0]
        groups, centers = [groups[i] for i in keep], np.asarray(centers)[keep]
    imgs, msks = [], []
    for grp, cp in zip(groups, centers):
        if grp:
            st = np.stack([_sample(V, k, C, er, ec, g, sig) for k in grp])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                imgs.append(np.nanmean(st, axis=0) if len(grp) > 1 else st[0])
        else:
            imgs.append(np.full((out, out), np.nan, np.float32))
        if M is not None:
            pts = (C + (cp - tc) * nc) + g[:, None, None] * er + g[None, :, None] * ec
            msks.append(mask_at(M, pts))
    img = np.stack(imgs).astype(np.float32)
    pos = (np.asarray(centers, float) - tc).astype(np.float32)
    meta = dict(d_mm=round(float(d), 4), slice_mm=round(float(np.median(np.diff(centers))) if len(centers) > 1 else float(V.dz), 3),
                mode=mode, n=int(len(centers)), n_in_range=n_in, n_full=int(n_full), cover=round(cover, 3),
                fov=round(float(np.isfinite(img).mean()), 3), aa=None if sig is None else [round(float(x), 2) for x in sig])
    return img, (np.stack(msks) if M is not None else None), pos, meta


def norm_stats(V, box, view, s):
    v = np.array([])
    if box is not None:
        try:
            img = crop(V, box, view, s, None, out=64, margin=0.0, max_slices=16, aa=False)[0]
            v = img[np.isfinite(img)]
        except Exception:
            v = np.array([])
    src = 'bones_box'
    if v.size < 500:
        v, src = V.vol[:, ::4, ::4].ravel(), 'whole_series'
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-3:
        hi = lo + 1.0
    fg = v[v > lo + 0.05 * (hi - lo)]
    p50 = float(np.median(fg)) if fg.size else float(hi)
    return float(lo), float(hi), src, p50


def to_u8(img, lo, hi):
    x = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return np.where(np.isfinite(x), x * 255.0 + 0.5, 0).astype(np.uint8)


def enc_stack(u8):
    buf = io.BytesIO()
    im = Image.fromarray(np.ascontiguousarray(u8.reshape(-1, u8.shape[2])), 'L')
    if WEBP:
        im.save(buf, 'WEBP', lossless=True, quality=10, method=1)
    else:
        im.save(buf, 'PNG', compress_level=6)
    return np.frombuffer(buf.getvalue(), np.uint8)


def dec_stack(b, n):
    a = cv2.imdecode(np.frombuffer(np.asarray(b, np.uint8).tobytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError('could not decode a crop image')
    return a.reshape(n, -1, a.shape[1])


def decode(z, meta=None):
    keys = list(z.keys()) if hasattr(z, 'keys') else list(z.files)
    if meta is None:
        meta = json.loads(str(z['meta']))
    A = {}
    for name, cm in meta['crops'].items():
        n = int(cm['n'])
        if f'{name}__imgw' in keys:
            A[f'{name}__img'] = dec_stack(z[f'{name}__imgw'], n)
        elif f'{name}__img' in keys:
            A[f'{name}__img'] = np.asarray(z[f'{name}__img'])
        if f'{name}__mskz' in keys:
            A[f'{name}__msk'] = np.frombuffer(zlib.decompress(np.asarray(z[f'{name}__mskz'], np.uint8).tobytes()),
                                              np.uint8).reshape(n, OUT_PX, OUT_PX)
        elif f'{name}__msk' in keys:
            A[f'{name}__msk'] = np.asarray(z[f'{name}__msk'])
        if f'{name}__pos' in keys:
            A[f'{name}__pos'] = np.asarray(z[f'{name}__pos'])
    return A, meta


def read_study(path):
    with np.load(path) as z:
        return decode(z)


def _centroid(m, lab, min_px=150):
    z = np.argwhere(m == lab)
    return z.mean(0) if len(z) >= min_px else None


def orient_checks(msk, view, roi):
    c = {b: _centroid(msk, lab) for lab, b in ((1, 'fem'), (2, 'pat'), (3, 'fib'), (4, 'tib'))}
    out = {}
    if view in ('sag', 'cor') and c['fem'] is not None and c['tib'] is not None:
        out['femur_above_tibia'] = bool(c['fem'][1] < c['tib'][1])
    if c['pat'] is not None and c['fem'] is not None:
        if view == 'sag':
            out['patella_anterior'] = bool(c['pat'][2] < c['fem'][2])
        if view == 'ax':
            out['patella_anterior'] = bool(c['pat'][1] < c['fem'][1])
    if c['fib'] is not None and c['tib'] is not None:
        if view in ('cor', 'ax'):
            out['fibula_lateral'] = bool(c['fib'][2] < c['tib'][2])
        if view == 'sag':
            out['fibula_lateral'] = bool(c['fib'][0] < c['tib'][0])
    if roi == 'medial_TF':
        out['no_fibula_medial'] = bool((msk == 3).sum() < 150)
    return out


def align_shift(u8, msk, d, valid=None, max_mm=8.0):
    b = msk > 0
    ok = [k for k in range(len(b)) if b[k].sum() >= 200]
    if len(ok) < 2:
        return None
    R = max(1, int(round(max_mm / d)))
    H, W = u8.shape[1:]
    N = (H + 2 * R, W + 2 * R)
    frame = np.zeros((H, W), bool)
    frame[2:-2, 2:-2] = True
    acc, nb = 0, 0
    for k in ok:
        im = ndi.gaussian_filter(u8[k].astype(np.float32), 1.0)
        G = np.hypot(ndi.sobel(im, 0), ndi.sobel(im, 1))
        if valid is not None:
            G[~ndi.binary_erosion(valid[k], iterations=3)] = 0
        B = (b[k] & ~ndi.binary_erosion(b[k]) & frame).astype(np.float32)
        acc = acc + np.fft.rfft2(G, N) * np.conj(np.fft.rfft2(B, N))
        nb += B.sum()
    cc = np.fft.irfft2(acc, N) / max(nb, 1.0)
    sc = cc[np.ix_(np.arange(-R, R + 1) % N[0], np.arange(-R, R + 1) % N[1])]
    i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
    s0 = sc[R, R]
    return dict(dr_mm=round(float((i - R) * d), 2), dc_mm=round(float((j - R) * d), 2),
                gain=round(float(sc[i, j] / s0), 4) if s0 > 0 else None)


def _seq_info(series_rows, uid):
    try:
        r = series_rows[series_rows.SeriesInstanceUID.astype(str) == uid].iloc[0]
    except Exception:
        return {}

    def g(k):
        return r[k] if k in r.index else None
    tr, te = _f(g('TR')), _f(g('TE'))
    txt = f"{g('image_type') or ''} {g('series_desc') or ''}".upper()
    seq = str(g('scanning_sequence') or '').upper()
    return dict(desc=str(g('series_desc'))[:40], tr=tr, te=te,
                gre=bool(('GR' in seq and 'SE' not in seq) or any(w in txt for w in GRE_WORDS)),
                kind=None if tr is None or tr != tr else ('T1' if tr < 1000 else 'PD' if (te is not None and te <= 60) else 'T2'))


def study_crops(sid, series_rows, boxes, lat, series_dir, mask_path=None, ref_uid=None, box_src=None, crops=CROPS):
    s = 1.0 if lat == 'L' else -1.0
    M = load_mask(mask_path) if mask_path and os.path.exists(mask_path) else None
    ranked = rank_series(series_rows)
    A, used = {}, set()
    meta = dict(StudyInstanceUID=sid, version=VERSION, lat=lat, side_sign=s, mask=M is not None, slots={}, crops={}, errors=[])
    for slot in SLOTS:
        need = [r for r, sl in crops if sl == slot and boxes.get(r) is not None]
        if not need:
            continue
        labelled = [u for u in ranked.get(slot, [])[:3] if u not in used]
        others = [u for sl2, lst in ranked.items()
                  if sl2[-3:] == slot[-3:] and VIEW[sl2] != VIEW[slot]
                  for u in lst if u not in used and u not in labelled][:2]
        cands = labelled + others
        if not cands:
            meta['slots'][slot] = dict(missing='no series of this kind')
            continue
        V, tried = None, []
        for uid in cands:
            try:
                V = load_series(os.path.join(series_dir, uid))
                frame_for(V, VIEW[slot], s)
                break
            except Exception as e:
                tried.append(f'{uid[-10:]}: {type(e).__name__}: {str(e)[:100]}')
                V = None
        if V is None:
            if labelled:
                meta['slots'][slot] = dict(missing='no usable series', tried=tried)
                meta['errors'].append(f'{slot}: no usable series ({"; ".join(tried)})')
            else:
                meta['slots'][slot] = dict(missing='no series of this kind')
            continue
        used.add(V.uid)
        lo, hi, nsrc, p50 = norm_stats(V, boxes.get('bones'), VIEW[slot], s)
        sm = dict(uid=V.uid, rank=cands.index(V.uid), relabelled=V.uid not in ranked.get(slot, []), lo=round(lo, 3),
                  hi=round(hi, 3), p50=round(p50, 3), norm=nsrc, dz=round(V.dz, 3), dy=round(V.dy, 4), dx=round(V.dx, 4),
                  is_ref=bool(ref_uid is not None and V.uid == ref_uid), info=V.info, failed_before=tried,
                  seq=_seq_info(series_rows, V.uid))
        for roi in need:
            name = f'{roi}__{slot}'
            try:
                img, msk, pos, cm = crop(V, boxes[roi], VIEW[slot], s, M, max_slices=MAX_LARGE if roi in LARGE_ROIS else MAX_SLICES)
                u8 = to_u8(img, lo, hi)
                A[f'{name}__imgw'], A[f'{name}__pos'] = enc_stack(u8), pos.astype(np.float16)
                if msk is not None:
                    A[f'{name}__mskz'] = np.frombuffer(zlib.compress(np.ascontiguousarray(msk).tobytes(), 6), np.uint8)
                    cm['checks'] = orient_checks(msk, VIEW[slot], roi)
                    cm['bone_px'] = {b: int((msk == lab).sum()) for lab, b in ((1, 'femur'), (2, 'patella'), (3, 'fibula'), (4, 'tibia'))}
                    if roi == ANCHOR[slot]:
                        sm['align'] = align_shift(u8, msk, cm['d_mm'], np.isfinite(img))
                cm.update(series=V.uid, slot=slot, roi=roi, box_source=(box_src or {}).get(roi))
                meta['crops'][name] = cm
            except Exception as e:
                meta['errors'].append(f'{name}: {type(e).__name__}: {str(e)[:150]}')
        meta['slots'][slot] = sm
        del V
    return A, meta


def js(o):
    return o.item() if hasattr(o, 'item') else str(o)
