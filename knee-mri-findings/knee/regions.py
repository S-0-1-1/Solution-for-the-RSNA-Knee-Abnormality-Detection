import os
import json
import time
import glob
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from knee import bones

KEEP = ('StudyInstanceUID', 'SeriesInstanceUID', 'series_desc', 'mode', 'step', 'spacing', 'size', 'fov_center', 'site_key',
        'ref_flags', 'alt_flags')
ROI_COLS = ['StudyInstanceUID', 'roi', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'source', 'lat', 'lat_source', 'qc_ok',
            'bones_ok', 'mask_stage']
S = {}


def side_vector(j):
    return np.array([1.0 if j.get('lat_final') == 'L' else -1.0, 1.0, 1.0])


def build_atlas(region_jsons, bounds):
    clean = [j for j in region_jsons if j.get('qc_ok') == 1 and j.get('rois') and len(j['rois']) == len(bones.ROIS)]
    off, half, kf = {r: [] for r in bones.ROIS}, {r: [] for r in bones.ROIS}, []
    for j in clean:
        V, K = side_vector(j), np.array(j['K'])
        kf.append((K - np.array(j['fov_center'])) * V)
        for r in bones.ROIS:
            b = np.array(j['rois'][r])
            c = (b[[0, 2, 4]] + b[[1, 3, 5]]) / 2
            off[r].append((c - K) * V)
            half[r].append((b[[1, 3, 5]] - b[[0, 2, 4]]) / 2)
    return dict(studies=len(clean), knee_from_fov=np.median(kf, 0).tolist(), bounds=bounds,
                regions={r: dict(offset=np.median(off[r], 0).tolist(), half=np.median(half[r], 0).tolist()) for r in bones.ROIS})


def header_fov_center(d):
    hs = [pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True) for f in os.listdir(d) if f.endswith('.dcm')]
    shape0 = Counter((int(h.Rows), int(h.Columns)) for h in hs).most_common(1)[0][0]
    hs = [h for h in hs if (int(h.Rows), int(h.Columns)) == shape0]
    iop = np.array(hs[0].ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]
    ipp = np.array([np.array(h.ImagePositionPatient, float) for h in hs])
    w = ipp @ np.cross(r, c)
    ps = [float(v) for v in hs[0].PixelSpacing]
    mid = (ipp[int(np.argmin(w))] + ipp[int(np.argmax(w))]) / 2
    return (mid + r * ps[1] * (shape0[1] - 1) / 2 + c * ps[0] * (shape0[0] - 1) / 2).tolist()


def setup(work, strict):
    st = pd.read_csv(f'{work}/meta/studies.csv', dtype=str, keep_default_na=False)
    S.update(work=work, strict=strict, studies=st,
             tags={s: (v if v in ('L', 'R') else None) for s, v in zip(st.StudyInstanceUID, st.lat_tag)})


def study(sid):
    W, strict, ST, tags = S['work'], S['strict'], S['studies'], S['tags']
    qf, mf = f'{W}/seg/qc/{sid}.json', f'{W}/seg/masks/{sid}.nii.gz'
    if strict and not os.path.exists(mf):
        raise RuntimeError(f'{sid}: no bone mask (no usable sagittal series, or the segmentation failed)')
    q0 = json.load(open(qf)) if os.path.exists(qf) else dict(StudyInstanceUID=sid, flag_reasons='no_segmentation')
    try:
        m1, info = bones.fix_labels(sitk.ReadImage(mf))
        m2, st = bones.clean_mask(m1)
        ext = tags.get(sid)
        q = bones.bone_regions(m2, st, None if info['fibula_found'] else ext)
        q.update(st)
        q.update(info)
        q.update(side_dicom=ext, side_ext_source='dicom' if ext else 'none',
                 side_disagree=int(bool(info['fibula_found'] and ext and q.get('lat_fibula') and q['lat_fibula'] != ext)))
        why = [w for w in q['flag_reasons'].split('|') if w and w not in ('fibula_missing', 'patella_small')]
        notes = []
        if info.get('shaft_added_ml', 0) > 0:
            why = [w for w in why if w not in ('femur_fragmented', 'tibia_fragmented')]
            notes.append('shaft_reattached')
        if not info['femur_top']:
            notes.append('short_femur')
        if not info['tibia_bottom']:
            notes.append('short_tibia')
        if q['side_disagree']:
            notes.append('side_disagree')
        if not info['patella_found']:
            why.append('patella_not_found')
        if not info['fibula_found']:
            why.append('fibula_not_found')
        q['flag_reasons'] = '|'.join(dict.fromkeys(why))
        q['qc_ok'] = int(not why)
        q['notes'] = '|'.join(notes)
        for k in KEEP:
            if k in q0:
                q[k] = q0[k]
        q['stage'] = q0.get('stage', 'ref')
        q['seg_flags'] = q0.get('flag_reasons', '')
        sitk.WriteImage(m2, f'{W}/masks/{sid}.nii.gz', True)
    except Exception as e:
        if strict:
            raise RuntimeError(f'{sid}: {e!r}') from e
        q = dict(StudyInstanceUID=sid, SeriesInstanceUID=q0.get('SeriesInstanceUID'), qc_ok=0, geom_ok=0, flag_reasons='error',
                 error=repr(e)[:200], site_key=q0.get('site_key'), fov_center=q0.get('fov_center'),
                 seg_flags=q0.get('flag_reasons', ''), changed_ml=0.0)
        if tags.get(sid):
            q['lat_final'], q['lat_source'] = tags[sid], 'tag'
    if not q.get('fov_center'):
        r = ST[ST.StudyInstanceUID == sid].iloc[0]
        for d, uid in ((r.ref_dir, r.ref_uid), (r.any_dir, r.any_uid)):
            if not d:
                continue
            if strict:
                q['fov_center'] = header_fov_center(d)
            else:
                try:
                    q['fov_center'] = header_fov_center(d)
                except Exception:
                    continue
            if not q.get('SeriesInstanceUID'):
                q['SeriesInstanceUID'] = uid
            break
    if not q.get('site_key'):
        q['site_key'] = ST.loc[ST.StudyInstanceUID == sid, 'site_key'].iloc[0]
    bones.jdump(q, f'{W}/regions/{sid}.json')
    return sid


def guarded(sid):
    if S['strict']:
        return study(sid)
    try:
        return study(sid)
    except Exception as e:
        bones.jdump(dict(StudyInstanceUID=sid, qc_ok=0, geom_ok=0, flag_reasons='error', error=repr(e)[:200]),
                    f'{S["work"]}/regions/{sid}.json')
        return sid


def run_studies(sids, nproc, strict, work, t0):
    done, remaining = set(), list(sids)
    while remaining:
        try:
            with ProcessPoolExecutor(nproc, initializer=setup, initargs=(work, strict)) as pool:
                for sid in pool.map(guarded, remaining, chunksize=1):
                    done.add(sid)
                    if len(done) % 250 == 0:
                        print(f'[regions] {len(done)}/{len(sids)} | {(time.time() - t0) / 60:.1f} min', flush=True)
            remaining = []
        except BrokenProcessPool:
            if strict:
                raise
            remaining = [x for x in remaining if x not in done]
            for x in remaining[:nproc]:
                try:
                    with ProcessPoolExecutor(1, initializer=setup, initargs=(work, strict)) as p1:
                        p1.submit(guarded, x).result()
                except BrokenProcessPool:
                    bones.jdump(dict(StudyInstanceUID=x, qc_ok=0, geom_ok=0, flag_reasons='error', error='worker died'),
                                f'{work}/regions/{x}.json')
                done.add(x)
            remaining = remaining[nproc:]


def add_flags(J, bounds):
    for j in J.values():
        why = [w for w in str(j.get('flag_reasons') or '').split('|') if w]
        for c, b in (('femur_win_ml', 'femur'), ('tibia_win_ml', 'tibia'), ('patella_ml', 'patella')):
            if j.get(c) is not None and j[c] > bounds[c][1]:
                why.append(f'{b}_big')
        if j.get('fibula_ml') is not None and j['fibula_ml'] > bounds['fibula_ml'][1]:
            j['notes'] = '|'.join([n for n in str(j.get('notes') or '').split('|') if n] + ['fibula_big'])
        w_ = j.get('knee_width_mm')
        if w_ is not None and not (bounds['knee_width_mm'][0] <= w_ <= bounds['knee_width_mm'][1]):
            why.append('knee_width_odd')
        why = list(dict.fromkeys(why))
        j['flag_reasons'], j['qc_ok'] = '|'.join(why), int(not why)
        j['bones_ok'] = int(j.get('geom_ok', 0) == 1 and not any(w.startswith(('femur', 'tibia', 'geom_femur', 'knee_width', 'error'))
                                                                 for w in why))
        j['patella_ok'] = int(j['bones_ok'] == 1 and not any(w.startswith(('patella', 'geom_patella')) for w in why))


def region_boxes(J, atlas):
    KF = np.array(atlas['knee_from_fov'])
    ATL = {r: (np.array(v['offset']), np.array(v['half'])) for r, v in atlas['regions'].items()}
    rows = []
    for sid, j in J.items():
        V, R = side_vector(j), (j.get('rois') or {})
        if j['bones_ok'] and j.get('K'):
            K, ks = np.array(j['K']), 'knee'
        elif j.get('fov_center'):
            K, ks = np.array(j['fov_center']) + KF * V, 'fov'
        else:
            continue
        for r in bones.ROIS:
            if j['bones_ok'] and r in R and (r != 'PF' or j['patella_ok']):
                b, src = R[r], 'bones'
            else:
                o, h = ATL[r]
                c = K + o * V
                b, src = [c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] - h[2], c[2] + h[2]], f'atlas_{ks}'
            rows.append(dict(StudyInstanceUID=sid, roi=r, xmin=b[0], xmax=b[1], ymin=b[2], ymax=b[3], zmin=b[4], zmax=b[5],
                             source=src, lat=j.get('lat_final'), lat_source=j.get('lat_source'), qc_ok=j['qc_ok'],
                             bones_ok=j['bones_ok'], mask_stage=j.get('stage')))
    return pd.DataFrame(rows, columns=ROI_COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--anatomy', required=True)
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--strict', type=int, default=1)
    A = ap.parse_args()
    W, t0 = A.work, time.time()
    for d in (f'{W}/masks', f'{W}/regions'):
        os.makedirs(d, exist_ok=True)
    sids = pd.read_csv(f'{W}/meta/studies.csv', dtype=str, keep_default_na=False).StudyInstanceUID.tolist()
    run_studies(sids, A.nproc, bool(A.strict), W, t0)
    atlas = json.load(open(A.anatomy))
    J = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f'{W}/regions/*.json')}
    add_flags(J, atlas['bounds'])
    for sid, j in J.items():
        bones.jdump(j, f'{W}/regions/{sid}.json')
    RO = region_boxes(J, atlas)
    RO.to_csv(f'{W}/meta/rois.csv', index=False)
    Q = pd.DataFrame(list(J.values()))
    for c in ('SeriesInstanceUID', 'site_key'):
        if c not in Q:
            Q[c] = None
    Q.drop(columns=[c for c in ('rois', 'K', 'fov_center', 'spacing', 'size') if c in Q]).to_csv(f'{W}/meta/regions.csv', index=False)
    n_bones = int(Q.bones_ok.sum()) if 'bones_ok' in Q else 0
    print(f'[regions] done: {len(J)} studies in {(time.time() - t0) / 60:.1f} min | bone boxes for {n_bones}', flush=True)


if __name__ == '__main__':
    main()
