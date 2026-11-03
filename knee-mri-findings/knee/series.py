import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pydicom

FS_TOKENS = {'FS', 'SAT', 'FSAT', 'FATSAT', 'STIR', 'SPIR', 'SPAIR', 'TIRM', 'SPECIAL', 'CHESS'}
FS_PARTS = ('FAT', 'WATER', 'DIXON', 'SPAIR', 'STIR', 'FATSAT', 'FS_', '_FS')
GRE_WORDS = ['FFE', 'GRE', 'SPGR', 'FLASH', 'MPRAGE', 'DESS', 'FIESTA', 'VIBE', 'WATS', 'MEDIC', 'TRUFI', 'MERGE', 'COSMIC']
HCOLS = ['n_slices', 'manufacturer', 'model', 'field', 'laterality', 'scanning_sequence', 'image_type', 'series_desc', 'mr_acq',
         'TR', 'TE', 'thickness', 'tsx', 'plane_hdr', 'fs_hdr', 'error']
STUDY_COLS = ['StudyInstanceUID', 'split', 'study_dir', 'lat_tag', 'site_key', 'ref_uid', 'ref_dir', 'alt_uid', 'alt_dir',
              'any_uid', 'any_dir']
PLANES = {'sag': 'Sagittal', 'cor': 'Coronal', 'axi': 'Axial', 'tra': 'Axial'}


def subdirs(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(f'{p}/{d}')) if os.path.isdir(p) else []


def read_header(sid, uid, d):
    files = [f for f in os.listdir(d) if f.endswith('.dcm')]
    if not files:
        raise FileNotFoundError(f'{sid} {uid}: no .dcm file in {d}')
    h = pydicom.dcmread(os.path.join(d, files[0]), stop_before_pixels=True)

    def g(k):
        return str(getattr(h, k, '')).strip()
    iop = h.get('ImageOrientationPatient')
    plane = None
    if iop is not None and len(iop) == 6:
        v = np.array([float(x) for x in iop])
        plane = ['Sagittal', 'Coronal', 'Axial'][int(np.argmax(np.abs(np.cross(v[:3], v[3:]))))]
    txt = (g('SeriesDescription') + ' ' + g('SequenceName') + ' ' + g('ScanOptions') + ' ' + g('ProtocolName')).upper()
    tok = set(txt.replace('-', ' ').replace('_', ' ').replace('/', ' ').replace('\\', ' ').split())
    fs = int(bool(tok & FS_TOKENS) or any(w in txt for w in FS_PARTS) or 'IR' in re.findall(r'[A-Z]+', g('ScanningSequence').upper()))
    return dict(StudyInstanceUID=sid, SeriesInstanceUID=uid, n_slices=len(files), manufacturer=g('Manufacturer'),
                model=g('ManufacturerModelName'), field=g('MagneticFieldStrength'), laterality=g('Laterality').upper(),
                scanning_sequence=g('ScanningSequence').upper(), image_type=g('ImageType').upper(),
                series_desc=g('SeriesDescription').upper(), mr_acq=g('MRAcquisitionType').upper(),
                TR=g('RepetitionTime'), TE=g('EchoTime'), thickness=g('SliceThickness'),
                tsx=h.file_meta.TransferSyntaxUID.name, plane_hdr=plane, fs_hdr=fs)


def header_row(job):
    sid, uid, d, strict = job
    try:
        return read_header(sid, uid, d)
    except Exception as e:
        if strict:
            raise RuntimeError(f'header of {d}: {e!r}') from e
        return dict(StudyInstanceUID=sid, SeriesInstanceUID=uid, n_slices=0, error=repr(e)[:200])


def mirror_study(job):
    sid, src, dst = job
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    groups = {}
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                u = str(pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['SeriesInstanceUID']).SeriesInstanceUID)
            except Exception:
                continue
            groups.setdefault(u, []).append(p)
    rows = []
    for u, ps in groups.items():
        d = os.path.join(dst, u)
        os.makedirs(d, exist_ok=True)
        for i, p in enumerate(ps):
            os.symlink(p, os.path.join(d, f'{i:05d}.dcm'))
        rows.append(dict(StudyInstanceUID=sid, SeriesInstanceUID=u, series_dir=d))
    return rows


def series_rows(ids, base, csv, split):
    if os.path.exists(csv):
        C = pd.read_csv(csv, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    else:
        C = pd.DataFrame(columns=['StudyInstanceUID', 'SeriesInstanceUID'])
    C = C[C.StudyInstanceUID.isin(set(ids))].drop_duplicates(['StudyInstanceUID', 'SeriesInstanceUID'])
    C = C.drop(columns=[c for c in C.columns if c in HCOLS or c in ('split', 'series_dir')])
    listed = set(C.StudyInstanceUID)
    extra = pd.DataFrame([dict(StudyInstanceUID=s, SeriesInstanceUID=u) for s in ids if s not in listed for u in subdirs(f'{base}/{s}')],
                         columns=['StudyInstanceUID', 'SeriesInstanceUID'])
    S = pd.concat([C, extra], ignore_index=True) if len(extra) else C.reset_index(drop=True)
    S['split'] = split
    S['series_dir'] = [f'{base}/{s}/{u}' for s, u in zip(S.StudyInstanceUID, S.SeriesInstanceUID)]
    return S


def rebuild_folders(TS, ids, test_dir, folders, out):
    def has_dcm(d):
        return os.path.isdir(d) and any(f.endswith('.dcm') for f in os.listdir(d))
    ok = {s for s, g in TS[TS.split == 'test'].groupby('StudyInstanceUID') if any(has_dcm(d) for d in g.series_dir)}
    odd = [s for s in ids if s in folders and s not in ok]
    if not odd:
        return TS, 0
    with ProcessPoolExecutor(4) as pool:
        M = [r for rows in pool.map(mirror_study, [(s, f'{test_dir}/{s}', f'{out}/{s}') for s in odd]) for r in rows]
    M = pd.DataFrame(M, columns=['StudyInstanceUID', 'SeriesInstanceUID', 'series_dir'])
    M = M.merge(TS[TS.StudyInstanceUID.isin(set(odd))].drop(columns=['series_dir', 'split']),
                on=['StudyInstanceUID', 'SeriesInstanceUID'], how='left').assign(split='test')
    return pd.concat([TS[~TS.StudyInstanceUID.isin(set(odd))], M], ignore_index=True), M.StudyInstanceUID.nunique()


def add_headers(TS, strict):
    jobs = list(TS.assign(strict=strict)[['StudyInstanceUID', 'SeriesInstanceUID', 'series_dir', 'strict']].itertuples(index=False, name=None))
    with ProcessPoolExecutor(4) as pool:
        H = pd.DataFrame(list(pool.map(header_row, jobs, chunksize=16)))
    for c in HCOLS:
        if c not in H:
            H[c] = None
    SM = TS.merge(H[['StudyInstanceUID', 'SeriesInstanceUID'] + HCOLS], on=['StudyInstanceUID', 'SeriesInstanceUID'], how='left')
    for c in ('Anatomical_Plane', 'Fat_Suppression'):
        if c not in SM:
            SM[c] = None
    ap = SM.Anatomical_Plane.fillna('').astype(str).str.strip()
    SM['Anatomical_Plane'] = ap.str[:3].str.lower().map(PLANES).where(lambda x: x.notna(), ap.where(ap != '', SM.plane_hdr))
    words = {'true': 1, 'yes': 1, 'false': 0, 'no': 0}
    fs = pd.to_numeric(SM.Fat_Suppression.map(lambda v: words.get(str(v).strip().lower(), v)), errors='coerce')
    SM['Fat_Suppression'] = fs.where(fs.notna(), pd.to_numeric(SM.fs_hdr, errors='coerce')).fillna(0).astype(int)
    if 'Fluid_Sensitive' not in SM:
        SM['Fluid_Sensitive'] = SM.Fat_Suppression
    for c in ('manufacturer', 'model', 'field'):
        SM[c] = SM[c].fillna('').astype(str)
    SM['n_slices'] = pd.to_numeric(SM.n_slices, errors='coerce').fillna(0).astype(int)
    SM['site_key'] = SM.manufacturer + '|' + SM.model + '|' + SM.field
    return SM


def study_table(SM):
    SM = SM.copy()
    SM['lat_norm'] = SM.laterality.map({'R': 'R', 'L': 'L', 'RIGHT': 'R', 'LEFT': 'L'})
    sag = SM[(SM.Anatomical_Plane == 'Sagittal') & SM.error.isna()].copy()
    txt = sag.image_type.fillna('').astype(str) + ' ' + sag.series_desc.fillna('').astype(str)
    seq = sag.scanning_sequence.fillna('').astype(str)
    sag['is_gre'] = (seq.str.contains('GR') & ~seq.str.contains('SE')) | txt.map(lambda s: any(w in s for w in GRE_WORDS)).astype(bool)
    sag['is_3d'] = (sag.mr_acq.fillna('') == '3D') | (sag.n_slices >= 60)
    sag['odd_n'] = (sag.n_slices < 12) | (sag.n_slices > 50)
    sag = sag.sort_values(['StudyInstanceUID', 'is_gre', 'is_3d', 'Fat_Suppression', 'odd_n', 'n_slices'],
                          ascending=[True, True, True, True, True, False])
    sag['rank'] = sag.groupby('StudyInstanceUID').cumcount()
    refs = sag[sag['rank'] == 0].set_index('StudyInstanceUID')
    alts = sag[(sag['rank'] == 1) & ~sag.is_gre].set_index('StudyInstanceUID')
    order = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
    rows = []
    for (sid, split), g in SM.groupby(['StudyInstanceUID', 'split'], sort=False):
        tags = g.lat_norm.dropna().unique()
        ok = g[g.error.isna()].sort_values('Anatomical_Plane', key=lambda s: s.map(order).fillna(3))
        rows.append(dict(StudyInstanceUID=sid, split=split, study_dir=os.path.dirname(g.series_dir.iloc[0]),
                         lat_tag=tags[0] if len(tags) == 1 else '', site_key=g.site_key.mode().iloc[0],
                         ref_uid=refs.SeriesInstanceUID.get(sid, ''), ref_dir=refs.series_dir.get(sid, ''),
                         alt_uid=alts.SeriesInstanceUID.get(sid, ''), alt_dir=alts.series_dir.get(sid, ''),
                         any_uid=ok.SeriesInstanceUID.iloc[0] if len(ok) else '', any_dir=ok.series_dir.iloc[0] if len(ok) else ''))
    return pd.DataFrame(rows, columns=STUDY_COLS)
