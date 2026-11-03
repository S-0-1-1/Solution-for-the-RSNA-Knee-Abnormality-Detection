import os
import json
import time
import glob
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from knee import bones

ap = argparse.ArgumentParser()
ap.add_argument('--table', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--deadline', type=float, default=1e18)
ap.add_argument('--strict', type=int, default=1)
A = ap.parse_args()
torch.backends.cudnn.benchmark = True
torch.set_num_threads(bones.THREADS)
MASKS, QC = f'{A.out}/masks', f'{A.out}/qc'
for d in (MASKS, QC):
    os.makedirs(d, exist_ok=True)
plans = json.load(open(f'{A.model}/plans.json'))
sp = plans['configurations']['3d_fullres']['spacing']
TGT = [0.0] * 3
for i, t in enumerate(plans['transpose_forward']):
    TGT[2 - t] = float(sp[i])
TGT = tuple(TGT)
ISO_SCALE, STEP = 0.7, 1.0
fold = sorted(os.path.basename(f) for f in glob.glob(f'{A.model}/fold_*'))[0].replace('fold_', '')
cuda = torch.cuda.is_available()
predictor = nnUNetPredictor(tile_step_size=STEP, use_gaussian=True, use_mirroring=False, perform_everything_on_device=cuda,
                            device=torch.device('cuda', 0) if cuda else torch.device('cpu'), verbose=False,
                            verbose_preprocessing=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(A.model, use_folds=(int(fold),) if fold.isdigit() else (fold,),
                                               checkpoint_name='checkpoint_final.pth')


def predict(las):
    geo = bones.resample_to(las, tuple(t / ISO_SCALE for t in TGT), sitk.sitkLinear)
    props = {'sitk_stuff': {'spacing': TGT, 'origin': geo.GetOrigin(), 'direction': geo.GetDirection()}, 'spacing': list(TGT)[::-1]}
    seg = predictor.predict_single_npy_array(sitk.GetArrayFromImage(geo)[None].astype(np.float32), props, None, None, False)
    si = sitk.GetImageFromArray(seg.astype(np.uint8))
    si.CopyInformation(geo)
    out = sitk.Resample(si, las, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    if cuda:
        torch.cuda.empty_cache()
    return out


def run_one(sid, d, uid, tag):
    t0 = time.time()
    img, desc = bones.series_image(d)
    las = bones.to_las(img)
    t_load = time.time() - t0
    m, st = bones.clean_mask(predict(las))
    q = bones.bone_regions(m, st, tag)
    q.update(st)
    q.update(StudyInstanceUID=sid, SeriesInstanceUID=uid, series_desc=desc, mode='iso', step=STEP,
             spacing=[float(v) for v in las.GetSpacing()], size=[int(v) for v in las.GetSize()], fov_center=bones.fov_center(las),
             t_load=round(t_load, 1), t_total=round(time.time() - t0, 1))
    return m, q


def attempt(sid, d, uid, tag, stage):
    try:
        m, q = run_one(sid, d, uid, tag)
        q['stage'] = stage
        return m, q
    except Exception as e:
        if A.strict:
            raise RuntimeError(f'{sid} {uid}: {e!r}') from e
        if cuda:
            torch.cuda.empty_cache()
        return None, dict(StudyInstanceUID=sid, SeriesInstanceUID=uid, qc_ok=0, geom_ok=0, flag_reasons='error', error=repr(e)[:300],
                          stage=stage)


TB = pd.read_csv(A.table, dtype=str, keep_default_na=False).sort_values('StudyInstanceUID').reset_index(drop=True)
TB = TB[TB.index % A.nshards == A.shard]
TB = TB[[not os.path.exists(f'{QC}/{s}.json') for s in TB.StudyInstanceUID]]
print(f'[segment {A.shard}] {len(TB)} studies', flush=True)
t0, n = time.time(), 0
for r in TB.itertuples(index=False):
    if time.time() > A.deadline:
        print(f'[segment {A.shard}] time limit after {n} studies', flush=True)
        break
    sid, tag = r.StudyInstanceUID, (r.lat_tag if r.lat_tag in ('L', 'R') else None)
    m, q = attempt(sid, r.ref_dir, r.ref_uid, tag, 'ref')
    if not q.get('qc_ok') and r.alt_dir:
        m2, q2 = attempt(sid, r.alt_dir, r.alt_uid, tag, 'alt')
        if m2 is not None and bones.better(q2, q):
            q2['ref_flags'] = q.get('flag_reasons')
            m, q = m2, q2
        else:
            q['alt_flags'] = q2.get('flag_reasons')
    if m is not None:
        sitk.WriteImage(m, f'{MASKS}/{sid}.nii.gz', True)
    q['site_key'] = r.site_key
    bones.jdump(q, f'{QC}/{sid}.json')
    n += 1
    if n % 25 == 0 or n == len(TB):
        el = (time.time() - t0) / n
        print(f'[segment {A.shard}] {n}/{len(TB)} | {el:.1f} s/study | {el * (len(TB) - n) / 60:.0f} min left', flush=True)
print(f'[segment {A.shard}] done: {n} studies in {(time.time() - t0) / 60:.1f} min', flush=True)
