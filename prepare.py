import os
import glob
import time
import argparse
import pandas as pd
import torch
from knee import series
from knee.launch import worker, env, run_parallel

REPORT_R = (r'right knee|rodilla derecha|sağ diz|desno koljeno|desnog koljena|δεξι[όο] γόνατο|δεξιού γόνατος|rechtes knie|'
            r'rechten knies|knie rechts|rechter knie|rechterknie|дясно коляно|дясна колянна|genou droit')
REPORT_L = (r'left knee|rodilla izquierda|sol diz|lijevo koljeno|lijevog koljena|αριστερ[όο] γόνατο|αριστερού γόνατος|linkes knie|'
            r'linken knies|knie links|linker knie|linkerknie|ляво коляно|лява колянна|genou gauche')


def report_sides(train_csv):
    T = pd.read_csv(train_csv, usecols=['StudyInstanceUID', 'Report'], dtype=str)
    rep = T.Report.fillna('')
    r, l = rep.str.contains(REPORT_R, case=False, regex=True), rep.str.contains(REPORT_L, case=False, regex=True)
    return pd.DataFrame({'StudyInstanceUID': T.StudyInstanceUID, 'side': ['R' if a and not b else 'L' if b and not a else ''
                                                                          for a, b in zip(r, l)]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--segmentation', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hours', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + a.hours * 3600 if a.hours else 1e18
    W = os.path.abspath(a.out)
    for d in (W, f'{W}/meta', f'{W}/seg', f'{W}/nnunet'):
        os.makedirs(d, exist_ok=True)
    ids = pd.read_csv(f'{a.data}/train.csv', dtype=str).StudyInstanceUID.tolist()
    ids = ids[:a.limit] if a.limit else ids
    TS = series.series_rows(ids, f'{a.data}/train_series', f'{a.data}/train_series.csv', 'train')
    SM = series.add_headers(TS, False)
    SM.drop(columns=['series_dir', 'split', 'plane_hdr', 'fs_hdr']).to_csv(f'{W}/meta/series_meta.csv', index=False)
    studies = series.study_table(SM)
    studies.to_csv(f'{W}/meta/studies.csv', index=False)
    sides = report_sides(f'{a.data}/train.csv')
    sides.to_csv(f'{W}/meta/report_sides.csv', index=False)
    print(f'{len(studies)} studies | series {len(SM)} | header errors {int(SM.error.notna().sum())} | '
          f'no sagittal {int((studies.ref_dir == "").sum())} | side stated in the report {int((sides.side != "").sum())}', flush=True)
    n_gpu = torch.cuda.device_count()
    nw = 2 * max(1, n_gpu)
    seg_table = f'{W}/meta/seg_table.csv'
    todo = studies[studies.ref_dir != '']
    todo[['StudyInstanceUID', 'ref_dir', 'ref_uid', 'alt_dir', 'alt_uid', 'lat_tag', 'site_key']].to_csv(seg_table, index=False)
    nn = dict(KNEE_THREADS=1, nnUNet_raw=f'{W}/nnunet', nnUNet_preprocessed=f'{W}/nnunet', nnUNet_results=f'{W}/nnunet')
    run_parallel([(worker('segment', '--table', seg_table, '--model', a.segmentation, '--out', f'{W}/seg', '--shard', i,
                          '--nshards', nw, '--deadline', deadline, '--strict', 0),
                   env(CUDA_VISIBLE_DEVICES=i % n_gpu, **nn) if n_gpu else env(**nn), f'{W}/seg/log{i}.txt') for i in range(nw)])
    n_seg = len(glob.glob(f'{W}/seg/qc/*.json'))
    print(f'segmentation: {n_seg} of {len(todo)} studies | {(time.time() - t0) / 60:.1f} min', flush=True)
    if n_seg < len(todo):
        print('time limit reached: run the same command again to continue', flush=True)
        return
    if not (os.path.exists(f'{W}/meta/rois.csv') and os.path.exists(f'{W}/anatomy.json')):
        run_parallel([(worker('regions', '--work', W, '--report-sides', f'{W}/meta/report_sides.csv', '--nproc', 4, '--strict', 0),
                       env(KNEE_THREADS=1), f'{W}/regions_log.txt')])
    run_parallel([(worker('cropping', '--work', W, '--nproc', 4, '--deadline', deadline, '--strict', 0), env(), f'{W}/crops_log.txt')])
    n_crops = len(glob.glob(f'{W}/crops/*.npz'))
    print(f'crops: {n_crops} of {len(studies)} studies | {(time.time() - t0) / 60:.1f} min', flush=True)
    if n_crops < len(studies):
        print('some studies have no crops yet: run the same command again to continue, or read crops_log.txt', flush=True)
        return
    print(f'done: {W}/crops for train.py, {W}/anatomy.json for predict.py', flush=True)


if __name__ == '__main__':
    main()
