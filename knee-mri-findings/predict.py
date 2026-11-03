import os
import json
import time
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from knee import series, crops
from knee.findings import LABELS
from knee.launch import worker, env, run_parallel


def ids_from_csv(path, strict):
    if not os.path.exists(path):
        return None
    C = pd.read_csv(path, dtype=str)
    if 'StudyInstanceUID' not in C and strict:
        raise RuntimeError(f'{path} has no StudyInstanceUID column: {list(C.columns)}')
    return list(dict.fromkeys(C['StudyInstanceUID' if 'StudyInstanceUID' in C else C.columns[0]].dropna().str.strip()))


def find_studies(a, W, t0):
    data = a.data
    test_dir = f'{data}/test_series'
    if not os.path.isdir(test_dir):
        cand = [d for d in series.subdirs(data) if d.lower().startswith('test')]
        if cand:
            test_dir = f'{data}/{cand[0]}'
    folders = series.subdirs(test_dir)
    fset = set(folders)
    s_ids, t_ids = ids_from_csv(f'{data}/sample_submission.csv', False), ids_from_csv(f'{data}/test.csv', False)
    hidden = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None or max(len(s_ids or []), len(t_ids or []), len(folders)) > 50
    strict = {'auto': not hidden, 'strict': True, 'lenient': False}[a.mode]
    if strict:
        s_ids, t_ids = ids_from_csv(f'{data}/sample_submission.csv', True), ids_from_csv(f'{data}/test.csv', True)
    fit = [(c, n) for c, n in ((s_ids, 'sample_submission.csv'), (t_ids, 'test.csv'))
           if c and (not fset or len(fset.intersection(c)) >= 0.5 * len(c))]
    test_ids, id_src = max(fit, key=lambda x: len(x[0])) if fit else (folders, 'folders')
    if fset and len(test_ids) < 0.5 * len(fset):
        test_ids, id_src = folders, 'folders'
    if not test_ids:
        raise RuntimeError(f'no test study found in {data}: {sorted(os.listdir(data))}')
    if strict and s_ids is not None and t_ids is not None and set(s_ids) != set(t_ids):
        raise RuntimeError(f'sample_submission.csv and test.csv list different studies ({len(s_ids)} vs {len(t_ids)})')
    no_dir = [s for s in test_ids if s not in fset]
    if no_dir and (strict or len(no_dir) == len(test_ids)):
        raise RuntimeError(f'{len(no_dir)} of {len(test_ids)} test studies have no folder in {test_dir}: {no_dir[:5]}')
    check_ids = []
    if not hidden and a.check_crops and os.path.exists(f'{data}/train_series.csv'):
        TR = pd.read_csv(f'{data}/train_series.csv', dtype={'StudyInstanceUID': str})
        have = [s for s in sorted(TR.StudyInstanceUID.unique()) if os.path.exists(f'{a.check_crops}/{s}.npz')]
        check_ids = [have[i] for i in np.random.default_rng(7).choice(len(have), min(6, len(have)), replace=False)]
    TS = pd.concat([series.series_rows(test_ids, test_dir, f'{data}/test_series.csv', 'test'),
                    series.series_rows(check_ids, f'{data}/train_series', f'{data}/train_series.csv', 'train')], ignore_index=True)
    n_rebuilt = 0
    if not strict:
        TS, n_rebuilt = series.rebuild_folders(TS, test_ids, test_dir, fset, f'{W}/mirror')
    no_img = sorted(set(test_ids) - set(TS[TS.split == 'test'].StudyInstanceUID))
    if no_img and (strict or len(no_img) == len(test_ids)):
        raise RuntimeError(f'{len(no_img)} test studies have no series: {no_img[:5]}')
    SM = series.add_headers(TS, strict)
    SM.drop(columns=['series_dir', 'split', 'plane_hdr', 'fs_hdr']).to_csv(f'{W}/meta/series_meta.csv', index=False)
    studies = series.study_table(SM)
    no_ref = studies[studies.ref_dir == ''].StudyInstanceUID.tolist()
    if no_ref and strict:
        raise RuntimeError(f'{len(no_ref)} studies have no usable sagittal series: {no_ref[:5]}')
    studies.to_csv(f'{W}/meta/studies.csv', index=False)
    print(f'test studies {len(test_ids)} from {id_src} | series {int((SM.split == "test").sum())} | '
          f'header errors {int(SM.error.notna().sum())} | no sagittal {len(no_ref)} | no folder {len(no_dir)} | '
          f'rebuilt folders {n_rebuilt} | check studies {len(check_ids)} | strict {strict} | {(time.time() - t0) / 60:.1f} min', flush=True)
    return dict(test_ids=test_ids, check_ids=check_ids, strict=strict, series=SM, studies=studies, no_ref=no_ref)


def gpu_env(n_gpu, i, **extra):
    return env(CUDA_VISIBLE_DEVICES=i % n_gpu, **extra) if n_gpu else env(**extra)


def run_preprocessing(a, W, t0, limit, run):
    strict, studies, n_all = run['strict'], run['studies'], len(run['studies'])
    n_gpu = torch.cuda.device_count()
    seg_table = f'{W}/meta/seg_table.csv'
    studies[studies.ref_dir != ''][['StudyInstanceUID', 'ref_dir', 'ref_uid', 'alt_dir', 'alt_uid', 'lat_tag', 'site_key']].to_csv(
        seg_table, index=False)
    nw = 2 * max(1, n_gpu)
    deadline = t0 + limit - n_all * 4.5 - 600
    nn = dict(KNEE_THREADS=1, nnUNet_raw=f'{W}/nnunet', nnUNet_preprocessed=f'{W}/nnunet', nnUNet_results=f'{W}/nnunet')
    run_parallel([(worker('segment', '--table', seg_table, '--model', a.segmentation, '--out', f'{W}/seg', '--shard', i,
                          '--nshards', nw, '--deadline', deadline, '--strict', int(strict)),
                   gpu_env(n_gpu, i, **nn), f'{W}/seg/log{i}.txt') for i in range(nw)], strict)
    print(f'segmentation: {len(glob.glob(f"{W}/seg/qc/*.json"))} of {n_all} studies | {(time.time() - t0) / 60:.1f} min', flush=True)
    run_parallel([(worker('regions', '--work', W, '--anatomy', f'{a.models}/anatomy.json', '--nproc', 4, '--strict', int(strict)),
                   env(KNEE_THREADS=1), f'{W}/regions_log.txt')], strict)
    deadline = t0 + limit - n_all * 1.0 - 300
    run_parallel([(worker('cropping', '--work', W, '--nproc', 4, '--deadline', deadline, '--strict', int(strict)),
                   env(), f'{W}/crops_log.txt')], strict)
    print(f'crops: {len(glob.glob(f"{W}/crops/*.npz"))} of {n_all} studies | {(time.time() - t0) / 60:.1f} min', flush=True)


def run_models(a, W, t0, limit, run):
    blend = json.load(open(f'{a.models}/blend.json'))
    fams = []
    for name in blend:
        weights = sorted(glob.glob(f'{a.models}/{name}/*/best.pt'))
        if not weights:
            raise RuntimeError(f'no weights in {a.models}/{name}')
        fams.append(dict(name=name, config=f'{a.models}/{name}/config.json', weights=weights))
    cfgs = [json.load(open(f['config'])) for f in fams]
    names = cfgs[0]['crop_names']
    if any(c['crop_names'] != names for c in cfgs):
        raise RuntimeError('the model families were trained on different crops')
    rows = [dict(key=s, crops_file=f'{W}/crops/{s}.npz') for s in run['studies'].StudyInstanceUID]
    rows += [dict(key=f'stored|{s}', crops_file=f'{a.check_crops}/{s}.npz') for s in run['check_ids']]
    pd.DataFrame(rows).to_csv(f'{W}/meta/score.csv', index=False)
    job = dict(crop_names=names, min_fov=cfgs[0].get('min_fov', 0.25), k_eval=cfgs[0].get('K_eval', 12),
               studies=f'{W}/meta/score.csv', families=fams, out=W, workers=2)
    json.dump(job, open(f'{W}/meta/score_job.json', 'w'))
    n_gpu = torch.cuda.device_count()
    parts = max(1, n_gpu)
    run_parallel([(worker('scoring', '--job', f'{W}/meta/score_job.json', '--part', g, '--parts', parts, '--deadline', t0 + limit,
                          '--strict', int(run['strict'])), gpu_env(n_gpu, g), f'{W}/score_log{g}.txt') for g in range(parts)], run['strict'])
    keys = pd.read_csv(f'{W}/meta/score.csv', dtype=str).key.tolist()
    PR = {f['name']: np.full((len(keys), len(LABELS)), np.nan) for f in fams}
    for g in range(parts):
        p = f'{W}/scores_{g}of{parts}.npz'
        if not os.path.exists(p):
            if run['strict']:
                raise RuntimeError(f'missing model output {p}')
            continue
        z = np.load(p)
        for k in PR:
            PR[k][z['rows']] = z[k]
    num = sum(np.nan_to_num(PR[k]) * blend[k] * ~np.isnan(PR[k]) for k in PR)
    den = sum(blend[k] * ~np.isnan(PR[k]) for k in PR)
    P = np.full(num.shape, np.nan)
    np.divide(num, den, out=P, where=den > 0)
    print(f'models: {[f["name"] for f in fams]} | {(time.time() - t0) / 60:.1f} min', flush=True)
    return P, dict(zip(keys, range(len(keys))))


def write_submission(a, W, t0, run, P, at):
    order, strict = run['test_ids'], run['strict']
    rows = np.array([P[at[s]] if s in at else np.full(len(LABELS), np.nan) for s in order]).reshape(len(order), len(LABELS))
    ok = ~np.isnan(rows).any(1)
    if strict and not ok.all():
        raise RuntimeError(f'{int((~ok).sum())} test studies have no prediction: {[s for s, o in zip(order, ok) if not o][:5]}')
    if not ok.any():
        raise RuntimeError('no test study has a prediction')
    tset = set(order)
    RO = pd.read_csv(f'{W}/meta/rois.csv', dtype={'StudyInstanceUID': str})
    RO = RO[RO.StudyInstanceUID.isin(tset)]
    seg_err = set()
    for s in tset:
        f = f'{W}/seg/qc/{s}.json'
        if os.path.exists(f) and 'error' in [str(json.load(open(f)).get(k) or '') for k in ('flag_reasons', 'ref_flags', 'alt_flags')]:
            seg_err.add(s)
    crop_err = {s for s in tset if os.path.exists(f'{W}/crops/{s}.npz') and json.loads(str(np.load(f'{W}/crops/{s}.npz')['meta']))['errors']}
    SM = run['series']
    fb = {'no prediction': {s for s, o in zip(order, ok) if not o}, 'standard boxes': set(RO[RO.source != 'bones'].StudyInstanceUID),
          'unreadable series': set(SM[(SM.split == 'test') & SM.error.notna()].StudyInstanceUID), 'no sagittal': set(run['no_ref']) & tset,
          'segmentation errors': seg_err, 'crop errors': crop_err}
    fallback = set().union(*fb.values())
    cap = max(5, int(np.ceil(0.01 * len(order))))
    print(f'studies on a fallback path {len(fallback)} (limit {cap}) | ' + ' | '.join(f'{k} {len(v)}' for k, v in fb.items()), flush=True)
    if not strict and len(fallback) > cap:
        raise RuntimeError(f'{len(fallback)} of {len(order)} test studies needed a fallback (limit {cap}): '
                           + ', '.join(f'{k} {len(v)}' for k, v in fb.items()))
    rows = np.where(np.isnan(rows), np.nanmedian(rows[ok], 0)[None], rows)
    sub = pd.DataFrame(rows, columns=LABELS)
    sub.insert(0, 'StudyInstanceUID', order)
    sub.to_csv(a.out, index=False)
    print(f'{a.out}: {len(sub)} studies | predicted {int(ok.sum())} | filled {int((~ok).sum())} | '
          f'bone boxes {RO[RO.source == "bones"].StudyInstanceUID.nunique()} | {(time.time() - t0) / 60:.1f} min', flush=True)


def check_against_training(a, W, run, P, at):
    d_pred, d_pix, n_both, n_same, n_one = [], [], 0, 0, 0
    for s in run['check_ids']:
        d_pred.append(np.abs(P[at[s]] - P[at[f'stored|{s}']]))
        A1, m1 = crops.read_study(f'{W}/crops/{s}.npz')
        A2, m2 = crops.read_study(f'{a.check_crops}/{s}.npz')
        c1, c2 = set(m1['crops']), set(m2['crops'])
        n_one += len(c1 ^ c2)
        for c in c1 & c2:
            n_both += 1
            x, y = A1[f'{c}__img'], A2[f'{c}__img']
            if x.shape == y.shape:
                n_same += 1
                d_pix.append(float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()))
    d = np.array(d_pred)
    print(f'check on {len(d)} training studies against their training crops: prediction difference mean {d.mean():.4f}, '
          f'max {d.max():.4f} | crops in both {n_both} (same size {n_same}), in one only {n_one} | '
          f'mean pixel difference {np.mean(d_pix) if d_pix else float("nan"):.2f} of 255', flush=True)
    if np.isnan(d).any() or d.mean() > 0.02:
        raise RuntimeError('the pipeline does not reproduce the training crops')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--segmentation', required=True)
    ap.add_argument('--work', default='work')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--mode', choices=['auto', 'strict', 'lenient'], default='auto')
    ap.add_argument('--check-crops', default='')
    ap.add_argument('--hours', type=float, default=8.6)
    a = ap.parse_args(argv)
    t0, limit = time.time(), a.hours * 3600
    W = os.path.abspath(a.work)
    for d in (W, f'{W}/meta', f'{W}/seg', f'{W}/nnunet'):
        os.makedirs(d, exist_ok=True)
    run = find_studies(a, W, t0)
    run_preprocessing(a, W, t0, limit, run)
    P, at = run_models(a, W, t0, limit, run)
    write_submission(a, W, t0, run, P, at)
    if run['check_ids']:
        check_against_training(a, W, run, P, at)


if __name__ == '__main__':
    main()
