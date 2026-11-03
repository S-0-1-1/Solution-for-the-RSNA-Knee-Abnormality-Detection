import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from knee import crops
from knee.findings import LABELS, multilabel_folds
from knee.launch import worker, env, run_queue


def study_table(labels_csv, crops_dir, seed):
    L = pd.read_csv(labels_csv)
    missing = [c for c in ['StudyInstanceUID', 'is_gold'] + LABELS if c not in L]
    if missing:
        raise RuntimeError(f'{labels_csv} has no column {missing}')
    T = L[[os.path.exists(f'{crops_dir}/{s}.npz') for s in L.StudyInstanceUID]].reset_index(drop=True)
    if len(T) < 0.97 * len(L):
        raise RuntimeError(f'only {len(T)} of {len(L)} labelled studies have a crops file in {crops_dir}')
    if 'fold' in T and T.fold.notna().all():
        T['fold'] = T.fold.astype(int)
    else:
        T['fold'] = multilabel_folds(np.c_[T[LABELS].values >= 0.5, T.is_gold.values == 1], k=5, seed=seed)
    return T


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def out_of_fold(out, T, runs):
    O = pd.concat([pd.read_csv(f'{out}/{name}/oof.csv') for name, _ in runs])
    O.to_csv(f'{out}/oof.csv', index=False)
    E = T.merge(O, on='StudyInstanceUID', suffixes=('_y', ''))
    g = (E.is_gold == 1).values
    rows = {'gold studies, radiologist labels': {l: auc(E.loc[g, f'{l}_y'].values >= 0.5, E.loc[g, l].values) for l in LABELS},
            'other studies, training labels': {l: auc(E.loc[~g, f'{l}_y'].values >= 0.5, E.loc[~g, l].values) for l in LABELS}}
    if all(f'report_{l}' in E for l in LABELS):
        rows['other studies, report labels'] = {l: auc(E.loc[~g, f'report_{l}'].values >= 0.5, E.loc[~g, l].values) for l in LABELS}
    tab = pd.DataFrame(rows).T
    tab.insert(0, 'macro', tab[LABELS].mean(axis=1))
    tab.to_csv(f'{out}/auc.csv')
    print(f'out-of-fold AUC, {len(E)} studies ({int(g.sum())} gold)')
    print(tab.round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'kneenet.json'))
    ap.add_argument('--labels', required=True)
    ap.add_argument('--crops', required=True)
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    ap.add_argument('--full', type=int, default=0)
    ap.add_argument('--gpus', type=int, default=0)
    ap.add_argument('--hours', type=float, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    n_slots = a.gpus or max(1, torch.cuda.device_count())
    cfg = json.load(open(a.config))
    cfg.update(crop_names=list(crops.CROP_NAMES), n_crops=len(crops.CROP_NAMES), crops_dir=os.path.abspath(a.crops),
               pretrained=os.path.abspath(a.pretrained))
    os.makedirs(a.out, exist_ok=True)
    T = study_table(a.labels, a.crops, cfg['seed'])
    T.to_csv(f'{a.out}/table.csv', index=False)
    json.dump(cfg, open(f'{a.out}/config.json', 'w'), indent=1)
    runs = [(f'full{k}', -k) for k in range(1, a.full + 1)] if a.full else [(f'fold{k}', k) for k in a.folds]
    deadline = t0 + a.hours * 3600 if a.hours else 0
    todo = [(n, i) for n, i in runs if not os.path.exists(f'{a.out}/{n}/done.json')]
    print(f'{len(T)} studies ({int(T.is_gold.sum())} gold) | runs {[n for n, _ in todo]} | {n_slots} at a time', flush=True)
    for name, _ in todo:
        os.makedirs(f'{a.out}/{name}', exist_ok=True)
    run_queue([(worker('fit', '--run', os.path.abspath(a.out), '--index', i, '--name', n, '--deadline', deadline), env(),
                f'{a.out}/{n}/train.log') for n, i in todo], n_slots, deadline)
    if not a.full and all(os.path.exists(f'{a.out}/{n}/oof.csv') for n, _ in runs):
        out_of_fold(a.out, T, runs)
    print(f'training finished in {(time.time() - t0) / 3600:.1f} h', flush=True)


if __name__ == '__main__':
    main()
