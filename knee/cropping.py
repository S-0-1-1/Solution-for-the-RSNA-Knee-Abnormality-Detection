import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np
import pandas as pd
from knee import crops

S = {}


def setup(work, strict):
    SM = pd.read_csv(f'{work}/meta/series_meta.csv')
    RO = pd.read_csv(f'{work}/meta/rois.csv')
    QF = pd.read_csv(f'{work}/meta/regions.csv', usecols=lambda c: c in ('StudyInstanceUID', 'SeriesInstanceUID', 'site_key'))
    ST = pd.read_csv(f'{work}/meta/studies.csv', dtype=str)
    S.update(work=work, strict=strict, empty=SM.iloc[:0], series={k: g for k, g in SM.groupby('StudyInstanceUID')},
             boxes={sid: {r.roi: [r.xmin, r.xmax, r.ymin, r.ymax, r.zmin, r.zmax] for r in g.itertuples()}
                    for sid, g in RO.groupby('StudyInstanceUID')},
             sources={sid: dict(zip(g.roi, g.source)) for sid, g in RO.groupby('StudyInstanceUID')},
             side=RO.groupby('StudyInstanceUID').lat.first().to_dict(), ref=dict(zip(QF.StudyInstanceUID, QF.SeriesInstanceUID)),
             site=dict(zip(QF.StudyInstanceUID, QF.site_key)) if 'site_key' in QF else {},
             dirs=dict(zip(ST.StudyInstanceUID, ST.study_dir)))


def study(sid):
    W, t0 = S['work'], time.time()
    out = f'{W}/crops/{sid}.npz'
    if os.path.exists(out):
        return dict(StudyInstanceUID=sid, skipped=1)
    try:
        if sid not in S['boxes']:
            raise KeyError('no region boxes')
        lat = S['side'].get(sid)
        lat = lat if lat in ('L', 'R') else None
        Ar, meta = crops.study_crops(sid, S['series'].get(sid, S['empty']), S['boxes'][sid], lat, S['dirs'][sid],
                                     f'{W}/masks/{sid}.nii.gz', S['ref'].get(sid), S['sources'].get(sid))
        if S['strict'] and meta['errors']:
            raise RuntimeError(f'crop errors {meta["errors"][:3]}')
        meta['site_key'] = S['site'].get(sid)
        Ar['meta'] = np.array(json.dumps(meta, default=crops.js))
        tmp = f'{W}/crops/{sid}.tmp.npz'
        np.savez(tmp, **Ar)
        os.replace(tmp, out)
        return dict(StudyInstanceUID=sid, n_crops=len(meta['crops']), problems=len(meta['errors']), sec=round(time.time() - t0, 2))
    except Exception as e:
        if S['strict']:
            raise RuntimeError(f'{sid}: {e!r}') from e
        return dict(StudyInstanceUID=sid, fatal=f'{type(e).__name__}: {str(e)[:300]}', sec=round(time.time() - t0, 2))


def run_all(todo, nproc, deadline, work, strict):
    handled, remaining = set(), list(todo)
    while remaining:
        try:
            with ProcessPoolExecutor(nproc, initializer=setup, initargs=(work, strict)) as pool:
                for m in pool.map(study, remaining, chunksize=1):
                    handled.add(m['StudyInstanceUID'])
                    yield m
                    if time.time() > deadline:
                        print('[crops] time limit', flush=True)
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
            remaining = []
        except BrokenProcessPool:
            if strict:
                raise
            remaining = [s for s in remaining if s not in handled]
            for s in remaining[:nproc]:
                try:
                    with ProcessPoolExecutor(1, initializer=setup, initargs=(work, strict)) as p1:
                        m = p1.submit(study, s).result()
                except BrokenProcessPool:
                    m = dict(StudyInstanceUID=s, fatal='worker died')
                handled.add(s)
                yield m
            remaining = remaining[nproc:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--deadline', type=float, default=1e18)
    ap.add_argument('--strict', type=int, default=1)
    A = ap.parse_args()
    os.makedirs(f'{A.work}/crops', exist_ok=True)
    todo = pd.read_csv(f'{A.work}/meta/studies.csv', dtype=str).StudyInstanceUID.tolist()
    print(f'[crops] {len(todo)} studies', flush=True)
    t0, n, bad, prob = time.time(), 0, 0, 0
    for m in run_all(todo, A.nproc, A.deadline, A.work, bool(A.strict)):
        n += 1
        bad += int('fatal' in m)
        prob += int(m.get('problems', 0) > 0)
        if n % 100 == 0 or n == len(todo):
            el = time.time() - t0
            print(f'[crops] {n}/{len(todo)} | failed {bad} | {el / n:.2f} s/study | {el * (len(todo) - n) / n / 60:.0f} min left',
                  flush=True)
    print(f'[crops] done: {n} studies in {(time.time() - t0) / 60:.1f} min | failed {bad} | with a missing crop {prob}', flush=True)


if __name__ == '__main__':
    main()
