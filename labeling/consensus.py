import os
import argparse
import numpy as np
import pandas as pd
from labeling.prompt import LABELS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--teachers', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weight', type=float, default=0.5)
    a = ap.parse_args(argv)
    T = pd.read_csv(a.labels).set_index('StudyInstanceUID')
    missing = [c for c in ['fold', 'is_gold'] + LABELS if c not in T]
    if missing:
        raise RuntimeError(f'{a.labels} has no column {missing}')
    oof = [pd.read_csv(f).set_index('StudyInstanceUID').reindex(T.index)[LABELS] for f in a.teachers]
    for f, o in zip(a.teachers, oof):
        if o.isna().any().any():
            raise RuntimeError(f'{f} has no prediction for {int(o.isna().any(axis=1).sum())} studies')
    teacher = sum(o.values for o in oof) / len(oof)
    report = T[LABELS].values
    gold = T.is_gold.values == 1
    cons = np.clip(a.weight * report + (1 - a.weight) * teacher, 0.02, 0.98)
    cons[gold] = report[gold]
    O = pd.DataFrame(cons, columns=LABELS, index=T.index)
    O.insert(0, 'is_gold', T.is_gold.astype(int).values)
    O.insert(0, 'fold', T.fold.astype(int).values)
    for j, l in enumerate(LABELS):
        O[f'report_{l}'] = report[:, j]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    O.reset_index().to_csv(a.out, index=False)
    moved = np.abs(cons - report)[~gold]
    print(f'{len(O)} studies ({int(gold.sum())} radiologist-labelled) | mean change from the report labels {moved.mean():.3f} | '
          f'labels crossing 0.5: {((cons >= 0.5) != (report >= 0.5))[~gold].mean():.1%}')


if __name__ == '__main__':
    main()
