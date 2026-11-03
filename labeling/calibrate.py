import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss
from labeling.prompt import LABELS, STATES, STATE_RANK

NONE = STATE_RANK['none']


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def unpack(t):
    return t[0] + np.concatenate([[0.0], np.cumsum(softplus(t[1:6]))]), t[6:18]


def fit(S, Y, lam):
    def loss(t):
        a, b = unpack(t)
        p = np.clip(sigmoid(a[S] + b[None, :]), 1e-6, 1 - 1e-6)
        return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).sum() + lam * (b ** 2).sum()
    return minimize(loss, np.concatenate([[-2.0], np.full(5, 0.3), np.zeros(12)]), method='L-BFGS-B').x


def predict(t, S):
    a, b = unpack(t)
    return sigmoid(a[S] + b[None, :])


def leave_one_out(S, Y, lam):
    P, idx = np.zeros_like(Y), np.arange(len(Y))
    for i in idx:
        P[i] = predict(fit(S[idx != i], Y[idx != i], lam), S[i:i + 1])[0]
    return P


def state_auc(S, Y):
    return {l: roc_auc_score(Y[:, j], S[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan for j, l in enumerate(LABELS)}


def load_reader(paths, ids):
    rows = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            if r['StudyInstanceUID'] not in rows or (r['ok'] and not rows[r['StudyInstanceUID']][1]):
                rows[r['StudyInstanceUID']] = ([STATE_RANK[r['states'][l]] for l in LABELS], r['ok'])
    S = np.array([rows[s][0] if s in rows else [NONE] * len(LABELS) for s in ids])
    return S, np.array([s in rows for s in ids])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--reader', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    TR = pd.read_csv(f'{a.data}/train.csv')
    ids = TR.StudyInstanceUID.tolist()
    gold = TR[LABELS].notna().all(axis=1).values
    Yg = TR.loc[gold, LABELS].values.astype(float)
    readers, answered, lam = {}, {}, {}
    for spec in a.reader:
        name, paths = spec.split('=', 1)
        S, ans = load_reader(paths.split(','), ids)
        if not ans[gold].all():
            raise RuntimeError(f'{name} has not read all {int(gold.sum())} radiologist-labelled studies: it cannot be calibrated')
        readers[name], answered[name] = S, ans
        best = None
        for lm in (0.3, 1.0, 3.0, 10.0):
            ll = log_loss(Yg.ravel(), leave_one_out(S[gold], Yg, lm).ravel())
            if best is None or ll < best[1]:
                best = (lm, ll)
        lam[name] = best[0]
        print(f'{name}: {int(ans.sum())} of {len(ids)} reports | gold macro AUC {np.nanmean(list(state_auc(S[gold], Yg).values())):.3f} | '
              f'calibration log loss {best[1]:.3f}')
    names = list(readers)
    combos = [(n,) for n in names] + (list(itertools.combinations(names, 2)) if len(names) > 2 else [])
    combos += [tuple(names)] if len(names) > 1 else []
    score = {c: state_auc(np.mean([readers[n][gold].astype(float) for n in c], axis=0), Yg) for c in combos}
    table = pd.DataFrame({' + '.join(c): v for c, v in score.items()}).T
    table.insert(0, 'macro', table.mean(axis=1))
    print('AUC on the radiologist-labelled studies, single readers and averages:')
    print(table.sort_values('macro', ascending=False).round(3).to_string())
    macro = {c: np.nanmean(list(v.values())) for c, v in score.items()}
    use = max([c for c in combos if macro[c] >= max(macro.values()) - 0.005], key=lambda c: (len(c), macro[c]))
    P_of = {n: predict(fit(readers[n][gold], Yg, lam[n]), readers[n]) for n in names}
    num, den = np.zeros((len(ids), len(LABELS))), np.zeros((len(ids), 1))
    for n in use:
        w = answered[n][:, None].astype(float)
        num, den = num + w * P_of[n], den + w
    gap = den[:, 0] == 0
    for n in names:
        if n not in use:
            w = (answered[n] & gap)[:, None].astype(float)
            num, den = num + w * P_of[n], den + w
    P = np.where(den > 0, num / np.maximum(den, 1e-9), P_of[use[0]])
    soft = pd.DataFrame(np.clip(P, 0.02, 0.98), columns=LABELS)
    soft.insert(0, 'StudyInstanceUID', ids)
    soft.loc[gold, LABELS] = TR.loc[gold, LABELS].values
    soft.insert(1, 'is_gold', gold.astype(int))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    soft.to_csv(a.out, index=False)
    print(f'labels from {" + ".join(use)} (macro {macro[use]:.3f}) | read by none of them: {int(gap.sum())} | '
          f'read by nobody: {int((den[:, 0] == 0).sum())} | worth of each state for an average finding: '
          + ', '.join(f'{s} {float(sigmoid(unpack(fit(readers[use[0]][gold], Yg, lam[use[0]]))[0][i])):.2f}' for i, s in enumerate(STATES)))


if __name__ == '__main__':
    main()
