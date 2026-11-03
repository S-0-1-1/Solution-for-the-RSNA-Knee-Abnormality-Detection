import numpy as np

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']

FOCUS = {
    'ACL': (['notch'], ['lateral_TF', 'medial_TF', 'joint']),
    'MCL': (['MCL'], ['medial_TF', 'bones']),
    'Medial Meniscus': (['medial_TF'], ['notch', 'MCL']),
    'Lateral Meniscus': (['lateral_TF'], ['notch']),
    'Medial OA': (['medial_TF'], ['bones', 'notch', 'MCL']),
    'Lateral OA': (['lateral_TF'], ['bones', 'notch']),
    'PF OA': (['PF'], ['joint']),
    'Effusion': (['joint', 'PF'], ['popliteal', 'notch']),
    'Synovitis': (['joint', 'PF'], ['notch', 'popliteal']),
    "Baker's": (['popliteal'], ['joint', 'medial_TF']),
    'Contusion': (['bones', 'medial_TF', 'lateral_TF'], ['PF', 'notch']),
    'Fracture': (['bones', 'medial_TF', 'lateral_TF', 'PF'], ['notch', 'MCL']),
}


def relevance(crop_names):
    R = np.full((len(LABELS), len(crop_names)), 2, np.int64)
    for i, label in enumerate(LABELS):
        main, near = FOCUS[label]
        for j, name in enumerate(crop_names):
            region = name.split('__')[0]
            R[i, j] = 0 if region in main else 1 if region in near else 2
    return R


def multilabel_folds(Y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, bool)
    n, L = Y.shape
    fold = np.full(n, -1)
    need_l = np.tile(Y.sum(0) / k, (k, 1)).astype(float)
    need_f = np.full(k, n / k)
    left = np.ones(n, bool)
    while left.any():
        cnt = Y[left].sum(0)
        if cnt.max() == 0:
            for i in rng.permutation(np.where(left)[0]):
                f = int(np.argmax(need_f + rng.random(k) * 1e-06))
                fold[i] = f
                need_f[f] -= 1
                left[i] = False
            break
        col = int(np.argmin(np.where(cnt > 0, cnt, np.inf)))
        for i in rng.permutation(np.where(left & Y[:, col])[0]):
            f = int(np.lexsort((rng.random(k), need_f, need_l[:, col]))[-1])
            fold[i] = f
            need_l[f] -= Y[i]
            need_f[f] -= 1
            left[i] = False
    return fold
