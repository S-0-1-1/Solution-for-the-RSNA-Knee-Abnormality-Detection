import json
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from knee import crops
from knee.findings import LABELS, relevance
from knee.data import slice_index
from knee.model import build

PASSES = [(0.0, 1.0), (5.0, 1.0), (-5.0, 1.0), (0.0, 1.05)]


class Studies(Dataset):

    def __init__(self, files, rows, names, min_fov, strict):
        self.files, self.rows, self.names, self.min_fov, self.strict = files, rows, names, min_fov, strict

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        f = self.files[r]
        try:
            A, meta = crops.read_study(f)
        except Exception as e:
            if self.strict:
                raise RuntimeError(f'crops of {f}: {e!r}') from e
            return None, None, None, r, f'{type(e).__name__}: {str(e)[:150]}'
        S = np.zeros(len(self.names), np.int64)
        imgs, poss = {}, {}
        for j, n in enumerate(self.names):
            if f'{n}__img' not in A:
                continue
            if float(meta['crops'].get(n, {}).get('fov', 1.0)) < self.min_fov:
                continue
            imgs[j], poss[j] = A[f'{n}__img'], A[f'{n}__pos'].astype(np.float32)
            S[j] = len(imgs[j])
        H = crops.OUT_PX
        IMG = np.zeros((len(self.names), max(1, int(S.max())), H, H), np.uint8)
        POS = np.zeros(IMG.shape[:2], np.float32)
        for j in imgs:
            IMG[j, :S[j]] = imgs[j]
            POS[j, :S[j]] = poss[j]
        return torch.from_numpy(IMG), torch.from_numpy(POS), torch.from_numpy(S), r, ''


def autocast(amp):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if amp else torch.autocast(device_type='cpu', enabled=False)


def model_input(IMG, POS, S_t, S_np, dev, K):
    idx = np.zeros((len(S_np), K), np.int64)
    val = np.zeros(idx.shape, bool)
    for j, s in enumerate(S_np):
        if s > 0:
            x = slice_index(int(s), K, False, None)
            idx[j, :len(x)] = x
            val[j, :len(x)] = True
    idx_t, val_t = torch.from_numpy(idx).to(dev), torch.from_numpy(val).to(dev)
    C = len(S_np)
    top = (S_t - 1).clamp(min=0)[:, None].expand(C, K)
    lo, hi = (idx_t - 1).clamp(min=0), torch.minimum(idx_t + 1, top)
    c = torch.arange(C, device=dev)[:, None].expand(C, K)
    x = torch.stack([IMG[c, lo], IMG[c, idx_t], IMG[c, hi]], 2).float().div_(255)
    x.mul_(val_t[..., None, None, None].to(x.dtype))
    pos = POS[c, idx_t] / 50.0 * val_t
    xs = []
    for rot, zoom in PASSES:
        if rot == 0.0 and zoom == 1.0:
            xs.append(x)
            continue
        t = math.radians(rot)
        th = torch.tensor([[math.cos(t) / zoom, -math.sin(t) / zoom, 0.0], [math.sin(t) / zoom, math.cos(t) / zoom, 0.0]],
                          device=dev, dtype=x.dtype)
        flat, m = x.reshape(-1, *x.shape[2:]), val_t.reshape(-1)
        out = torch.zeros_like(flat)
        sub = flat[m]
        if len(sub):
            grid = F.affine_grid(th[None].expand(len(sub), 2, 3), list(sub.shape), align_corners=False)
            out[m] = F.grid_sample(sub, grid, mode='bilinear', padding_mode='zeros', align_corners=False).clamp(0, 1)
        xs.append(out.reshape(x.shape))
    B = len(PASSES)
    return torch.stack(xs), val_t[None].expand(B, C, K).contiguous(), pos[None].expand(B, C, K).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--deadline', type=float, default=1e18)
    ap.add_argument('--strict', type=int, default=1)
    a = ap.parse_args()
    J = json.load(open(a.job))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    T = pd.read_csv(J['studies'], dtype=str)
    rows = np.arange(len(T))[a.part::a.parts]
    names = list(J['crop_names'])
    R = relevance(names)
    models = {}
    for fam in J['families']:
        cfg = json.load(open(fam['config']))
        models[fam['name']] = []
        for path in fam['weights']:
            net = build(cfg, R)
            net.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
            models[fam['name']].append(net.to(dev).eval())
    P = {k: np.full((len(rows), len(LABELS)), np.nan, np.float32) for k in models}
    where = {int(r): i for i, r in enumerate(rows)}
    print(f'[score {a.part}] {len(rows)} studies | ' + ', '.join(f'{k} {len(v)}' for k, v in models.items()), flush=True)
    ds = Studies(T.crops_file.tolist(), rows, names, float(J['min_fov']), bool(a.strict))
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=int(J['workers']), pin_memory=amp,
                    prefetch_factor=4 if int(J['workers']) > 0 else None)
    t0, done, failed = time.time(), 0, 0
    with torch.no_grad():
        for IMG, POS, S, r, err in dl:
            if time.time() > a.deadline:
                print(f'[score {a.part}] time limit after {done} studies', flush=True)
                break
            i = where[r]
            if err:
                failed += 1
            else:
                S_np = S.numpy()
                IMG, POS, S_t = IMG.to(dev, non_blocking=True), POS.to(dev, non_blocking=True), S.to(dev)
                x, val, pos = model_input(IMG, POS, S_t, S_np, dev, int(J['k_eval']))
                for k, nets in models.items():
                    acc = np.zeros(len(LABELS), np.float64)
                    for net in nets:
                        with autocast(amp):
                            lg, _ = net(x, val, pos)
                        acc += torch.sigmoid(lg.float()).mean(0).cpu().numpy()
                    P[k][i] = acc / len(nets)
            done += 1
            if done % 100 == 0 or done == len(rows):
                el = time.time() - t0
                print(f'[score {a.part}] {done}/{len(rows)} | {done / el:.2f} studies/s | {(len(rows) - done) * el / done / 60:.0f} min left',
                      flush=True)
    np.savez(f'{J["out"]}/scores_{a.part}of{a.parts}.npz', rows=rows, **P)
    print(f'[score {a.part}] done: {done} studies ({failed} without crops) in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
