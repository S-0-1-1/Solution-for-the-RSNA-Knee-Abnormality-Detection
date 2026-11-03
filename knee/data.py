import math
import numpy as np
import torch
import torch.nn.functional as F
from . import crops


def slice_index(S, K, train, rng):
    if S <= K:
        return np.arange(S)
    if not train:
        return np.unique(np.round(np.linspace(0, S - 1, K)).astype(int))
    edges = np.linspace(0, S, K + 1)
    return np.array([rng.integers(int(math.floor(a)), max(int(math.floor(a)) + 1, int(math.ceil(b))))
                     for a, b in zip(edges[:-1], edges[1:])]).clip(0, S - 1)


class StudyData:

    def __init__(self, crops_dir, crop_names, K_train=8, K_eval=12, min_fov=0.25):
        self.dir, self.names = crops_dir, list(crop_names)
        self.K_train, self.K_eval, self.min_fov = K_train, K_eval, min_fov

    def load(self, sid, train, rng):
        A, meta = crops.read_study(f'{self.dir}/{sid}.npz')
        K = self.K_train if train else self.K_eval
        H = crops.OUT_PX
        x = np.zeros((len(self.names), K, 3, H, H), np.uint8)
        valid = np.zeros((len(self.names), K), bool)
        pos = np.zeros((len(self.names), K), np.float32)
        fov = np.zeros(len(self.names), np.float32)
        for j, n in enumerate(self.names):
            if f'{n}__img' not in A:
                continue
            fov[j] = float(meta['crops'].get(n, {}).get('fov', 1.0))
            if fov[j] < self.min_fov:
                continue
            img, p = A[f'{n}__img'], A[f'{n}__pos'].astype(np.float32)
            S = len(img)
            idx = slice_index(S, K, train, rng)
            for t, i in enumerate(idx):
                x[j, t, 0], x[j, t, 1], x[j, t, 2] = img[max(i - 1, 0)], img[i], img[min(i + 1, S - 1)]
            valid[j, :len(idx)] = True
            pos[j, :len(idx)] = p[idx] / 50.0
        return x, valid, pos, fov


def augment_gpu(x, valid, gen=None, rot_deg=10.0, scale=0.1, shift=0.06, contrast=0.15, bright=0.08, gamma=0.15, drop_crop=0.05):
    B, Cn, K = valid.shape
    dev = x.device

    def r(*s):
        return torch.rand(*s, device=dev, generator=gen)
    ang = (r(B, Cn) * 2 - 1) * math.radians(rot_deg)
    sc = 1 + (r(B, Cn) * 2 - 1) * scale
    tx, ty = (r(B, Cn) * 2 - 1) * shift * 2, (r(B, Cn) * 2 - 1) * shift * 2
    th = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], -1),
                      torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], -1)], -2)
    m = valid.reshape(-1)
    flat = x.reshape(B * Cn * K, *x.shape[3:])[m]
    thf = th[:, :, None].expand(B, Cn, K, 2, 3).reshape(-1, 2, 3)[m]
    grid = F.affine_grid(thf, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    c = (1 + (r(B, Cn) * 2 - 1) * contrast)[:, :, None].expand(B, Cn, K).reshape(-1)[m]
    b = ((r(B, Cn) * 2 - 1) * bright)[:, :, None].expand(B, Cn, K).reshape(-1)[m]
    g = (1 + (r(B, Cn) * 2 - 1) * gamma)[:, :, None].expand(B, Cn, K).reshape(-1)[m]
    flat = (flat.clamp(0, 1) ** g[:, None, None, None] * c[:, None, None, None] + b[:, None, None, None]).clamp(0, 1)
    out = torch.zeros_like(x.reshape(B * Cn * K, *x.shape[3:]))
    out[m] = flat
    keep = r(B, Cn) >= drop_crop
    has = (valid.any(-1) & keep).any(-1, keepdim=True)
    keep = torch.where(has, keep, torch.ones_like(keep))
    return out.reshape(x.shape), valid & keep[:, :, None]
