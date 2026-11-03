import os
import json
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from timm.utils import ModelEmaV3
from knee.findings import LABELS, relevance
from knee.data import StudyData, augment_gpu
from knee.model import build

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--index', type=int, required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--deadline', type=float, default=0)
a = ap.parse_args()
cfg = json.load(open(f'{a.run}/config.json'))
torch.manual_seed(cfg['seed'] + a.index)
np.random.seed(cfg['seed'] + a.index)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = dev.type == 'cuda'
OUT = f'{a.run}/{a.name}'
os.makedirs(OUT, exist_ok=True)


def log(*s):
    print(f'[{a.name}]', *s, flush=True)


T = pd.read_csv(f'{a.run}/table.csv')
tr, va = T[T.fold != a.index].reset_index(drop=True), T[T.fold == a.index].reset_index(drop=True)
SD = StudyData(cfg['crops_dir'], cfg['crop_names'], cfg['K_train'], cfg['K_eval'], cfg['min_fov'])


class Studies(Dataset):

    def __init__(self, tab, train):
        self.t, self.train, self.rng = tab, train, None

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng((cfg['seed'] * 1000 + a.index * 100 + os.getpid()) % 2 ** 32)
        r = self.t.iloc[i]
        x, valid, pos, fov = SD.load(r.StudyInstanceUID, self.train, self.rng)
        y = r[LABELS].values.astype(np.float32)
        w = np.full(len(LABELS), cfg['gold_w'] if r.is_gold == 1 else 1.0, np.float32)
        return torch.from_numpy(x), torch.from_numpy(valid), torch.from_numpy(pos), torch.from_numpy(y), torch.from_numpy(w), i


def loader(tab, train, bs):
    return DataLoader(Studies(tab, train), batch_size=bs, shuffle=train, drop_last=train, num_workers=cfg['workers'],
                      pin_memory=AMP, persistent_workers=cfg['workers'] > 0, prefetch_factor=2 if cfg['workers'] > 0 else None)


def autocast():
    return torch.autocast(device_type='cuda', dtype=torch.float16) if AMP else torch.autocast(device_type='cpu', enabled=False)


model = build(cfg, relevance(cfg['crop_names'])).to(dev)
sd = torch.load(cfg['pretrained'], map_location='cpu')
res = model.enc.load_state_dict(sd, strict=False)
n_enc = len([k for k in model.enc.state_dict() if not k.endswith('num_batches_tracked')])
missing = [k for k in res.missing_keys if not k.endswith('num_batches_tracked')]
if len(missing) > 0.05 * n_enc:
    raise RuntimeError(f'{cfg["pretrained"]} does not fit the {cfg["backbone"]} encoder (missing {missing[:5]})')
enc_p = list(model.enc.parameters())
enc_ids = {id(q) for q in enc_p}
opt = torch.optim.AdamW([{'params': enc_p, 'lr': cfg['lr_enc']},
                         {'params': [q for q in model.parameters() if id(q) not in enc_ids], 'lr': cfg['lr_head']}],
                        weight_decay=cfg['wd'])
base_lr = [g['lr'] for g in opt.param_groups]
scaler = torch.amp.GradScaler('cuda', enabled=AMP)
ema = ModelEmaV3(model, decay=cfg['ema']) if cfg.get('ema') else None
dl_tr = loader(tr, True, cfg['batch'])
steps_per_epoch = len(dl_tr) // cfg['accum']
total = max(1, steps_per_epoch * cfg['epochs'])
warm = min(300, total // 10)


def set_lr(step):
    f = step / max(1, warm) if step < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    for g, b in zip(opt.param_groups, base_lr):
        g['lr'] = b * f


def to_dev(batch):
    x, valid, pos, y, w, idx = batch
    return (x.to(dev, non_blocking=True).float().div_(255), valid.to(dev, non_blocking=True), pos.to(dev, non_blocking=True),
            y.to(dev, non_blocking=True), w.to(dev, non_blocking=True), idx)


@torch.no_grad()
def predict(net, tab):
    net.eval()
    P = np.zeros((len(tab), len(LABELS)), np.float32)
    for batch in loader(tab, False, cfg['batch_eval']):
        x, valid, pos, y, w, idx = to_dev(batch)
        with autocast():
            lg, _ = net(x, valid, pos)
        P[idx.numpy()] = torch.sigmoid(lg.float()).cpu().numpy()
    return P


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def metrics(tab, P):
    Y = tab[LABELS].values.astype(np.float32)
    Pc = np.clip(P, 1e-06, 1 - 1e-06)
    W = np.where(tab.is_gold.values[:, None] == 1, cfg['gold_w'], 1.0) * np.ones_like(Y)
    bce = float(-(W * (Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc))).sum() / W.sum())
    return bce, float(np.nanmean([auc(Y[:, j] >= 0.5, P[:, j]) for j in range(len(LABELS))]))


log(f'{len(tr)} studies to train, {len(va)} to validate | {cfg["epochs"]} epochs x {steps_per_epoch} steps | {dev}')
best, step, t_start, hist = None, 0, time.time(), []
for ep in range(cfg['epochs']):
    model.train()
    t_ep, seen, run_loss = time.time(), 0, 0.0
    for i, batch in enumerate(dl_tr):
        x, valid, pos, y, w, _ = to_dev(batch)
        x, valid = augment_gpu(x, valid)
        with autocast():
            lg, _ = model(x, valid, pos)
            loss = (F.binary_cross_entropy_with_logits(lg.float(), y, reduction='none') * w).sum() / w.sum()
        scaler.scale(loss / cfg['accum']).backward()
        run_loss += loss.item()
        seen += len(y)
        if (i + 1) % cfg['accum'] == 0:
            set_lr(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                ema.update(model, step=step)
        if (i + 1) % cfg['log_every'] == 0:
            el = time.time() - t_ep
            log(f'epoch {ep + 1}/{cfg["epochs"]} step {i + 1}/{len(dl_tr)} | loss {run_loss / (i + 1):.4f} | '
                f'{seen / el:.1f} studies/s | {(len(dl_tr) - i - 1) * el / (i + 1) / 60:.1f} min left in epoch')
    net = ema.module if ema is not None else model
    rec = dict(epoch=ep + 1, train_loss=run_loss / max(1, len(dl_tr)))
    if len(va):
        P = predict(net, va)
        rec['val_loss'], rec['val_auc'] = metrics(va, P)
        g = va.is_gold.values == 1
        gauc = [auc(va.loc[g, l].values >= 0.5, P[g, j]) for j, l in enumerate(LABELS)] if g.sum() >= 4 else []
        gauc = [v for v in gauc if v == v]
        rec['gold_auc'] = float(np.mean(gauc)) if gauc else None
    rec['min'] = (time.time() - t_ep) / 60
    hist.append(rec)
    saved = not len(va) or best is None or rec['val_loss'] < best['val_loss']
    if saved:
        best = rec
        torch.save(dict(state_dict=net.state_dict(), epoch=ep + 1), f'{OUT}/best.pt')
    log(f'epoch {ep + 1} | train loss {rec["train_loss"]:.4f}'
        + (f' | val loss {rec["val_loss"]:.4f} | val auc {rec["val_auc"]:.4f}' if len(va) else '')
        + (f' | gold auc {rec["gold_auc"]:.3f}' if rec.get('gold_auc') is not None else '')
        + (' | saved' if saved else '') + f' | {rec["min"]:.1f} min')
    json.dump(hist, open(f'{OUT}/history.json', 'w'), indent=1)
    if a.deadline and time.time() + (time.time() - t_ep) * 1.2 > a.deadline and ep + 1 < cfg['epochs']:
        log('stopping: the next epoch would not end before the time limit')
        break
if len(va):
    model.load_state_dict(torch.load(f'{OUT}/best.pt', map_location='cpu')['state_dict'])
    P = predict(model, va)
    oof = va[['StudyInstanceUID']].copy()
    for j, l in enumerate(LABELS):
        oof[l] = P[:, j]
    oof['fold'] = a.index
    oof.to_csv(f'{OUT}/oof.csv', index=False)
json.dump(dict(best=best, epochs_run=len(hist), minutes=(time.time() - t_start) / 60), open(f'{OUT}/done.json', 'w'), indent=1)
log(f'done | best epoch {best["epoch"]} | {(time.time() - t_start) / 60:.0f} min')
