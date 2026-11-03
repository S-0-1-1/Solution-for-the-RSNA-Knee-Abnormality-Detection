import math
import numpy as np
import torch
import torch.nn as nn
import timm

PRIOR_BIAS = {0: 0.0, 1: -1.0, 2: -3.0}


class KneeNet(nn.Module):

    def __init__(self, backbone, n_crops, relevance, d=256, n_layers=1, heads=4, drop=0.1):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=3)
        nf = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(nf, d), nn.LayerNorm(d))
        self.crop_emb = nn.Embedding(n_crops, d)
        self.pos = nn.Sequential(nn.Linear(1, d // 4), nn.GELU(), nn.Linear(d // 4, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=drop, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False) if n_layers else None
        R = np.asarray(relevance)
        L = R.shape[0]
        self.q = nn.Parameter(torch.randn(L, d) * 0.02)
        self.bias = nn.Parameter(torch.tensor(np.vectorize(PRIOR_BIAS.get)(R), dtype=torch.float32))
        self.drop = nn.Dropout(0.2)
        self.head_w = nn.Parameter(torch.randn(L, d) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(L))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.d = d

    def forward(self, x, valid, pos):
        B, Cn, K = valid.shape
        flat = x.reshape(B * Cn * K, *x.shape[3:])[valid.reshape(-1)]
        f = self.proj(self.enc((flat - self.mean) / self.std).float())
        tok = torch.zeros(B * Cn * K, self.d, device=x.device, dtype=f.dtype)
        tok[valid.reshape(-1)] = f
        tok = tok.view(B, Cn, K, self.d) + self.crop_emb.weight[None, :, None, :] + self.pos(pos[..., None].float())
        tok = tok.view(B, Cn * K, self.d)
        m = valid.view(B, Cn * K)
        if self.mix is not None:
            tok = self.mix(tok, src_key_padding_mask=~m)
        crop_of = torch.arange(Cn, device=x.device).repeat_interleave(K)
        att = torch.einsum('bnd,ld->bln', tok, self.q) / math.sqrt(self.d) + self.bias[:, crop_of][None]
        att = att.masked_fill(~m[:, None, :], -10000.0).softmax(-1)
        pooled = self.drop(torch.einsum('bln,bnd->bld', att, tok))
        return (pooled * self.head_w).sum(-1) + self.head_b, att


def build(cfg, relevance):
    return KneeNet(cfg['backbone'], cfg['n_crops'], relevance, d=cfg.get('d', 256), n_layers=cfg.get('n_layers', 1),
                   heads=cfg.get('heads', 4))
