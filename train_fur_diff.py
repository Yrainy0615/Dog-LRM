#!/usr/bin/env python3
"""Token-DiT strand-latent diffusion with SINGLE/SPARSE-view conditioning.

Tokens = fixed anchor roots on the D-SMAL template (FPS over verts). Per-token state =
whitened strand-codec latent z (train_strand_codec.py). Self-attention over all anchors
gives globally coherent grooming (replaces DiffLocks' scalp-UV UNet; no UV seams).

Conditioning per training sample = random 1..max_views of the groom's 8 views:
  - pixel-aligned frozen-DINOv2 feature per anchor, averaged over views where the anchor
    is VISIBLE (gsplat vertex-splat depth test -- same code path will serve real D-SMAL
    fits at inference, where Blender depth doesn't exist);
  - visibility fraction flag (anchors seen by no view get zero feature -> the prior +
    attention must hallucinate them: exactly the single-view occluded-side case);
  - global CLS style token (mean over chosen views) fused into the timestep embedding;
  - fourier(canonical pos) + normal + L_geo/w_ear scalars.

  python train_fur_diff.py --data synth_fur/dataset --codec synth_fur/strand_codec.npz
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from PIL import Image
from train_strand_predictor import vnormals, tbn_frames, project, fourier

dev = "cuda"


def fps(x, n, seed=0):
    g = torch.Generator(device=x.device.type).manual_seed(seed)
    idx = torch.zeros(n, dtype=torch.long, device=x.device)
    idx[0] = torch.randint(x.shape[0], (1,), generator=g, device=x.device)
    d = (x - x[idx[0]]).norm(dim=-1)
    for i in range(1, n):
        idx[i] = d.argmax()
        d = torch.minimum(d, (x - x[idx[i]]).norm(dim=-1))
    return idx


def splat_depth(V, K, c2w, res, diag, blender_cam=True):
    """z-buffer proxy: render mesh verts as small opaque gaussians, depth mode"""
    from gsplat import rasterization
    flip = torch.diag(torch.tensor([1., -1., -1., 1.], device=dev))
    viewmat = (flip @ torch.inverse(c2w)) if blender_cam else torch.inverse(c2w)
    n = V.shape[0]
    quats = torch.zeros(n, 4, device=dev); quats[:, 0] = 1
    scales = torch.full((n, 3), 0.006 * diag, device=dev)
    op = torch.ones(n, device=dev)
    col = torch.zeros(n, 3, device=dev)
    out, _, _ = rasterization(V, quats, scales, op, col, viewmat[None], K[None], res, res,
                              render_mode="ED", backgrounds=torch.full((1, 1), 1e6, device=dev))
    return out[0, :, :, 0]                                            # [res,res] expected depth


def anchor_visibility(R, K, c2w, dmap, res, diag, blender_cam=True):
    uv, z = project(R, K, c2w) if blender_cam else project_cv(R, K, c2w)
    gu = (uv / res * 2 - 1).clamp(-1, 1)
    d = F.grid_sample(dmap[None, None], gu[None, :, None, :], align_corners=False)[0, 0, :, 0]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < res) & (uv[:, 1] >= 0) & (uv[:, 1] < res)
    return inb & (z > 0) & (z <= d + 0.02 * diag)


def project_cv(P, K, c2w):
    w2c = torch.inverse(c2w)
    cam = (w2c[:3, :3] @ P.T + w2c[:3, 3:4]).T
    z = cam[:, 2].clamp(min=1e-4)
    uv = (K @ (cam / z[:, None]).T).T[:, :2]
    return uv, cam[:, 2]


class Codec:
    def __init__(self, path):
        z = np.load(path)
        self.mean = torch.from_numpy(z["mean"]).to(dev)
        self.comps = torch.from_numpy(z["comps"]).to(dev)             # [C, K*3]
        self.scale = torch.from_numpy(z["scale"]).to(dev)             # [C]
        self.K, self.diag = int(z["K"]), float(z["diag"])
        self.C = self.comps.shape[0]
    def encode(self, strands_local):                                  # [N,K,3] -> [N,C] whitened
        x = strands_local.reshape(strands_local.shape[0], -1) - self.mean
        return (x @ self.comps.T) / self.scale
    def decode(self, z):                                              # [N,C] -> [N,K,3]
        x = (z * self.scale) @ self.comps + self.mean
        return x.view(-1, self.K, 3)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
    def forward(self, x, c):                                          # x [B,N,d], c [B,d]
        s1, b1, g1, s2, b2, g2 = self.mod(c)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, zdim, feat=1024, geo=49, d=384, depth=8, heads=6):
        super().__init__()
        self.zin = nn.Linear(zdim, d)
        self.cfeat = nn.Sequential(nn.Linear(feat, d), nn.SiLU(), nn.Linear(d, d))
        self.cgeo = nn.Linear(geo, d)
        self.cglob = nn.Sequential(nn.Linear(feat, d), nn.SiLU(), nn.Linear(d, d))
        self.tdim = d
        self.tmlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_n = nn.LayerNorm(d, elementwise_affine=False)
        self.out = nn.Linear(d, zdim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def temb(self, t):
        f = torch.exp(torch.arange(self.tdim // 2, device=t.device) * -(math.log(10000) / (self.tdim // 2 - 1)))
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], -1)
    def forward(self, z, t, feat, geo, glob):
        # z [B,N,C]  feat [B,N,1024]  geo [B,N,geo]  glob [B,1024]
        x = self.zin(z) + self.cfeat(feat) + self.cgeo(geo)
        c = self.tmlp(self.temb(t)) + self.cglob(glob)
        for b in self.blocks:
            x = b(x, c)
        return self.out(self.out_n(x))


def build_geo(R, Rn, diag, visfrac):
    return torch.cat([fourier(R / diag), Rn, visfrac[:, None]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth_fur/dataset")
    ap.add_argument("--template", default="synth_fur/blender_input.npz")
    ap.add_argument("--codec", default="synth_fur/strand_codec.npz")
    ap.add_argument("--nanchor", type=int, default=4096)
    ap.add_argument("--max_views", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--heldout_color", type=int, default=4)
    ap.add_argument("--real_data", default="synth_fur/student_data")
    ap.add_argument("--p_real", type=float, default=0.6, help="prob of a real-dog step (vs synth)")
    ap.add_argument("--cond_drop", type=float, default=0.1, help="prob of dropping image cond (enables CFG)")
    ap.add_argument("--regress", action="store_true", help="deterministic z0 regression instead of diffusion")
    ap.add_argument("--heldout_split", default="v6_heldout_split.json")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--out", default="exps/fur_diff")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tz = np.load(args.template)
    V = torch.from_numpy(tz["verts"].astype(np.float32)).to(dev)
    Fc = torch.from_numpy(tz["faces"].astype(np.int64)).to(dev)
    Nrm = vnormals(V, Fc)
    TBN = tbn_frames(V, Nrm)
    diag = float((V.max(0).values - V.min(0).values).norm())
    lgeo_v = torch.from_numpy(tz["L_geo"].astype(np.float32)).to(dev)
    wear_v = torch.from_numpy(tz["w_ear"].astype(np.float32)).to(dev)

    aidx = fps(V, args.nanchor)
    R, Rn, Rtbn = V[aidx], Nrm[aidx], TBN[aidx]
    lgeo, wear = lgeo_v[aidx] / max(float(lgeo_v.max()), 1e-6), wear_v[aidx]
    codec = Codec(args.codec)
    assert abs(codec.diag - diag) / diag < 1e-3

    from transformers import AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    nsd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    ph = args.res // 14

    @torch.no_grad()
    def dfeat(img):                                                   # -> patch featmap [1,1024,ph,ph], cls [1024]
        x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float()
        x = (F.interpolate(x, (args.res, args.res), mode="bilinear", align_corners=False) - nm) / nsd
        o = dino(pixel_values=x).last_hidden_state
        return o[:, 1:].transpose(1, 2).reshape(1, -1, ph, ph), o[0, 0]

    # ---------- precompute per (groom,view): per-anchor aligned feature + visibility; per groom: GT z
    train, evl = [], []
    for gp in sorted(glob.glob(os.path.join(args.data, "*.npz"))):
        z = np.load(gp, allow_pickle=True)
        name = os.path.basename(gp)[:-4]
        ci = int(name.split("_")[1])
        strands = torch.from_numpy(z["strands"]).to(dev)
        roots = torch.from_numpy(z["roots"]).to(dev)
        nn_ = torch.cdist(R, roots).argmin(1)
        loc = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), strands[nn_] - R[:, None]) / diag
        z0 = codec.encode(loc)                                        # [nanchor, C]
        Ks = torch.from_numpy(z["Ks"]).to(dev)
        c2ws = torch.from_numpy(z["c2ws"]).to(dev)
        feats, viss, clss = [], [], []
        for vi in range(len(z["imgs"])):
            img = np.array(Image.open(os.path.join(args.data, str(z["imgs"][vi]))).convert("RGB")) / 255.
            res_img = img.shape[0]
            fm, cls = dfeat(img)
            uv, _ = project(R, Ks[vi], c2ws[vi])
            gu = (uv / res_img * 2 - 1)[None, :, None, :]
            f = F.grid_sample(fm, gu, align_corners=False)[0, :, :, 0].T          # [nanchor,1024]
            dmap = splat_depth(V, Ks[vi], c2ws[vi], res_img, diag)
            vis = anchor_visibility(R, Ks[vi], c2ws[vi], dmap, res_img, diag)
            feats.append(f.half().cpu()); viss.append(vis.cpu()); clss.append(cls.half().cpu())
        rec = dict(z0=z0.cpu(), feats=torch.stack(feats), vis=torch.stack(viss),
                   cls=torch.stack(clss), style=str(z["style"]), name=name)
        (evl if ci == args.heldout_color else train).append(rec)
        print(f"[prep] {name}: vis/view {rec['vis'].float().mean(1).mul(100).int().tolist()}%", flush=True)
    C = codec.C
    print(f"[diff] train {len(train)} eval {len(evl)} synth grooms | anchors {args.nanchor} zdim {C}", flush=True)

    real_train, real_evl = [], []
    if args.real_data and os.path.isdir(args.real_data):
        test_dogs = set(json.load(open(args.heldout_split))["test"])
        for rp in sorted(glob.glob(os.path.join(args.real_data, "*.npz"))):
            dog = os.path.basename(rp)[:-4]
            z = np.load(rp)
            rec = dict(dog=dog,
                       z0=torch.from_numpy(z["z"]),
                       feats=torch.from_numpy(z["feats"]),            # [V,N,1024] fp16 cpu
                       vis=torch.from_numpy(z["vis"]),
                       cls=torch.from_numpy(z["cls"]),
                       geo_base=build_geo(torch.from_numpy(z["anchors"]).to(dev),
                                          torch.from_numpy(z["normals"]).to(dev),
                                          float(z["diag"]),
                                          torch.zeros(z["anchors"].shape[0], device=dev))[:, :-1].cpu())
            (real_evl if dog in test_dogs else real_train).append(rec)
        print(f"[diff] real dogs: train {len(real_train)} heldout {len(real_evl)}", flush=True)
    torch.save(dict(train_names=[r["name"] for r in train], evl_names=[r["name"] for r in evl]),
               os.path.join(args.out, "split.pt"))

    def make_cond_real(rec, view_ids):
        f = rec["feats"][view_ids].to(dev).float()
        v = rec["vis"][view_ids].to(dev).float()
        w = v[:, :, None]
        feat = (f * w).sum(0) / w.sum(0).clamp(min=1)
        visfrac = v.max(0).values
        feat = feat * visfrac[:, None]
        glob = rec["cls"][view_ids].to(dev).float().mean(0)
        geo = torch.cat([rec["geo_base"].to(dev), visfrac[:, None]], -1)
        return feat, geo, glob

    def make_cond(rec, view_ids):
        f = rec["feats"][view_ids].to(dev).float()                    # [k,N,1024]
        v = rec["vis"][view_ids].to(dev).float()                      # [k,N]
        w = v[:, :, None]
        feat = (f * w).sum(0) / w.sum(0).clamp(min=1)                 # mean over visible views
        visfrac = v.max(0).values                                     # seen by any view?
        feat = feat * visfrac[:, None]
        glob = rec["cls"][view_ids].to(dev).float().mean(0)
        geo = build_geo(R, Rn, diag, visfrac)
        return feat, geo, glob

    net = DiT(zdim=C, geo=build_geo(R, Rn, diag, lgeo).shape[1], d=args.d, depth=args.depth, heads=max(args.d // 64, 4)).to(dev)
    print(f"[diff] params {sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    betas = torch.linspace(1e-4, 0.02, args.T, device=dev)
    ac = torch.cumprod(1 - betas, 0)

    nview_total = train[0]["feats"].shape[0]
    for it in range(args.iters + 1):
        use_real = real_train and np.random.rand() < args.p_real
        rec = real_train[np.random.randint(len(real_train))] if use_real else train[np.random.randint(len(train))]
        k = np.random.randint(1, args.max_views + 1)
        vids = np.random.choice(rec["feats"].shape[0], k, replace=False)
        feat, geo, gl = (make_cond_real if use_real else make_cond)(rec, vids)
        if args.cond_drop > 0 and np.random.rand() < args.cond_drop:
            feat = torch.zeros_like(feat); gl = torch.zeros_like(gl)  # unconditional step -> CFG at sampling
        z0 = rec["z0"].to(dev)[None]
        if args.regress:
            pred = net(torch.zeros_like(z0), torch.zeros(1, device=dev, dtype=torch.long),
                       feat[None], geo[None], gl[None])
            loss = F.mse_loss(pred, z0)
        else:
            t = torch.randint(0, args.T, (1,), device=dev)
            eps = torch.randn_like(z0)
            a = ac[t][:, None, None]
            zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
            loss = F.mse_loss(net(zt, t, feat[None], geo[None], gl[None]), eps)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"it{it:6d} k={k} {'real' if use_real else 'synth'} loss={float(loss):.5f}", flush=True)
        if it % 5000 == 0 and it > 0:
            torch.save(dict(net=net.state_dict(), aidx=aidx.cpu(), args=vars(args)),
                       os.path.join(args.out, "diff.pt"))
    torch.save(dict(net=net.state_dict(), aidx=aidx.cpu(), args=vars(args)),
               os.path.join(args.out, "diff.pt"))
    print(f"[diff] saved -> {args.out}/diff.pt", flush=True)


if __name__ == "__main__":
    main()
