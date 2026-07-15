#!/usr/bin/env python3
"""Fit the strand shape CODEC: PCA over root-TBN-local strand polylines (K pts x 3).

Strands from every groom are expressed in their own root's TBN frame (view/pose-independent,
/diag units), then PCA-whitened -> z (C dims, ~unit variance per dim; ready for diffusion).
Decode = z @ comps * scale + mean. Reports per-family recon error at several C.

  python train_strand_codec.py --data synth_fur/dataset --keep 24
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from train_strand_predictor import vnormals, tbn_frames

dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth_fur/dataset")
    ap.add_argument("--template", default="synth_fur/blender_input.npz")
    ap.add_argument("--keep", type=int, default=24, help="latent dims to keep in saved codec")
    ap.add_argument("--max_per_groom", type=int, default=6000)
    ap.add_argument("--out", default="synth_fur/strand_codec.npz")
    args = ap.parse_args()

    tz = np.load(args.template)
    V = torch.from_numpy(tz["verts"].astype(np.float32)).to(dev)
    Fc = torch.from_numpy(tz["faces"].astype(np.int64)).to(dev)
    N = vnormals(V, Fc)
    TBN = tbn_frames(V, N)
    diag = float((V.max(0).values - V.min(0).values).norm())

    Xs, fam = [], []
    g = torch.Generator().manual_seed(0)
    for gp in sorted(glob.glob(os.path.join(args.data, "*.npz"))):
        z = np.load(gp, allow_pickle=True)
        strands = torch.from_numpy(z["strands"]).to(dev)          # [Ns,K,3] world
        roots = torch.from_numpy(z["roots"]).to(dev)
        sel = torch.randperm(strands.shape[0], generator=g)[: args.max_per_groom].to(dev)
        strands, roots = strands[sel], roots[sel]
        nnv = torch.cdist(roots, V).argmin(1)                     # nearest template vert -> TBN
        loc = torch.einsum("rij,rkj->rki", TBN[nnv].transpose(1, 2), strands - roots[:, None]) / diag
        Xs.append(loc.reshape(loc.shape[0], -1).cpu())
        fam += [str(z["style"])] * loc.shape[0]
    X = torch.cat(Xs).to(dev)
    fam = np.array(fam)
    K = X.shape[1] // 3
    print(f"[codec] {X.shape[0]} strands, K={K} pts ({X.shape[1]}d) from {len(Xs)} grooms", flush=True)

    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    var = (S**2) / (X.shape[0] - 1)
    cum = torch.cumsum(var, 0) / var.sum()
    scale = (S / (X.shape[0] - 1) ** 0.5)                          # per-comp std -> whitening

    for C in (8, 16, 24, 32, X.shape[1]):
        C = min(C, X.shape[1])
        Zc = Xc @ Vt[:C].T
        rec = Zc @ Vt[:C] + mean
        err = (rec - X).view(-1, K, 3).norm(dim=-1).mean(1)        # per-strand mean point err (xdiag)
        line = " ".join(f"{f}:{float(err[fam == f].mean()):.5f}" for f in np.unique(fam))
        print(f"[codec] C={C:3d} var={float(cum[C-1]):.4f} err(xdiag) {float(err.mean()):.5f} | {line}", flush=True)

    C = args.keep
    np.savez(args.out, mean=mean.cpu().numpy(), comps=Vt[:C].cpu().numpy(),
             scale=scale[:C].cpu().numpy(), K=K, diag=diag)
    print(f"[codec] saved C={C} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
