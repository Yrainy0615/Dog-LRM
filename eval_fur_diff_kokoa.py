#!/usr/bin/env python3
"""Single held-out dog (kokoa) direction check for the fur diffusion prior.
Bypasses the train-dog mean baseline (train student_data was deleted); instead
reports a naive 'strand == surface normal' floor as the no-grooming reference."""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from train_fur_diff import DiT, Codec, build_geo
from sample_fur_diff import predict, strand_stats, stats_err

dev = "cuda"


def dir_deg(dp, dg):
    return float(torch.rad2deg(torch.acos((dp * dg).sum(-1).clamp(-1, 1))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exps/fur_diff_v2/diff.pt")
    ap.add_argument("--npz", default="synth_fur/student_data/00100-kokoa.npz")
    ap.add_argument("--codec", default="synth_fur/strand_codec.npz")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=dev)
    A = ck["args"]
    codec = Codec(args.codec)
    net = DiT(zdim=codec.C, geo=ck["net"]["cgeo.weight"].shape[1], d=A["d"],
              depth=A["depth"], heads=max(A["d"] // 64, 4)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()

    z = np.load(args.npz)
    z0 = torch.from_numpy(z["z"]).to(dev)
    R = torch.from_numpy(z["anchors"]).to(dev)
    Rn = torch.from_numpy(z["normals"]).to(dev)
    diag = float(z["diag"])
    feats = torch.from_numpy(z["feats"])
    vis = torch.from_numpy(z["vis"])
    cls = torch.from_numpy(z["cls"])
    gt_pts = codec.decode(z0)

    Lg, dg, cg = strand_stats(gt_pts)
    # naive no-grooming floor: strand points straight along surface normal
    naive_dir = torch.nn.functional.normalize(Rn, dim=-1)
    print(f"[kokoa] {R.shape[0]} anchors | codec recon {float(z['rec_err']):.4f} xdiag", flush=True)
    print(f"  NAIVE (strand==normal) dir err vs GT: {dir_deg(naive_dir, dg):5.1f} deg", flush=True)

    for k in [1, 2, 4]:
        vids = np.linspace(0, feats.shape[0] - 1, k).astype(int)
        f = feats[vids].to(dev).float()
        v = vis[vids].to(dev).float()
        w = v[:, :, None]
        feat = (f * w).sum(0) / w.sum(0).clamp(min=1)
        visfrac = v.max(0).values
        feat = feat * visfrac[:, None]
        geo = build_geo(R, Rn, diag, visfrac)
        gl = cls[vids].to(dev).float().mean(0)
        zs = predict(net, (feat, geo, gl), codec.C, A)
        pred_pts = codec.decode(zs)
        se = stats_err(pred_pts, gt_pts, diag)
        err = (pred_pts - gt_pts).norm(dim=-1)
        occ = float(err[visfrac < 0.5].mean()) if (visfrac < 0.5).any() else 0.0
        nocc = int((visfrac < 0.5).sum())
        print(f"  DIFFUSION k={k}: dir {se[1]:5.1f}deg | len {se[0]:.4f} curl {se[2]:.3f} | "
              f"point {float(err.mean()):.4f} occ {occ:.4f}({nocc} anchors)", flush=True)


if __name__ == "__main__":
    main()
