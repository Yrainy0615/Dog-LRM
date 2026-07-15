#!/usr/bin/env python3
"""Evaluate the dual-domain fur diffusion on the 9 REAL held-out dogs (never trained).

For k in {1,2,4} evenly-spaced real views: DDIM-sample anchor latents, decode, compare
to the dog's TEACHER strands (pseudo-GT). Baseline = mean train-dog latent. Also reports
occluded-anchor error (visfrac=0) — the generative-completion axis.

  python eval_fur_diff_real.py --ckpt exps/fur_diff/diff.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from train_fur_diff import DiT, Codec, build_geo
from sample_fur_diff import ddim, predict, stats_err

dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exps/fur_diff_v2/diff.pt")
    ap.add_argument("--data", default="synth_fur/student_data")
    ap.add_argument("--codec", default="synth_fur/strand_codec.npz")
    ap.add_argument("--split", default="v6_heldout_split.json")
    ap.add_argument("--view_counts", default="1,2,4")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=dev)
    A = ck["args"]
    codec = Codec(args.codec)
    net = DiT(zdim=codec.C, geo=ck["net"]["cgeo.weight"].shape[1], d=A["d"], depth=A["depth"], heads=max(A["d"] // 64, 4)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()

    test = set(json.load(open(args.split))["test"])
    mean_z, cnt = 0, 0
    evl = []
    for rp in sorted(glob.glob(os.path.join(args.data, "*.npz"))):
        dog = os.path.basename(rp)[:-4]
        z = np.load(rp)
        if dog in test:
            evl.append((dog, z))
        else:
            mean_z = mean_z + torch.from_numpy(z["z"]); cnt += 1
    mean_z = (mean_z / cnt).to(dev)
    print(f"[eval] {len(evl)} held-out dogs | baseline from {cnt} train dogs", flush=True)

    ks = [int(x) for x in args.view_counts.split(",")]
    agg = {k: [] for k in ks}
    for dog, z in evl:
        z0 = torch.from_numpy(z["z"]).to(dev)
        R = torch.from_numpy(z["anchors"]).to(dev)
        Rn = torch.from_numpy(z["normals"]).to(dev)
        diag = float(z["diag"])
        feats = torch.from_numpy(z["feats"])
        vis = torch.from_numpy(z["vis"])
        cls = torch.from_numpy(z["cls"])
        gt_pts = codec.decode(z0)
        base_err = float((codec.decode(mean_z) - gt_pts).norm(dim=-1).mean())
        line = [f"{dog:16s} base {base_err:.4f}"]
        for k in ks:
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
            se = stats_err(codec.decode(zs), gt_pts)
            err = (codec.decode(zs) - gt_pts).norm(dim=-1)
            occ = err[visfrac < 0.5].mean() if (visfrac < 0.5).any() else err.mean() * 0
            agg[k].append((float(err.mean()), base_err, float(occ)) + se)
            line.append(f"k{k} {float(err.mean()):.4f}(occ {float(occ):.4f}, dir {se[1]:.0f}deg)")
        print("  ".join(line), flush=True)

    print("[eval] mean over held-out dogs: pointwise | phase-invariant len/dir/curl", flush=True)
    for k in ks:
        v = np.array(agg[k])
        print(f"  k={k}: diff {v[:,0].mean():.4f} base {v[:,1].mean():.4f} occl {v[:,2].mean():.4f}"
              f" | len {v[:,3].mean():.4f} dir {v[:,4].mean():5.1f}deg curl {v[:,5].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
