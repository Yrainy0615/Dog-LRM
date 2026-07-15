#!/usr/bin/env python3
"""Quick visual QC of fitted SMAL pseudo-GT: project posed verts into a few COLMAP
views and overlay on the image. One montage row per scene (N views).

  python preprocess/qc_fit.py --scenes <colmap_dir> ... --out qc.png
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.smal_model import SMALModel, load_pseudo_gt


def project(verts, c2w, K):
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ verts.T + w2c[:3, 3:4]).T          # [V,3] cam frame
    z = np.clip(cam[:, 2:3], 1e-4, None)
    uv = (K[:3, :3] @ (cam / z).T).T[:, :2]                 # [V,2] pixels
    return uv, cam[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--down", type=int, default=6, help="image downscale for the montage")
    ap.add_argument("--out", default="qc_fit.png")
    ap.add_argument("--head_color", action="store_true",
                    help="color skull/muzzle verts red (head-tail flip check)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smal = SMALModel(dev)
    head = None
    if args.head_color:
        import pickle
        import scipy.sparse as sp
        d = pickle.load(open("third_party/barc_release/data/smal_data/my_smpl_SMBLD_nbj_v3.pkl",
                             "rb"), encoding="latin1")
        Wsk = d["weights"]
        Wsk = Wsk.toarray() if sp.issparse(Wsk) else np.asarray(Wsk)
        head = (Wsk[:, [15, 16, 32, 33, 34]].sum(1) > 0.5)  # neck/skull/head/ears
    rows = []
    for sc in args.scenes:
        gt = load_pseudo_gt(sc, "preprocess", smal.num_betas, dev)
        V = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"],
                             gt["trans"], gt["scale"])[0].detach().cpu().numpy()
        frames = json.load(open(os.path.join(sc, "preprocess", "cameras.json")))["frames"]
        idx = np.linspace(0, len(frames) - 1, args.views).astype(int)
        tiles = []
        for j in idx:
            fr = frames[j]
            img = Image.open(os.path.join(sc, fr["image_path"])).convert("RGB")
            d = args.down
            im = np.asarray(img.resize((fr["width"] // d, fr["height"] // d))).copy()
            K = np.array([[fr["fx"] / d, 0, fr["cx"] / d],
                          [0, fr["fy"] / d, fr["cy"] / d], [0, 0, 1]])
            uv, z = project(V, np.array(fr["c2w"]), K)
            H, W = im.shape[:2]
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            px = uv[ok].astype(int)
            for dy in (-1, 0, 1):                             # 3x3 dots for visibility
                for dx in (-1, 0, 1):
                    q = np.clip(px + [dx, dy], 0, [W - 1, H - 1])
                    if head is not None:
                        hk = head[ok]
                        im[q[~hk, 1], q[~hk, 0]] = [0, 255, 0]    # body green
                        im[q[hk, 1], q[hk, 0]] = [255, 0, 0]      # head red
                    else:
                        im[q[:, 1], q[:, 0]] = [0, 255, 0]        # green verts
            tiles.append(im)
        h = min(t.shape[0] for t in tiles)
        rows.append(np.concatenate([t[:h] for t in tiles], axis=1))
        print(f"{os.path.basename(os.path.dirname(sc))}: {len(V)} verts projected")
    w = min(r.shape[1] for r in rows)
    Image.fromarray(np.concatenate([r[:, :w] for r in rows], axis=0)).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
