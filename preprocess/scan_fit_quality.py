#!/usr/bin/env python3
"""Rank all fitted scenes by SMAL fit quality.

Two metrics per scene (8 views each):
  - iou:      bare-SMAL silhouette (splatted subdiv verts, dilated) vs GT mask
  - head_err: BARC head-keypoint (eyes/mouth/ears/nose) reprojection error,
              normalized by mask bbox diag — catches head-tail flips that IoU misses.
Score = iou - head_err, CSV sorted descending to --out.

  python preprocess/scan_fit_quality.py --root <unzipped/0423> --out exps/fit_quality.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/barc_release/src"))
from dog_lrm.smal_model import SMALModel, load_pseudo_gt
from preprocess.fit_smal import build_kp_regressor, load_keypoints

HEAD_KP = [0, 1, 2, 20, 21, 22]  # eyes, mouth, ears, nose tip
NOSE_KP, TAIL_KP = 22, [7]  # flip test vs tail START (tail tip can curl up to the head)


def project(verts, c2w, K):
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ verts.T + w2c[:3, 3:4]).T
    z = np.clip(cam[:, 2:3], 1e-4, None)
    uv = (K[:3, :3] @ (cam / z).T).T[:, :2]
    return uv, cam[:, 2]


def scan_scene(sc, smal, dev, n_views=8, down=8):
    gt = load_pseudo_gt(sc, "preprocess", smal.num_betas, dev)
    V = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"],
                         gt["trans"], gt["scale"])
    Vsub = smal.subdivide(V)[0].detach().cpu().numpy()
    kp3d = (build_kp_regressor(smal, dev) @ V[0]).detach().cpu().numpy()

    frames = json.load(open(os.path.join(sc, "preprocess", "cameras.json")))["frames"]
    idx = np.linspace(0, len(frames) - 1, n_views).astype(int)
    sub = [frames[j] for j in idx]
    Hd, Wd = sub[0]["height"] // down, sub[0]["width"] // down
    tgt, wt = load_keypoints(sc, sub, "preprocess", (Hd, Wd), dev, conf_thr=0.3)
    tgt, wt = tgt.cpu().numpy(), wt.cpu().numpy()

    ious, herrs, flips = [], [], []
    for i, fr in enumerate(sub):
        mp = os.path.join(sc, "preprocess", "masks",
                          os.path.splitext(fr["name"])[0] + ".png")
        if not os.path.exists(mp):
            continue
        m = np.asarray(Image.open(mp).convert("L").resize((Wd, Hd))) > 127
        K = np.array([[fr["fx"] / down, 0, fr["cx"] / down],
                      [0, fr["fy"] / down, fr["cy"] / down], [0, 0, 1]])
        uv, z = project(Vsub, np.array(fr["c2w"]), K)
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < Hd)
        grid = torch.zeros(Hd, Wd)
        px = uv[ok].astype(int)
        grid[px[:, 1], px[:, 0]] = 1.0
        sil = (F.max_pool2d(grid[None, None], 7, 1, 3)[0, 0] > 0.5).numpy()
        ious.append((sil & m).sum() / max((sil | m).sum(), 1))

        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        diag = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
        w = wt[i, HEAD_KP]
        if w.sum() < 1e-6:
            continue
        uvk, _ = project(kp3d[HEAD_KP], np.array(fr["c2w"]), K)
        err = np.linalg.norm(uvk - tgt[i, HEAD_KP], axis=1) / max(diag, 1)
        herrs.append((w * err).sum() / w.sum())

        # flip test: is the BARC nose target closer to the SMAL tail than the SMAL nose?
        if wt[i, NOSE_KP] > 0:
            uvn, _ = project(kp3d[[NOSE_KP] + TAIL_KP], np.array(fr["c2w"]), K)
            d = np.linalg.norm(uvn - tgt[i, NOSE_KP], axis=1)
            flips.append(float(d[0] > d[1:].min()))

    iou = float(np.mean(ious)) if ious else 0.0
    head_err = float(np.mean(herrs)) if herrs else 0.5  # no confident head view = unknown
    flip_frac = float(np.mean(flips)) if flips else 0.5
    return iou, head_err, flip_frac, len(herrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--out", default="exps/fit_quality.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smal = SMALModel(dev)
    rows = []
    scenes = sorted(glob.glob(os.path.join(args.root, "*", "colmap")))
    for sc in scenes:
        if not os.path.exists(os.path.join(sc, "preprocess", "smal_params.json")):
            continue
        name = os.path.basename(os.path.dirname(sc))
        try:
            iou, herr, flip, nh = scan_scene(sc, smal, dev)
        except Exception as e:
            print(f"{name}: FAIL {e}", flush=True)
            continue
        rows.append((sc, name, iou, herr, flip, nh, iou - herr - 0.5 * flip))
        print(f"{name}: iou={iou:.3f} head_err={herr:.3f} flip={flip:.2f} "
              f"(views w/ head kp: {nh})", flush=True)

    rows.sort(key=lambda r: -r[6])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_dir", "name", "iou", "head_err", "flip_frac", "n_head_views", "score"])
        w.writerows(rows)
    print(f"\nTOP 10 (score = iou - head_err - 0.5*flip_frac):")
    for r in rows[:10]:
        print(f"  {r[1]}: iou={r[2]:.3f} head_err={r[3]:.3f} flip={r[4]:.2f} score={r[6]:.3f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
