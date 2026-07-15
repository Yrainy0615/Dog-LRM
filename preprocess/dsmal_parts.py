#!/usr/bin/env python3
"""v6: D-SMAL semantic part segmentation (from 35-joint skin weights) + per-part fur-length
audit. Pinpoints which parts/dogs the reconstruction over-lengthens (e.g. legs/paws).

  PATH=$ENV/bin:$PATH python preprocess/dsmal_parts.py --dogs 00148-uta 00062-bear
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("preprocess"))
from dsmal_region_masks import load_smal
from dog_lrm.smal_model import build_subdiv

# joint groups (from fur_strand_init LEG/TAIL, dsmal_region_masks FACE, animate EAR, VLM legs)
PARTS = {
    "body":      [0, 1, 2, 3, 4, 5, 6],
    "head":      [15],
    "muzzle":    [16, 32],
    "ear":       [33, 34],
    "tail":      [25, 26, 27, 28, 29, 30, 31],
    "leg_upper": [7, 8, 11, 12, 17, 18, 21, 22],
    "leg_mid":   [9, 13, 19, 23],
    "paw":       [10, 14, 20, 24],
}
ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
VLM = "exps/vlm_priors"


def part_labels(dev="cpu"):
    """per-(subdivided)-vertex part id [15550] + name list, by argmax of grouped skin weights."""
    smal = load_smal()
    W = smal.weights if torch.is_tensor(smal.weights) else torch.tensor(np.asarray(smal.weights))
    W = W.float()                                                    # [3889, Nj]
    faces = (smal.faces if torch.is_tensor(smal.faces) else torch.tensor(np.asarray(smal.faces))).long()
    M = build_subdiv(faces, 1, dev)
    Ws = torch.sparse.mm(M, W)                                       # [15550, Nj]
    names = list(PARTS)
    grouped = torch.stack([Ws[:, js].sum(1) for js in PARTS.values()], 1)  # [15550, P]
    return grouped.argmax(1).numpy(), names, grouped.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogs", nargs="+", default=["00148-uta", "00062-bear", "00029-oto"])
    ap.add_argument("--out", default="synth_fur/dsmal_parts.npz")
    args = ap.parse_args()

    pid, names, grouped = part_labels()                             # grouped [15550,P] soft weights
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, part_id=pid.astype(np.int16), names=np.array(names),
             part_weights=grouped.astype(np.float32))
    W = grouped / (grouped.sum(0, keepdims=True) + 1e-9)            # normalize per part -> soft mean

    for dog in args.dogs:
        fa = np.load(os.path.join(ROOT, dog, "colmap/preprocess/fur_anchors.npz"))
        L, Lg, diag = fa["L"], fa["L_geo"], float(fa["diag"])
        vp = os.path.join(VLM, dog + ".json")
        bbox_cm = json.load(open(vp))["dog_bbox_diag_cm"] if os.path.exists(vp) else None
        cm = (lambda x: f"{x/diag*bbox_cm:4.1f}cm") if bbox_cm else (lambda x: "  ?  ")
        print(f"\n=== {dog} (diag {diag:.3f}, bbox {bbox_cm}cm) | soft-weighted per-part fur length ===")
        print(f"{'part':10s} {'L(VLM)/diag':>12s} {'L_geo/diag':>11s} {'L cm':>8s} {'L_geo cm':>9s}")
        for k, n in enumerate(names):
            w = W[:, k]
            lv, lg = float((w * L).sum()), float((w * Lg).sum())     # weighted means
            print(f"{n:10s} {lv/diag*100:10.1f}% {lg/diag*100:9.1f}% {cm(lv):>8s} {cm(lg):>9s}")


if __name__ == "__main__":
    main()
