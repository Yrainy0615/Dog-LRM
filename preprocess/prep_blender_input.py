#!/usr/bin/env python3
"""v6 stage-1: build the canonical D-SMAL template Blender will groom on.

Averages the rest-pose (canon) shape + measured per-vertex fur fields (L_geo, w_ear,
w_face) across the held-out TRAIN dogs to get a breed-agnostic template, subdivides
once (matching the fur model's 15550-vertex working mesh), and dumps a single npz the
headless Blender groom script consumes. Averaging L_geo gives a clean canonical fur-
length / baldness profile (true-bald = eyes/nose where every dog measures ~0), far
less noisy than any single dog's fit.

  PATH=$ENV/bin:$PATH python preprocess/prep_blender_input.py --split v6_heldout_split.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_dog_lrm_fur_v2 import list_scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--split", default="v6_heldout_split.json")
    ap.add_argument("--out", default="synth_fur/blender_input.npz")
    args = ap.parse_args()

    train = set(json.load(open(args.split))["train"])
    scenes = [s for s in list_scenes(args.root) if s.split("/")[-2] in train]
    print(f"averaging template over {len(scenes)} train dogs", flush=True)

    canons, lgeo, wear, wface_sub = [], [], [], []
    faces0 = None
    for s in scenes:
        da = np.load(os.path.join(s, "preprocess", "dsmal_anchors.npz"))
        fa = np.load(os.path.join(s, "preprocess", "fur_anchors.npz"))
        canons.append(da["canon"])                                   # [3889,3]
        lgeo.append(fa["L_geo"]); wear.append(fa["w_ear"]); wface_sub.append(fa["w_face"])
        if faces0 is None:
            faces0 = da["faces"]
    canon_mean = np.stack(canons).mean(0)                            # [3889,3] mean rest shape
    L_geo = np.stack(lgeo).mean(0)                                   # [15550]
    w_ear = np.stack(wear).mean(0)
    w_face_sub = np.stack(wface_sub).mean(0)

    # subdivide the mean canonical mesh once -> 15550 verts (fur model's working mesh)
    faces0_t = torch.from_numpy(faces0).long()
    M = build_subdiv(faces0_t, 1, "cpu")
    verts_sub = torch.sparse.mm(M, torch.from_numpy(canon_mean).float()).numpy()   # [15550,3]
    faces_sub = subdivided_faces(faces0_t, 1).numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, verts=verts_sub, faces=faces_sub,
             L_geo=L_geo, w_ear=w_ear, w_face=w_face_sub)
    print(f"verts {verts_sub.shape} faces {faces_sub.shape} | "
          f"L_geo[min {L_geo.min():.4f} max {L_geo.max():.4f}] "
          f"bald(<1e-3): {int((L_geo < 1e-3).sum())} | w_ear>0.5: {int((w_ear > 0.5).sum())}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
