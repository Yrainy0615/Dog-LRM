#!/usr/bin/env python3
"""Prune canonical-space fringe/floater splats from the shba701 GS.

Removes gaussians sitting above the plausible fur length of their body part
(same per-part caps as the motion freeze in animate_fur_wind.py): hard prune
when >5mm beyond the cap, or beyond the cap at all with near-zero opacity.
These render as dark scraggly wisps against black backgrounds.

Writes exps/coatdepth_demo/shba701_{gs_pruned.ply, h_pruned.npy,
weights_pruned.json} for use via GS_PLY/H_NPY/WJ_JSON env overrides.
"""
import json, sys
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, ".")
import animate_gs_coatdepth as base
from animate_fur_wind import knn_smooth

DATA, OUT = "train_data", "exps/coatdepth_demo"
import os
SRC_PLY = os.environ.get("SRC_PLY", f"{DATA}/shba701_canonical_gs_7k.ply")
SRC_H = os.environ.get("SRC_H", f"{OUT}/shba701_h.npy")
SRC_WJ = os.environ.get("SRC_WJ", f"{DATA}/shba701_vertex_weights.json")
OUT_PFX = os.environ.get("OUT_PFX", f"{OUT}/shba701")

ply = PlyData.read(SRC_PLY)
v = ply["vertex"]
xyz_z = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
xyz = np.stack([xyz_z[:, 0], xyz_z[:, 2], -xyz_z[:, 1]], 1)          # Z-up -> Y-up
op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))

h = np.load(SRC_H)
h_s = knn_smooth(h, xyz)

nodes, joint_ids, ibm, verts, vj, vw = base.parse_glb(f"{DATA}/shba701_mesh_7k_noroot.glb")
wj = json.load(open(SRC_WJ))
weights = wj["weights"]
K = max(len(x) for x in weights)
bid = np.zeros((len(weights), K), np.int64)
bw = np.zeros((len(weights), K), np.float32)
for i, x in enumerate(weights):
    for k, (vi, wv) in enumerate(x):
        bid[i, k], bw[i, k] = vi, wv
bw /= bw.sum(1, keepdims=True).clip(1e-8)


def part_hcap(name):
    n = name.lower()
    if n.startswith("tail"): return 0.060
    if "ear" in n: return 0.030
    if any(k in n for k in ["head", "nose", "eye", "mouth", "tongue"]): return 0.010
    if any(k in n for k in ["claw", "foot", "shin", "leg", "thigh", "hip"]): return 0.015
    return 0.025


hc = np.array([part_hcap(nodes[j].get("name", "")) for j in joint_ids], np.float32)
vcap = (vw * hc[vj]).sum(1)
gcap = knn_smooth((bw * vcap[bid]).sum(1), xyz, k=16, iters=1)
excess = h_s - gcap

drop = (excess > 0.005) | ((excess > 0.0) & (op < 0.15)) | (op < 0.12)
keep = ~drop
print(f"total {len(keep)}  drop {drop.sum()} ({drop.mean()*100:.1f}%)  "
      f"[>cap+5mm: {(excess>0.005).sum()}, low-op fringe: {((excess>0)&(op<0.15)).sum()}, "
      f"ultra-sparse op<0.12: {(op<0.12).sum()}]")

el = PlyElement.describe(np.asarray(v.data)[keep], "vertex")
PlyData([el]).write(f"{OUT_PFX}_gs_pruned.ply")
np.save(f"{OUT_PFX}_h_pruned.npy", h[keep])
wj["weights"] = [weights[i] for i in np.where(keep)[0]]
wj["num_gaussians"] = int(keep.sum())
json.dump(wj, open(f"{OUT_PFX}_weights_pruned.json", "w"))
print("saved pruned ply / h / weights to", OUT)
