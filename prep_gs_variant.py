#!/usr/bin/env python3
"""Prepare h + gaussian->vertex binding for a GS ply variant of shba701
(e.g. the manually tail-fixed ply, which dropped/reordered gaussians so the
original vertex_weights.json no longer aligns).

h: signed distance to the template mesh surface (same as the original recipe).
binding: K=8 nearest triangles (centroid KDTree), inverse-distance weight per
tri split 1/3 to its vertices, normalized — reproducing the format described in
the original vertex_weights.json.

Usage: SRC_PLY=... OUT_PREFIX=... python prep_gs_variant.py
"""
import json, math, os, sys
import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree

DATA, OUT = "train_data", "exps/coatdepth_demo"
SRC = os.environ.get("SRC_PLY", f"{DATA}/shba701_canonical_gs_7k_fix.ply")
PFX = os.environ.get("OUT_PREFIX", f"{OUT}/shba701_fix")

v = PlyData.read(SRC)["vertex"]
xyz_z = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
xyz = np.stack([xyz_z[:, 0], xyz_z[:, 2], -xyz_z[:, 1]], 1)          # Z-up -> Y-up

mesh = trimesh.load(f"{DATA}/shba701_mesh_7k_noroot.glb", process=False, force="mesh")

# signed h via dense surface sampling
pts, fid = trimesh.sample.sample_surface(mesh, 500000)
d, idx = cKDTree(pts).query(xyz, k=1)
fn = mesh.face_normals[fid[idx]]
sign = np.sign(np.einsum("ij,ij->i", xyz - pts[idx], fn))
h = d * sign
np.save(f"{PFX}_h.npy", h)
print(f"h: med {np.median(h)*1000:.1f}mm  p95 {np.percentile(h,95)*1000:.1f}mm")

# binding: 8 nearest tris by centroid, 1/w = dist, split 1/3 per vertex
tc = mesh.triangles.mean(1)
td, ti = cKDTree(tc).query(xyz, k=8)
tw = 1.0 / np.maximum(td, 1e-6)
tw /= tw.sum(1, keepdims=True)
faces = np.asarray(mesh.faces)
weights = []
for i in range(len(xyz)):
    acc = {}
    for k in range(8):
        for vidx in faces[ti[i, k]]:
            acc[int(vidx)] = acc.get(int(vidx), 0.0) + tw[i, k] / 3.0
    top = sorted(acc.items(), key=lambda x: -x[1])[:10]
    s = sum(w for _, w in top)
    weights.append([[vi, round(w / s, 6)] for vi, w in top])

out = {"description": "Per-gaussian skinning weights onto mesh vertices (K=8 nearest tris, "
                      "split 1/3 to verts, sum~=1). Vertex indices match the accompanying glb.",
       "num_gaussians": len(xyz), "num_vertices": len(mesh.vertices),
       "binding": "gaussian->mesh vertex (2-level: mesh skinned to rig)", "weights": weights}
json.dump(out, open(f"{PFX}_weights.json", "w"))
print(f"saved {PFX}_h.npy / {PFX}_weights.json for {len(xyz)} gaussians")
