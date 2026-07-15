#!/usr/bin/env python3
"""Export D-SMAL posed MESH (.obj, the body surface) + STRAND polylines (.obj lines) in the same world
coords, so the geometry (root-on-surface? strand hugging or floating?) can be inspected in Blender/MeshLab
BEFORE worrying about colour."""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, ".")
from dog_lrm.smal_model import SMALModel, load_pseudo_gt
from train_fur_v6flow import FurV6Flow
ap = argparse.ArgumentParser()
ap.add_argument("--dog", default="00085-kotori"); ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
ap.add_argument("--pt", required=True, help="*_final.pt with fur_sd")
ap.add_argument("--out", default="exps/geom"); ap.add_argument("--max_strands", type=int, default=60000)
a = ap.parse_args(); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
scene = os.path.join(a.root, a.dog, "colmap")

# ---- D-SMAL posed mesh from dsmal_anchors (SAME source as fur_anchors roots; gt-pseudo posed is a DIFFERENT pose) ----
da = np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))
V = da["posed"].astype(np.float64); faces = da["faces"].astype(np.int64)
mesh_obj = os.path.join(a.out, f"{a.dog}_mesh.obj")
with open(mesh_obj, "w") as f:
    f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
    f.write("".join(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in faces))
print(f"[geom] mesh: {len(V)} verts {len(faces)} faces -> {mesh_obj}", flush=True)

# ---- strands from final.pt ----
ck = torch.load(a.pt, map_location=dev, weights_only=False)
fur_sd = ck["fur_sd"]; Kp = int(ck["Kp"]); diag = float(ck["diag"])
Nr = fur_sd["roots"].shape[0]; dum = lambda *s: torch.zeros(*s, device=dev)
fur = FurV6Flow(dum(Nr, 3), dum(Nr, 3), dum(Nr, 3), dum(Nr, 3), dum(Nr), dum(Nr), dum(Nr), dum(Nr),
                dum(Nr, 3), diag, dum(3), 0, Kp=Kp, off=0.0, curl_override=(0.0, 0.0, 0.2), clump_amt=0.0).to(dev)
fur.load_state_dict(fur_sd, strict=False)
fur.d_root.data.zero_()                                             # freeze d_root: roots EXACTLY on the uniform-sampled surface points (no learned drift)
with torch.no_grad():
    pts = fur.strand_points().cpu().numpy()                          # [Nr,Kp,3]
    op = fur.root_opacity().cpu().numpy()
    nrm = fur.n.cpu().numpy()
keep = op > 0.05; pts = pts[keep]; nrm = nrm[keep]                   # visible strands (drop nofur: face/ear/paw)
print(f"[geom] strands: {len(pts)}/{Nr} visible (Kp={Kp})", flush=True)
from scipy.spatial import cKDTree
surf = np.concatenate([V, V[faces].mean(1)], 0); tree = cKDTree(surf)   # verts + face centers as surface proxy
droot, _ = tree.query(pts[:, 0, :]); on = droot < 0.025 * diag
print(f"[geom] roots ON surface (<0.025diag): {int(on.sum())}/{len(pts)} = {100*on.mean():.1f}%  (off: {int((~on).sum())}, med {np.median(droot[~on])/diag if (~on).any() else 0:.3f}diag)", flush=True)
def write_obj(P3, path):
    if len(P3) > a.max_strands: P3 = P3[np.random.default_rng(0).choice(len(P3), a.max_strands, replace=False)]
    lines = []; vi = 1
    for s in P3:
        for p in s: lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        lines.append("l " + " ".join(str(vi + k) for k in range(len(s))) + "\n"); vi += len(s)
    open(path, "w").write("".join(lines)); print(f"[geom] {len(P3)} -> {path}", flush=True)
write_obj(pts[on], os.path.join(a.out, f"{a.dog}_strands_onsurf.obj"))
write_obj(pts[~on], os.path.join(a.out, f"{a.dog}_strands_offsurf.obj"))
