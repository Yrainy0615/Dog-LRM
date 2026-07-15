#!/usr/bin/env python3
"""Render the diffusion fur as ACTUAL gaussian geometry (shaded, white bg),
next to the teacher GT fur, on held-out dog kokoa. Body skin gaussians included
so it reads as a furry dog, not floating strands."""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from PIL import Image
from dog_lrm.render import render_gaussians
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from optimize_fur_dog import SmalRig, load_scene, scale_K
from train_fur_strand import quat_align_z
from train_fur_diff import DiT, Codec, build_geo, fps
from train_strand_predictor import vnormals, tbn_frames
from sample_fur_diff import predict, interp_latent
from prep_student_data import resample_polyline

dev = "cuda"
ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
DOG = "00100-kokoa"
NANCHOR = 4096
NDENSE = 80000
OUT = "exps/fur_diff_v2/kokoa_geom"
os.makedirs(OUT, exist_ok=True)


def fur_gauss(world, rgb, diag, rf=0.0006):
    """world [N,K,3] polylines, rgb [N,3] -> gaussian tubes along each segment."""
    p0, p1 = world[:, :-1].reshape(-1, 3), world[:, 1:].reshape(-1, 3)
    mid = (p0 + p1) / 2
    seg = p1 - p0
    sl = seg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    quat = quat_align_z(seg / sl)
    r = rf * diag
    scales = torch.cat([torch.full_like(sl, r), torch.full_like(sl, r), sl * 0.6], -1)
    Kseg = world.shape[1] - 1
    col = rgb[:, None].expand(-1, Kseg, -1).reshape(-1, 3)
    return mid, quat, scales, col


def main():
    codec = Codec("synth_fur/strand_codec.npz")
    T = torch.load(f"exps/fur_teacher_v8/{DOG}/teacher.pt", map_location=dev)
    strands, roots, albedo = T["strands"].to(dev), T["roots"].to(dev), T["albedo"].to(dev)
    bverts, body_rgb = T["bverts"].to(dev), T["body_rgb"].to(dev)

    rig = SmalRig(DOG, ROOT).to(dev)
    with torch.no_grad():
        for k, v in T["rig"].items():
            if hasattr(rig, k):
                getattr(rig, k).copy_(v.to(dev))
        base = rig.verts()
    da = np.load(os.path.join(ROOT, DOG, "colmap", "preprocess", "dsmal_anchors.npz"))
    faces0 = torch.from_numpy(da["faces"].astype(np.int64))
    M = build_subdiv(faces0, 1, "cpu").to(dev)
    verts = torch.sparse.mm(M, base)
    faces = subdivided_faces(faces0, 1).to(dev)
    vn = vnormals(verts, faces)
    TBN = tbn_frames(verts, vn)
    diag = float((verts.max(0).values - verts.min(0).values).norm())

    aidx = fps(roots, NANCHOR)
    R = roots[aidx]
    nnv = torch.cdist(R, verts).argmin(1)
    Rtbn, Rn = TBN[nnv], vn[nnv]

    # diffusion sample @ anchors (k=2, the best)
    z = np.load(f"synth_fur/student_data/{DOG}.npz")
    feats = torch.from_numpy(z["feats"]); vis = torch.from_numpy(z["vis"]); cls = torch.from_numpy(z["cls"])
    ck = torch.load("exps/fur_diff_v2/diff.pt", map_location=dev)
    A = ck["args"]
    net = DiT(zdim=codec.C, geo=ck["net"]["cgeo.weight"].shape[1], d=A["d"],
              depth=A["depth"], heads=max(A["d"] // 64, 4)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()
    vsel = np.linspace(0, feats.shape[0] - 1, 2).astype(int)
    f = feats[vsel].to(dev).float(); v = vis[vsel].to(dev).float()
    w = v[:, :, None]; feat = (f * w).sum(0) / w.sum(0).clamp(min=1)
    visfrac = v.max(0).values; feat = feat * visfrac[:, None]
    geo = build_geo(R, Rn, diag, visfrac)
    gl = cls[vsel].to(dev).float().mean(0)
    zs = predict(net, (feat, geo, gl), codec.C, A)

    # densify to NDENSE roots for a real fur look
    sub = torch.randperm(roots.shape[0], device=dev)[:NDENSE]
    droots = roots[sub]
    dnnv = torch.cdist(droots, verts).argmin(1)
    dtbn = TBN[dnnv]
    drgb = albedo[sub]
    dloc = codec.decode(interp_latent(zs, R, droots))
    diff_world = droots[:, None] + torch.einsum("rij,rkj->rki", dtbn, dloc * diag)

    # teacher GT fur (same roots), teacher strands are already world
    gt_world = resample_polyline(strands[sub], codec.K)

    # body skin gaussians
    bquat = torch.zeros(bverts.shape[0], 4, device=dev); bquat[:, 0] = 1
    bscl = torch.full((bverts.shape[0], 3), 0.004 * diag, device=dev)
    bop = torch.ones(bverts.shape[0], device=dev)

    imgs, masks, K0, c2ws, H, W = load_scene(ROOT, DOG)
    full = np.asarray(Image.open(glob.glob(os.path.join(ROOT, DOG, "colmap/preprocess/masks/*"))[0]))
    Ks = scale_K(K0, H, W, full.shape[0], full.shape[1]).to(dev)
    c2ws = c2ws.to(dev)
    white = torch.ones(3, device=dev)

    def render(world, rgb, vi):
        mid, quat, scales, col = fur_gauss(world, rgb, diag)
        means = torch.cat([mid, bverts]); quats = torch.cat([quat, bquat])
        scl = torch.cat([scales, bscl]); cols = torch.cat([col, body_rgb.clamp(0, 1)])
        ops = torch.cat([torch.full((mid.shape[0],), 0.9, device=dev), bop])
        rgb_img, _ = render_gaussians(means, quats, scl, ops, cols, c2ws[vi], Ks[vi], W, H, bg=white)
        return (rgb_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    for vi in np.linspace(0, len(imgs) - 1, 4).astype(int):
        Image.fromarray(render(diff_world, drgb, vi)).save(os.path.join(OUT, f"v{vi}_diffusion.png"))
        Image.fromarray(render(gt_world, drgb, vi)).save(os.path.join(OUT, f"v{vi}_teacher.png"))
        print(f"[geom] view {vi} done", flush=True)
    print("[geom] DONE", flush=True)


if __name__ == "__main__":
    main()
