#!/usr/bin/env python3
"""Render the fur the diffusion prior 'grows' on held-out dog kokoa.
Diffusion strands (cyan) vs teacher-GT strands (red) overlaid on real photos,
several views. Reuses prep_student_data's exact surface/anchor reconstruction."""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from optimize_fur_dog import SmalRig, load_scene, scale_K
from train_fur_diff import DiT, Codec, build_geo, splat_depth, anchor_visibility, fps
from train_strand_predictor import vnormals, tbn_frames
from sample_fur_diff import predict, interp_latent
from prep_student_data import resample_polyline

dev = "cuda"
ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
DOG = "00100-kokoa"
NANCHOR = 4096
OUT = "exps/fur_diff_v2/kokoa_render"
os.makedirs(OUT, exist_ok=True)


def to_world(local, R, Rtbn, diag):
    return R[:, None] + torch.einsum("rij,rkj->rki", Rtbn, local * diag)


def proj(pts, K, c2w, W, H):
    w2c = torch.inverse(c2w)
    cam = (w2c[:3, :3] @ pts.reshape(-1, 3).T + w2c[:3, 3:4]).T
    zc = cam[:, 2].clamp(min=1e-4)
    uv = (K @ (cam / zc[:, None]).T).T[:, :2]
    return uv.view(pts.shape[0], pts.shape[1], 2).cpu(), cam[:, 2].view(pts.shape[0], pts.shape[1]).cpu()


def draw(img, uv_list_colors, path):
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7 * H / W), dpi=130)
    ax.imshow(img)
    for uv, z, color in uv_list_colors:
        for s, sz in zip(uv, z):
            if (sz > 0).all():
                ax.plot(s[:, 0], s[:, 1], lw=0.35, c=color, alpha=0.55)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    fig.tight_layout(pad=0); fig.savefig(path); plt.close(fig)
    print(f"[render] {path}", flush=True)


def main():
    codec = Codec("synth_fur/strand_codec.npz")
    T = torch.load(f"exps/fur_teacher_v8/{DOG}/teacher.pt", map_location=dev)
    strands, roots = T["strands"].to(dev), T["roots"].to(dev)

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

    # GT (teacher) local strands
    S = resample_polyline(strands[aidx], codec.K)
    gt_local = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), S - R[:, None]) / diag
    gt_world = to_world(gt_local, R, Rtbn, diag)

    # conditioning from the cached npz (aligned: same FPS seed)
    z = np.load(f"synth_fur/student_data/{DOG}.npz")
    feats = torch.from_numpy(z["feats"]); vis = torch.from_numpy(z["vis"]); cls = torch.from_numpy(z["cls"])
    view_ids = z["view_ids"]

    ck = torch.load("exps/fur_diff_v2/diff.pt", map_location=dev)
    A = ck["args"]
    net = DiT(zdim=codec.C, geo=ck["net"]["cgeo.weight"].shape[1], d=A["d"],
              depth=A["depth"], heads=max(A["d"] // 64, 4)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()

    k = 2
    vsel = np.linspace(0, feats.shape[0] - 1, k).astype(int)
    f = feats[vsel].to(dev).float(); v = vis[vsel].to(dev).float()
    w = v[:, :, None]; feat = (f * w).sum(0) / w.sum(0).clamp(min=1)
    visfrac = v.max(0).values; feat = feat * visfrac[:, None]
    geo = build_geo(R, Rn, diag, visfrac)
    gl = cls[vsel].to(dev).float().mean(0)
    zs = predict(net, (feat, geo, gl), codec.C, A)
    diff_world = to_world(codec.decode(zs), R, Rtbn, diag)

    # dense fur look: interpolate anchor latents to all surface verts
    zd = interp_latent(zs, R, verts)
    vtbn = TBN
    dense_world = verts[:, None] + torch.einsum("rij,rkj->rki", vtbn, codec.decode(zd) * diag)

    imgs, masks, K0, c2ws, H, W = load_scene(ROOT, DOG)
    full = np.asarray(Image.open(glob.glob(os.path.join(ROOT, DOG, "colmap/preprocess/masks/*"))[0]))
    Ks = scale_K(K0, H, W, full.shape[0], full.shape[1]).to(dev)
    c2ws = c2ws.to(dev)

    render_views = np.linspace(0, len(imgs) - 1, 4).astype(int)
    for vi in render_views:
        img = (imgs[vi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        K, c2w = Ks[vi], c2ws[vi]
        du, dz = proj(diff_world[::4], K, c2w, W, H)
        gu, gz = proj(gt_world[::4], K, c2w, W, H)
        draw(img, [(gu, gz, "red"), (du, dz, "cyan")], os.path.join(OUT, f"v{vi}_compare.png"))
        ddu, ddz = proj(dense_world[::6], K, c2w, W, H)
        draw(img, [(ddu, ddz, "cyan")], os.path.join(OUT, f"v{vi}_diff_dense.png"))
    print("[render] DONE", flush=True)


if __name__ == "__main__":
    main()
