#!/usr/bin/env python3
"""Turntable gif: orbit the camera around the whole dog (composite gaussians from the trained ply) -> see
the full body from all angles."""
import os, sys, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from dog_lrm.render import load_ply, render_gaussians, intrinsics
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--dog", default="00085-kotori")
ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
ap.add_argument("--out", required=True); ap.add_argument("--N", type=int, default=48); ap.add_argument("--res", type=int, default=640)
ap.add_argument("--op_thr", type=float, default=0.0, help="drop gaussians with opacity below this (de-noise low-op speckles)")
a = ap.parse_args(); dev = "cuda"
g = load_ply(a.ply, dev)
if a.op_thr > 0:
    k = g["opacities"] > a.op_thr
    g = {kk: vv[k] for kk, vv in g.items()}
    print(f"[turntable] op_thr {a.op_thr}: kept {int(k.sum())}/{len(k)} gaussians", flush=True)
means = g["means"]
center = means.mean(0); ext = float((means.max(0).values - means.min(0).values).norm())
fa = np.load(os.path.join(a.root, a.dog, "colmap", "preprocess", "fur_anchors.npz"))
grav = torch.tensor(fa["gravity"], device=dev, dtype=torch.float32); up = F.normalize(-grav, dim=0)
ax = torch.tensor([1., 0, 0], device=dev)
if abs(float((ax * up).sum())) > 0.9: ax = torch.tensor([0., 0, 1.], device=dev)
e1 = F.normalize(torch.cross(up, ax, dim=0), dim=0); e2 = F.normalize(torch.cross(up, e1, dim=0), dim=0)
Rv = 0.95 * ext; elev = 0.18 * ext
W = H = a.res; fov = 40 * np.pi / 180; fx = W / (2 * np.tan(fov / 2)); K = intrinsics(fx, fx, W / 2, H / 2, dev)
white = torch.ones(3, device=dev); frames = []
for az in np.linspace(0, 2 * np.pi, a.N, endpoint=False):
    campos = center + Rv * (float(np.cos(az)) * e1 + float(np.sin(az)) * e2) + up * elev
    fwd = F.normalize(center - campos, dim=0); right = F.normalize(torch.cross(fwd, up, dim=0), dim=0); down = torch.cross(fwd, right, dim=0)
    c2w = torch.eye(4, device=dev); c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = campos
    with torch.no_grad():
        rgb, _ = render_gaussians(means, g["quats"], g["scales"], g["opacities"], g["rgb"], c2w, K, W, H, bg=white)
    frames.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=70, loop=0)
print(f"[turntable] {a.N} frames -> {a.out} ({len(means)} gaussians)", flush=True)
