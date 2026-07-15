#!/usr/bin/env python3
"""Fuse NeuralFur STRAND GEOMETRY (combed short strands from the GH optimizer) with our cascade's
COLOUR pipeline (nearest-body-gaussian albedo-query + neighbour smoothing), composited over the
colored body skin. Renders multiview-vs-GT + turntable. Frames match (both from dsmal_anchors posed)."""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F, trimesh
sys.path.insert(0, "/home/yyang/mnt/workspace")
from dog_lrm.render import render_gaussians, intrinsics
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--strands", default="NeuralFur/results/kotori_dense/strands_reconstruction/final/10000_pc.ply")
ap.add_argument("--body_pt", default="exps/fur_v11_kotori_clean/00085-kotori_final.pt")
ap.add_argument("--dog", default="00085-kotori")
ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
ap.add_argument("--out", default="exps/neuralfur_colored")
ap.add_argument("--radius_frac", type=float, default=0.0009, help="strand gaussian radius (frac of diag)")
ap.add_argument("--smooth_k", type=int, default=8, help="cross-strand colour smoothing neighbours")
ap.add_argument("--turntable", type=int, default=0, help="N frames for turntable gif (0=skip)")
ap.add_argument("--drop_paws", type=int, default=1, help="drop strands rooted on paws (clean skin paws)")
a = ap.parse_args(); dev = "cuda"; os.makedirs(a.out, exist_ok=True)

# ---- NeuralFur strand geometry ----
pc = trimesh.load(a.strands, process=False)
P = torch.tensor(np.asarray(pc.vertices), dtype=torch.float32, device=dev)   # [Ns*100,3]
P = P.reshape(-1, 100, 3); Ns = P.shape[0]
print(f"[geom] {Ns} strands x100 pts from {os.path.basename(a.strands)}")

# ---- drop strands rooted on paws (SMAL joints 10,14,20,24) -> clean skin paws, no fuzzy white fur ----
if a.drop_paws:
    from dog_lrm.smal_model import SMALModel
    da = np.load(os.path.join(a.root, a.dog, "colmap", "preprocess", "dsmal_anchors.npz"))
    Vp = torch.tensor(da["posed"], dtype=torch.float32, device=dev)
    smal = SMALModel(torch.device(dev), n_subdiv=2); _W = smal.smal.weights
    Wsk = (_W if torch.is_tensor(_W) else torch.tensor(_W)).float().to(dev)
    pawv = (Wsk[:, [10, 14, 20, 24]].sum(1) > 0.5).float()          # per D-SMAL vert paw mask
    roots = P[:, 0]; dd = torch.cdist(roots, Vp); nn = dd.argmin(1)  # root -> nearest D-SMAL vert
    keep = pawv[nn] < 0.5
    P = P[keep]; Ns = P.shape[0]
    print(f"[geom] dropped paw strands -> {Ns} kept")

# ---- body (colored skin) ----
ck = torch.load(a.body_pt, map_location=dev, weights_only=False); body = ck["body"]; diag = float(ck["diag"])
bmean = body["means"].to(dev); brgb = body["rgb"].to(dev).clamp(0, 1)
print(f"[body] {len(bmean)} body gaussians")

# ---- albedo query: nearest body gaussian rgb per strand point (chunked) ----
def nn_rgb(q):
    out = torch.empty(len(q), 3, device=dev)
    for i in range(0, len(q), 16000):
        d = torch.cdist(q[i:i+16000], bmean)
        out[i:i+16000] = brgb[d.argmin(1)]
    return out
pts = P.reshape(-1, 3)
col = nn_rgb(pts).reshape(Ns, 100, 3)
# along-strand smoothing (running mean) + slight darken toward root->tip handled by body already
col = 0.5 * col + 0.25 * torch.roll(col, 1, 1) + 0.25 * torch.roll(col, -1, 1)

# ---- cross-strand colour smoothing (KNN on root positions) ----
if a.smooth_k > 0:
    roots = P[:, 0]
    d = torch.cdist(roots, roots); knn = d.topk(a.smooth_k + 1, largest=False).indices  # [Ns,k+1]
    col = col[knn].mean(1)                                                                # smooth per-strand colour profile across neighbours
print(f"[color] albedo-queried + smoothed (k={a.smooth_k})")

# ---- build strand gaussians: per-segment oriented thin gaussian ----
seg_a = P[:, :-1]; seg_b = P[:, 1:]                          # [Ns,99,3]
mid = 0.5 * (seg_a + seg_b)
tang = seg_b - seg_a; seglen = tang.norm(dim=-1, keepdim=True).clamp(min=1e-6); tang = tang / seglen
# orthonormal frame
up = torch.tensor([0., 0., 1.], device=dev).expand_as(tang)
n1 = F.normalize(torch.cross(tang, up, dim=-1) + 1e-6, dim=-1)
n2 = F.normalize(torch.cross(tang, n1, dim=-1), dim=-1)
R = torch.stack([tang, n1, n2], dim=-1)                     # [Ns,99,3,3] columns = axes
def rot2quat(M):
    m = M.reshape(-1, 3, 3); w = torch.sqrt((1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]).clamp(min=1e-8)) / 2
    x = (m[:, 2, 1] - m[:, 1, 2]) / (4 * w); y = (m[:, 0, 2] - m[:, 2, 0]) / (4 * w); z = (m[:, 1, 0] - m[:, 0, 1]) / (4 * w)
    return F.normalize(torch.stack([w, x, y, z], -1), dim=-1)
quats = rot2quat(R)
r = a.radius_frac * diag
scales = torch.stack([seglen[..., 0] * 0.6, torch.full_like(seglen[..., 0], r), torch.full_like(seglen[..., 0], r)], -1).reshape(-1, 3)
means_s = mid.reshape(-1, 3); rgb_s = col[:, :-1].reshape(-1, 3)
op_s = torch.full((len(means_s),), 0.9, device=dev)
print(f"[gs] {len(means_s)} strand gaussians (r={r:.4f})")

# ---- composite: body skin (under) + strands (over) ----
def cat(*xs): return torch.cat(xs, 0)
means = cat(bmean, means_s)
quats = cat(body["quats"].to(dev), quats)
scales = cat(body["scales"].to(dev), scales)
ops = cat(body["opacities"].to(dev), op_s)
rgb = cat(brgb, rgb_s)

# ---- render multiview vs GT ----
scene = os.path.join(a.root, a.dog, "colmap")
frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
def load_gt(fr, s=4):
    base = os.path.splitext(os.path.basename(fr["name"]))[0]
    im = Image.open(os.path.join(scene, "preprocess", "cache_s4", base + ".jpg")).convert("RGB")
    return np.array(im)
white = torch.ones(3, device=dev); panels = []
for idx in [0, 24, 48, 72]:
    fr = frames[idx]; s = 4
    K = intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev)
    c2w = torch.tensor(fr["c2w"], device=dev).float(); W, H = fr["width"]//s, fr["height"]//s
    with torch.no_grad():
        img, _ = render_gaussians(means, quats, scales, ops, rgb, c2w, K, W, H, bg=white)
    gt = load_gt(fr); rd = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if gt.shape[:2] != rd.shape[:2]:
        gt = np.array(Image.fromarray(gt).resize((rd.shape[1], rd.shape[0])))
    panels.append(np.concatenate([gt, rd], 1))
Image.fromarray(np.concatenate(panels, 0)).save(os.path.join(a.out, f"{a.dog}_neuralfur_colored.png"))
print(f"[render] -> {a.out}/{a.dog}_neuralfur_colored.png")

# ---- turntable ----
if a.turntable > 0:
    fa = np.load(os.path.join(scene, "preprocess", "fur_anchors.npz"))
    grav = torch.tensor(fa["gravity"], device=dev, dtype=torch.float32); up = F.normalize(-grav, dim=0)
    center = means.mean(0); ext = float((means.max(0).values - means.min(0).values).norm())
    ax = torch.tensor([1., 0, 0], device=dev)
    if abs(float((ax*up).sum())) > 0.9: ax = torch.tensor([0., 0, 1.], device=dev)
    e1 = F.normalize(torch.cross(up, ax, dim=0), dim=0); e2 = F.normalize(torch.cross(up, e1, dim=0), dim=0)
    Rv = 0.95*ext; elev = 0.18*ext; Wt = Ht = 640; fov = 40*np.pi/180; fx = Wt/(2*np.tan(fov/2)); Kt = intrinsics(fx, fx, Wt/2, Ht/2, dev)
    frames_g = []
    for az in np.linspace(0, 2*np.pi, a.turntable, endpoint=False):
        cp = center + Rv*(float(np.cos(az))*e1 + float(np.sin(az))*e2) + up*elev
        fwd = F.normalize(center-cp, dim=0); right = F.normalize(torch.cross(fwd, up, dim=0), dim=0); down = torch.cross(fwd, right, dim=0)
        c2w = torch.eye(4, device=dev); c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = cp
        with torch.no_grad():
            img, _ = render_gaussians(means, quats, scales, ops, rgb, c2w, Kt, Wt, Ht, bg=white)
        frames_g.append(Image.fromarray((img.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)))
    gif = os.path.join(a.out, f"{a.dog}_neuralfur_turntable.gif")
    frames_g[0].save(gif, save_all=True, append_images=frames_g[1:], duration=70, loop=0)
    print(f"[turntable] -> {gif}")
