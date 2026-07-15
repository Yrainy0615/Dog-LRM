#!/usr/bin/env python3
"""Fur SWAY animation on NeuralFur strand geometry + cascade colour (same fusion as
build_neuralfur_colored.py). Simple wind logic from animate_v6flow.py: root fixed, displace along a
horizontal wind axis (perp to gravity) by sin(2pi t + 1.5*phase), amplitude growing toward the tip ->
tips swing, body frozen. Traveling wave via per-strand phase. Renders a gif from a real side camera."""
import os, sys, math, argparse, numpy as np, torch, torch.nn.functional as F, trimesh, json
sys.path.insert(0, "/home/yyang/mnt/workspace")
from dog_lrm.render import render_gaussians, intrinsics
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--strands", default="NeuralFur/results/kotori_dense/strands_reconstruction/final/10000_pc.ply")
ap.add_argument("--body_pt", default="exps/fur_v11_kotori_clean/00085-kotori_final.pt")
ap.add_argument("--dog", default="00085-kotori")
ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
ap.add_argument("--out", default="exps/neuralfur_colored")
ap.add_argument("--radius_frac", type=float, default=0.0009)
ap.add_argument("--smooth_k", type=int, default=8)
ap.add_argument("--drop_paws", type=int, default=1)
ap.add_argument("--T", type=int, default=48, help="frames")
ap.add_argument("--amp", type=float, default=0.07, help="tip sway amplitude (frac of diag)")
a = ap.parse_args(); dev = "cuda"; os.makedirs(a.out, exist_ok=True)

# ---- strand geometry (+ paw drop) ----
pc = trimesh.load(a.strands, process=False)
P = torch.tensor(np.asarray(pc.vertices), dtype=torch.float32, device=dev).reshape(-1, 100, 3)
ck = torch.load(a.body_pt, map_location=dev, weights_only=False); body = ck["body"]; diag = float(ck["diag"])
da = np.load(os.path.join(a.root, a.dog, "colmap", "preprocess", "dsmal_anchors.npz"))
if a.drop_paws:
    from dog_lrm.smal_model import SMALModel
    Vp = torch.tensor(da["posed"], dtype=torch.float32, device=dev)
    smal = SMALModel(torch.device(dev), n_subdiv=2); _W = smal.smal.weights
    Wsk = (_W if torch.is_tensor(_W) else torch.tensor(_W)).float().to(dev)
    pawv = (Wsk[:, [10, 14, 20, 24]].sum(1) > 0.5).float()
    nn = torch.cdist(P[:, 0], Vp).argmin(1); P = P[pawv[nn] < 0.5]
Ns = P.shape[0]; print(f"[geom] {Ns} strands")

# ---- colour: albedo query + smoothing ----
bmean = body["means"].to(dev); brgb = body["rgb"].to(dev).clamp(0, 1)
def nn_rgb(q):
    out = torch.empty(len(q), 3, device=dev)
    for i in range(0, len(q), 16000):
        out[i:i+16000] = brgb[torch.cdist(q[i:i+16000], bmean).argmin(1)]
    return out
col = nn_rgb(P.reshape(-1, 3)).reshape(Ns, 100, 3)
col = 0.5 * col + 0.25 * torch.roll(col, 1, 1) + 0.25 * torch.roll(col, -1, 1)
if a.smooth_k > 0:
    knn = torch.cdist(P[:, 0], P[:, 0]).topk(a.smooth_k + 1, largest=False).indices
    col = col[knn].mean(1)
rgb_seg = col[:, :-1].reshape(-1, 3)

# ---- sway setup: wind axis perp to gravity, per-strand traveling-wave phase ----
fa = np.load(os.path.join(a.root, a.dog, "colmap", "preprocess", "fur_anchors.npz"))
g = F.normalize(torch.tensor(fa["gravity"], device=dev, dtype=torch.float32), dim=0)
e = torch.tensor([1., 0, 0], device=dev)
if abs(float((e * g).sum())) > 0.9: e = torch.tensor([0., 0, 1.], device=dev)
wind = F.normalize(torch.cross(g, e, dim=0), dim=0)
roots = P[:, 0]
phase = (roots - roots.mean(0)) @ F.normalize(roots.std(0), dim=0) * 6.0      # traveling wave across body
phase = phase + torch.rand(Ns, device=dev) * 0.5
tipw = (torch.linspace(0, 1, 100, device=dev) ** 1.5)[None, :, None]            # 0 at root -> 1 at tip

# ---- body gaussians (static) ----
bq = body["quats"].to(dev); bs = body["scales"].to(dev); bo = body["opacities"].to(dev)
r = a.radius_frac * diag

def make_strand_gs(Pd):
    seg_a = Pd[:, :-1]; seg_b = Pd[:, 1:]; mid = 0.5 * (seg_a + seg_b)
    t = seg_b - seg_a; L = t.norm(dim=-1, keepdim=True).clamp(min=1e-6); t = t / L
    up = torch.tensor([0., 0., 1.], device=dev).expand_as(t)
    n1 = F.normalize(torch.cross(t, up, dim=-1) + 1e-6, dim=-1); n2 = F.normalize(torch.cross(t, n1, dim=-1), dim=-1)
    M = torch.stack([t, n1, n2], -1).reshape(-1, 3, 3)
    w = torch.sqrt((1 + M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]).clamp(min=1e-8)) / 2
    x = (M[:, 2, 1] - M[:, 1, 2]) / (4 * w); y = (M[:, 0, 2] - M[:, 2, 0]) / (4 * w); z = (M[:, 1, 0] - M[:, 0, 1]) / (4 * w)
    q = F.normalize(torch.stack([w, x, y, z], -1), dim=-1)
    sc = torch.stack([L[..., 0] * 0.6, torch.full_like(L[..., 0], r), torch.full_like(L[..., 0], r)], -1).reshape(-1, 3)
    return mid.reshape(-1, 3), q, sc

# ---- pick a real side camera (largest projected bbox area) ----
frames = json.load(open(os.path.join(a.root, a.dog, "colmap", "preprocess", "cameras.json")))["frames"]
allpts = torch.cat([bmean, P.reshape(-1, 3)], 0)
best = None
for fr in frames:
    s = 4; K = intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev)
    c2w = torch.tensor(fr["c2w"], device=dev).float(); W, H = fr["width"]//s, fr["height"]//s
    w2c = torch.inverse(c2w); cam = (w2c[:3, :3] @ allpts.T + w2c[:3, 3:]).T
    uv = (K[:3, :3] @ cam.T).T; z = uv[:, 2].clamp(min=1e-4); u = uv[:, 0]/z; v = uv[:, 1]/z
    inb = (u > 0) & (u < W) & (v > 0) & (v < H) & (z > 0); area = float(inb.float().mean())
    if best is None or area > best[0]: best = (area, fr)
fr = best[1]; s = 4
K = intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev)
c2w = torch.tensor(fr["c2w"], device=dev).float(); W, H = fr["width"]//s, fr["height"]//s
white = torch.ones(3, device=dev)
print(f"[view] camera {fr['name']} ({W}x{H}), coverage {best[0]*100:.0f}%")
# crop box (from static body projection + margin for sway) -> zoom on dog
w2c = torch.inverse(c2w); camb = (w2c[:3, :3] @ bmean.T + w2c[:3, 3:]).T
uvb = (K[:3, :3] @ camb.T).T; zb = uvb[:, 2].clamp(min=1e-4); ub = uvb[:, 0]/zb; vb = uvb[:, 1]/zb
mx = int(0.10 * W); my = int(0.10 * H)
x0 = max(0, int(ub.min()) - mx); x1 = min(W, int(ub.max()) + mx)
y0 = max(0, int(vb.min()) - my); y1 = min(H, int(vb.max()) + my)

# ---- render frames ----
imgs = []
for f in range(a.T):
    t = f / a.T
    disp = (a.amp * diag) * torch.sin(2 * math.pi * t + 1.5 * phase)[:, None, None] * wind[None, None] * tipw
    Pd = P + disp
    ms, qs, ss = make_strand_gs(Pd)
    means = torch.cat([bmean, ms]); quats = torch.cat([bq, qs]); scales = torch.cat([bs, ss])
    ops = torch.cat([bo, torch.full((len(ms),), 0.9, device=dev)]); rgb = torch.cat([brgb, rgb_seg])
    with torch.no_grad():
        im, _ = render_gaussians(means, quats, scales, ops, rgb, c2w, K, W, H, bg=white)
    arr = (im.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[y0:y1, x0:x1]
    imgs.append(Image.fromarray(arr))
gif = os.path.join(a.out, f"{a.dog}_fur_sway.gif")
imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=70, loop=0)
print(f"[anim] {a.T} frames -> {gif}")
