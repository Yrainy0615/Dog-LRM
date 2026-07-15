#!/usr/bin/env python3
"""More views of the NeuralFur-geometry + our-colour fusion: an N-view GT|render grid + fur close-ups."""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F, trimesh
sys.path.insert(0, "/home/yyang/mnt/workspace")
from dog_lrm.render import render_gaussians, intrinsics
from dog_lrm.smal_model import SMALModel
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dog", default="00085-kotori")
ap.add_argument("--strands", default="NeuralFur/results/kotori_dense/strands_reconstruction/final/20000_pc.ply")
ap.add_argument("--body_pt", default="exps/fur_v11_kotori_clean/00085-kotori_final.pt")
ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
ap.add_argument("--out", default="exps/neuralfur_final"); ap.add_argument("--nviews", type=int, default=12)
ap.add_argument("--radius_frac", type=float, default=0.0009); ap.add_argument("--smooth_k", type=int, default=8)
a = ap.parse_args(); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
scene = os.path.join(a.root, a.dog, "colmap")

# ---- fusion (same as build_neuralfur_colored) ----
P = torch.tensor(np.asarray(trimesh.load(a.strands, process=False).vertices), dtype=torch.float32, device=dev).reshape(-1, 100, 3)
ck = torch.load(a.body_pt, map_location=dev, weights_only=False); body = ck["body"]; diag = float(ck["diag"])
da = np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))
Vp = torch.tensor(da["posed"], dtype=torch.float32, device=dev)
smal = SMALModel(torch.device(dev), n_subdiv=2); _W = smal.smal.weights
Wsk = (_W if torch.is_tensor(_W) else torch.tensor(_W)).float().to(dev)
pawv = (Wsk[:, [10, 14, 20, 24]].sum(1) > 0.5).float()
P = P[pawv[torch.cdist(P[:, 0], Vp).argmin(1)] < 0.5]; Ns = P.shape[0]
bmean = body["means"].to(dev); brgb = body["rgb"].to(dev).clamp(0, 1)
def nn_rgb(q):
    out = torch.empty(len(q), 3, device=dev)
    for i in range(0, len(q), 16000): out[i:i+16000] = brgb[torch.cdist(q[i:i+16000], bmean).argmin(1)]
    return out
col = nn_rgb(P.reshape(-1, 3)).reshape(Ns, 100, 3)
col = 0.5*col + 0.25*torch.roll(col, 1, 1) + 0.25*torch.roll(col, -1, 1)
if a.smooth_k > 0:
    col = col[torch.cdist(P[:, 0], P[:, 0]).topk(a.smooth_k+1, largest=False).indices].mean(1)
sa, sb = P[:, :-1], P[:, 1:]; mid = 0.5*(sa+sb); t = sb-sa; L = t.norm(dim=-1, keepdim=True).clamp(min=1e-6); t = t/L
up = torch.tensor([0., 0, 1.], device=dev).expand_as(t)
n1 = F.normalize(torch.cross(t, up, dim=-1)+1e-6, dim=-1); n2 = F.normalize(torch.cross(t, n1, dim=-1), dim=-1)
M = torch.stack([t, n1, n2], -1).reshape(-1, 3, 3)
w = torch.sqrt((1+M[:, 0, 0]+M[:, 1, 1]+M[:, 2, 2]).clamp(min=1e-8))/2
q = F.normalize(torch.stack([w, (M[:, 2, 1]-M[:, 1, 2])/(4*w), (M[:, 0, 2]-M[:, 2, 0])/(4*w), (M[:, 1, 0]-M[:, 0, 1])/(4*w)], -1), dim=-1)
r = a.radius_frac*diag
sc = torch.stack([L[..., 0]*0.6, torch.full_like(L[..., 0], r), torch.full_like(L[..., 0], r)], -1).reshape(-1, 3)
means = torch.cat([bmean, mid.reshape(-1, 3)]); quats = torch.cat([body["quats"].to(dev), q])
scales = torch.cat([body["scales"].to(dev), sc]); ops = torch.cat([body["opacities"].to(dev), torch.full((len(q),), 0.9, device=dev)])
rgb = torch.cat([brgb, col[:, :-1].reshape(-1, 3)])
print(f"[fusion] {Ns} strands, {len(means)} gaussians")

# ---- N-view grid ----
frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
white = torch.ones(3, device=dev)
idxs = list(np.linspace(0, len(frames)-1, a.nviews).astype(int))
def render(fr, s=4):
    K = intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev)
    c2w = torch.tensor(fr["c2w"], device=dev).float(); W, H = fr["width"]//s, fr["height"]//s
    with torch.no_grad(): img, _ = render_gaussians(means, quats, scales, ops, rgb, c2w, K, W, H, bg=white)
    base = os.path.splitext(os.path.basename(fr["name"]))[0]
    gt = np.array(Image.open(os.path.join(scene, "preprocess", "cache_s4", base+".jpg")).convert("RGB"))
    rd = (img.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
    if gt.shape[:2] != rd.shape[:2]: gt = np.array(Image.fromarray(gt).resize((rd.shape[1], rd.shape[0])))
    return np.concatenate([gt, rd], 1)
cells = [render(frames[i]) for i in idxs]
th = min(c.shape[0] for c in cells); tw = min(c.shape[1] for c in cells)
cells = [np.array(Image.fromarray(c).resize((tw, th))) for c in cells]
cols = 3; rows = [np.concatenate(cells[i:i+cols], 1) for i in range(0, len(cells), cols)]
mw = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, mw-r.shape[1]), (0, 0)), constant_values=255) for r in rows]
Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out, f"{a.dog}_grid{a.nviews}.png"))
print(f"[grid] {a.nviews} views -> {a.out}/{a.dog}_grid{a.nviews}.png")

# ---- fur close-ups (render high-res, crop 3 patches) ----
fr = frames[idxs[len(idxs)//3]]
K = intrinsics(fr["fx"]/2, fr["fy"]/2, fr["cx"]/2, fr["cy"]/2, dev)
c2w = torch.tensor(fr["c2w"], device=dev).float(); W, H = fr["width"]//2, fr["height"]//2
with torch.no_grad(): img, _ = render_gaussians(means, quats, scales, ops, rgb, c2w, K, W, H, bg=white)
rd = (img.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
m = (rd < 250).any(2); ys, xs = np.where(m); cy, cx = int(ys.mean()), int(xs.mean()); hh = (ys.max()-ys.min())
crops = []
for (dy, dx) in [(-0.2, 0), (0.05, -0.12), (0.05, 0.12)]:
    yy = int(cy+dy*hh); xx = int(cx+dx*hh); s2 = hh//5
    crops.append(np.array(Image.fromarray(rd[max(0, yy-s2):yy+s2, max(0, xx-s2):xx+s2]).resize((300, 300))))
Image.fromarray(np.concatenate(crops, 1)).save(os.path.join(a.out, f"{a.dog}_closeup.png"))
print(f"[closeup] -> {a.out}/{a.dog}_closeup.png")
