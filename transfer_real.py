#!/usr/bin/env python3
"""#1 REAL-dog transfer test for the synthetic-trained strand diffusion.
Map the fixed template roots onto a real dog's posed D-SMAL mesh (colmap frame), project to the
real (cropped) photo, sample frozen-DINO features, run the generator, decode strands on the real
mesh. The frozen backbone is meant to bridge synth->real (DiffLocks premise). Visualize
[real image | generated strands on real mesh]."""
import os, sys, json, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image
from dog_lrm.smal_model import build_subdiv
from train_strand_predictor import vnormals, tbn_frames, fourier
from train_strand_diffusion import Denoiser
from train_dog_lrm_ddp import _load_rgb_mask
dev = "cuda"; K = 6; zdim = 18; res = 224; ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"

# template (synthetic training frame) for ridx + canonical cond pos/normal
bt = np.load("synth_fur/blender_input.npz")
Vt = torch.from_numpy(bt["verts"].astype(np.float32)).to(dev); Ft = torch.from_numpy(bt["faces"].astype(np.int64)).to(dev)
Nt = vnormals(Vt, Ft); diag_t = float((Vt.max(0).values-Vt.min(0).values).norm())
ck = torch.load("exps/strand_diff/diff.pt", map_location=dev); ridx = ck["ridx"].to(dev)
cond_pos = torch.cat([fourier(Vt[ridx]/diag_t), Nt[ridx]], -1)
zmean = ck["zmean"].to(dev); zstd = ck["zstd"].to(dev); betas = ck["betas"].to(dev); ac = torch.cumprod(1-betas, 0); T = len(betas)
net = Denoiser(zdim=zdim).to(dev); net.load_state_dict(ck["net"]); net.eval()

from transformers import AutoModel
os.environ.setdefault("HF_HUB_OFFLINE", "1")
dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); nsd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1); ph = res//14


def dfeat(img):
    x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float()
    x = (F.interpolate(x, (res, res), mode="bilinear", align_corners=False)-nm)/nsd
    with torch.no_grad(): f = dino(pixel_values=x).last_hidden_state[:, 1:]
    return f.transpose(1, 2).reshape(1, -1, ph, ph)


@torch.no_grad()
def sample(cond, steps=50):
    z = torch.randn(cond.shape[0], zdim, device=dev); ts = torch.linspace(T-1, 0, steps, device=dev).long()
    for i in range(steps):
        t = ts[i].expand(cond.shape[0]); a = ac[t][:, None]; eps = net(z, t, cond); z0 = (z-(1-a).sqrt()*eps)/a.sqrt()
        z = (ac[ts[i+1]].sqrt()*z0 + (1-ac[ts[i+1]]).sqrt()*eps) if i < steps-1 else z0
    return z*zstd + zmean


def real_dog(dog):
    sc = f"{ROOT}/{dog}/colmap"; da = np.load(f"{sc}/preprocess/dsmal_anchors.npz")
    posed = torch.from_numpy(da["posed"].astype(np.float32)).to(dev); faces = torch.from_numpy(da["faces"].astype(np.int64)).to(dev)
    M = build_subdiv(faces, 1, dev)
    P = torch.sparse.mm(M, posed)                                   # [15550,3] posed subdiv1 (colmap frame)
    Nrm = vnormals(P, Ft); TBN = tbn_frames(P, Nrm); diag = float((P.max(0).values-P.min(0).values).norm())
    Rr = P[ridx]; Rtbn = TBN[ridx]
    # widest (side) view, cropped to mask
    frames = json.load(open(f"{sc}/preprocess/cameras.json"))["frames"]; s = 4
    best = None
    for fr in frames:
        _, mk, _, _ = _load_rgb_mask(sc, fr, s); yy, xx = np.where(mk[:, :, 0] > 0.5)
        if len(xx) < 50: continue
        ar = (xx.max()-xx.min())/max(yy.max()-yy.min(), 1)
        if best is None or ar > best[0]: best = (ar, fr, (yy.min(), yy.max(), xx.min(), xx.max()))
    fr = best[1]; y0, y1, x0, x1 = best[2]; rgb, mask, W, H = _load_rgb_mask(sc, fr, s)
    K_ = torch.tensor([[fr["fx"]/s, 0, fr["cx"]/s], [0, fr["fy"]/s, fr["cy"]/s], [0, 0, 1]], device=dev)
    c2w = torch.tensor(fr["c2w"], device=dev).float()
    # project roots (colmap/opencv: viewmat=inv(c2w))
    w2c = torch.inverse(c2w); cam = (w2c[:3, :3] @ Rr.T + w2c[:3, 3:4]).T; z = cam[:, 2].clamp(min=1e-4)
    uvh = (K_ @ (cam/z[:, None]).T).T; uv = uvh[:, :2]
    pad = int(0.06*max(y1-y0, x1-x0)); cy0, cx0 = max(y0-pad, 0), max(x0-pad, 0)
    crop = rgb[cy0:y1+pad, cx0:x1+pad]; ch, cw = crop.shape[:2]
    grid = torch.stack([(uv[:, 0]-cx0)/cw*2-1, (uv[:, 1]-cy0)/ch*2-1], -1)[None, :, None, :]
    cond = torch.cat([F.grid_sample(dfeat(crop), grid, align_corners=False)[0, :, :, 0].T, cond_pos], -1)
    zs = sample(cond)
    strands = Rr[:, None] + torch.einsum("rij,rkj->rki", Rtbn, zs.view(-1, K, 3)*diag)
    return crop, P, strands


fig = plt.figure(figsize=(9, 6))
for r, dog in enumerate(["00031-itsuki", "00003-nara"]):
    crop, P, strands = real_dog(dog)
    ax = fig.add_subplot(2, 2, r*2+1); ax.imshow(crop); ax.set_axis_off(); ax.set_title(f"{dog} real", fontsize=8)
    ax2 = fig.add_subplot(2, 2, r*2+2, projection="3d")
    Pn = P.cpu().numpy(); ctr = Pn.mean(0); rng = (Pn.max(0)-Pn.min(0)).max()*0.55
    tris = Pn[Ft.cpu().numpy()]
    ax2.add_collection3d(Poly3DCollection(tris, facecolors=(0.78, 0.78, 0.78), edgecolors="none"))
    segs = strands.detach().cpu().numpy()[::4]
    ax2.add_collection3d(Line3DCollection(list(segs), colors=(0.9, 0.3, 0.1), linewidths=0.4, alpha=0.8))
    ax2.set_axis_off(); ax2.view_init(elev=12, azim=-90)
    ax2.set_xlim(ctr[0]-rng, ctr[0]+rng); ax2.set_ylim(ctr[1]-rng, ctr[1]+rng); ax2.set_zlim(ctr[2]-rng, ctr[2]+rng); ax2.set_box_aspect((1, 1, 1))
    ax2.set_title("generated strands", fontsize=8)
    print(f"[transfer] {dog}: strands {tuple(strands.shape)} mean_len {float((strands[:,-1]-strands[:,0]).norm(dim=-1).mean()):.3f}", flush=True)
plt.tight_layout(); plt.savefig("exps/strand_diff/_transfer_real.png", dpi=120); print("saved _transfer_real.png")
