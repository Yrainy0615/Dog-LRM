#!/usr/bin/env python3
"""Sample the trained strand diffusion for held-out images and compare GENERATED vs GT strand
geometry (matplotlib: template mesh + strand lines). Validates the generator visually."""
import os, sys, glob, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image
from train_strand_predictor import vnormals, tbn_frames, project, fourier
from train_strand_diffusion import Denoiser
dev = "cuda"; K = 6; zdim = 18; res = 224

tz = np.load("synth_fur/blender_input.npz")
V = torch.from_numpy(tz["verts"].astype(np.float32)).to(dev); Fc = torch.from_numpy(tz["faces"].astype(np.int64)).to(dev)
Nrm = vnormals(V, Fc); TBN = tbn_frames(V, Nrm); diag = float((V.max(0).values-V.min(0).values).norm())
ck = torch.load("exps/strand_diff/diff.pt", map_location=dev); ridx = ck["ridx"].to(dev)
R = V[ridx]; Rn = Nrm[ridx]; Rtbn = TBN[ridx]; posfix = R/diag
zmean = ck["zmean"].to(dev); zstd = ck["zstd"].to(dev); betas = ck["betas"].to(dev); ac = torch.cumprod(1-betas, 0); T = len(betas)
net = Denoiser(zdim=zdim).to(dev); net.load_state_dict(ck["net"]); net.eval()

from transformers import AutoModel
os.environ.setdefault("HF_HUB_OFFLINE", "1")
dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); nsd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
ph = res//14
cond_pos = torch.cat([fourier(posfix), Rn], -1)


def dfeat(img):
    x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float()
    x = (F.interpolate(x, (res, res), mode="bilinear", align_corners=False)-nm)/nsd
    with torch.no_grad(): f = dino(pixel_values=x).last_hidden_state[:, 1:]
    return f.transpose(1, 2).reshape(1, -1, ph, ph)


@torch.no_grad()
def sample(cond, steps=50):
    z = torch.randn(cond.shape[0], zdim, device=dev)
    ts = torch.linspace(T-1, 0, steps, device=dev).long()
    for i in range(steps):
        t = ts[i].expand(cond.shape[0]); a = ac[t][:, None]
        eps = net(z, t, cond); z0 = (z-(1-a).sqrt()*eps)/a.sqrt()
        z = (ac[ts[i+1]].sqrt()*z0 + (1-ac[ts[i+1]]).sqrt()*eps) if i < steps-1 else z0
    return z*zstd + zmean


def world_strands(z_local):                                  # [nroot,K,3] TBN-> world
    return R[:, None] + torch.einsum("rij,rkj->rki", Rtbn, z_local.view(-1, K, 3)*diag)


def plot(ax, strands, color):
    ctr = V.mean(0).cpu().numpy(); rng = float((V.max(0).values-V.min(0).values).max())*0.55
    tris = V.cpu().numpy()[Fc.cpu().numpy()]
    fn = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]); fn /= (np.linalg.norm(fn, 1, keepdims=True)+1e-9)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=(0.75, 0.75, 0.75), edgecolors="none"))
    segs = strands.detach().cpu().numpy()[::4]
    ax.add_collection3d(Line3DCollection(list(segs), colors=color, linewidths=0.4, alpha=0.8))
    ax.set_axis_off(); ax.view_init(elev=12, azim=90)
    ax.set_xlim(ctr[0]-rng, ctr[0]+rng); ax.set_ylim(ctr[1]-rng, ctr[1]+rng); ax.set_zlim(ctr[2]-rng, ctr[2]+rng); ax.set_box_aspect((1, 1, 1))


rows = []
for name in ["short_4", "long_4", "curly_4"]:
    z = np.load(f"synth_fur/dataset/{name}.npz", allow_pickle=True)
    strands = torch.from_numpy(z["strands"]).to(dev); roots = torch.from_numpy(z["roots"]).to(dev)
    nn_ = torch.cdist(R, roots).argmin(1)
    gt_local = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), strands[nn_]-R[:, None])/diag
    Ks = torch.from_numpy(z["Ks"]).to(dev); c2ws = torch.from_numpy(z["c2ws"]).to(dev)
    img = np.array(Image.open(f"synth_fur/dataset/{z['imgs'][0]}").convert("RGB"))/255.
    H, W = img.shape[:2]; uv, _ = project(R, Ks[0], c2ws[0]); grid = torch.stack([uv[:, 0]/W*2-1, uv[:, 1]/H*2-1], -1)[None, :, None, :]
    cond = torch.cat([F.grid_sample(dfeat(img), grid, align_corners=False)[0, :, :, 0].T, cond_pos], -1)
    gen = world_strands(sample(cond)); gt = world_strands(gt_local.reshape(-1, zdim))
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(131, projection="3d"); plot(ax1, gt, (0.1, 0.4, 0.9)); ax1.set_title(f"{name} GT", fontsize=8)
    ax2 = fig.add_subplot(132, projection="3d"); plot(ax2, gen, (0.9, 0.3, 0.1)); ax2.set_title("generated", fontsize=8)
    ax3 = fig.add_subplot(133); ax3.imshow(img); ax3.set_axis_off(); ax3.set_title("input img", fontsize=8)
    plt.tight_layout(); fp = f"exps/strand_diff/_sample_{name}.png"; plt.savefig(fp, dpi=110); plt.close()
    rows.append(fp); print("saved", fp, flush=True)
# stack
ims = [np.array(Image.open(r)) for r in rows]; w = min(i.shape[1] for i in ims)
Image.fromarray(np.concatenate([i[:, :w] for i in ims], 0)).save("exps/strand_diff/_samples.png")
print("saved exps/strand_diff/_samples.png")
