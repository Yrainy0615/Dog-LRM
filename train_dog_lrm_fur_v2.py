#!/usr/bin/env python3
"""P3/P4: feedforward strand-fur Dog-LRM on the 69 D-SMAL dogs (FUR_V2_PLAN).

Backbone = DogLRMDecomp (region tokens); fur = DogLRMFurV2 head -> NeuralFur-style
strands from cached fur_anchors.npz; per-dog curl-class embedding. Supervision at
1/4 res (cache_s4). Losses: mask-area RGB L1 + mask + LPIPS + penetration
(strand below root plane). Held-out views (every 12th) excluded from training.

  CUDA_VISIBLE_DEVICES=1,4,5,6 torchrun --nproc_per_node=4 train_dog_lrm_fur_v2.py \
      --root received_data_from_Pinstudio_20260424/unzipped/0423 --iters 8000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model_fur import DogLRMFurV2, DogLRMFurV7, DogLRMFurV8, DogLRMFurV9, DogLRMFurV10, load_fur_ckpt
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.motion import look_at
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask, collate, is_main
from train_dog_lrm_decomp import _label_grid


def _sep_gauss(x, sigma):
    """Separable Gaussian blur of x [1,C,H,W] (GPU, no cv2)."""
    k = int(2 * round(3 * sigma) + 1)
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(C, 1, 1, k), padding=(0, k // 2), groups=C)
    x = F.conv2d(x, g.view(1, 1, k, 1).expand(C, 1, k, 1), padding=(k // 2, 0), groups=C)
    return x


def photo_orientation(gray, sigma):
    """Structure-tensor fur-flow field of a GT image (the supervision TARGET; no grad).
    gray [H,W] in [0,1]. Returns cos2t,sin2t (double-angle of the along-strand flow,
    naturally mod-pi) and coh [H,W] (anisotropy = confidence). Coarse `sigma` (~16)
    captures macro fur-flow; fine sigma is texture noise on real dog fur (see v7 diag)."""
    g = _sep_gauss(gray[None, None], 1.0)                       # pre-smooth for gradients
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype,
                      device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, sx, padding=1); gy = F.conv2d(g, sx.transpose(2, 3), padding=1)
    Jxx = _sep_gauss(gx * gx, sigma)[0, 0]
    Jyy = _sep_gauss(gy * gy, sigma)[0, 0]
    Jxy = _sep_gauss(gx * gy, sigma)[0, 0]
    grad_ori = 0.5 * torch.atan2(2 * Jxy, Jxx - Jyy)            # dominant-gradient angle
    theta = grad_ori + np.pi / 2                                # flow runs perpendicular
    tmp = torch.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    coh = tmp / (Jxx + Jyy + 1e-8)                              # (lam1-lam2)/(lam1+lam2)
    return torch.cos(2 * theta), torch.sin(2 * theta), coh


def orient_loss(f0, gt, m, c2w, K, W, H, dev, sigma, coh_thr):
    """Render the fur strand-tangent G-buffer at this view and align its image-plane
    orientation with the GT photo's coarse fur-flow (undirected, mod-pi). Returns a
    scalar in [0,2], confidence-weighted by fur coverage x photo-coherence x in-plane
    projection length (edge-on strands carry no reliable 2D orientation)."""
    col = (f0["tangent"] + 1.0) * 0.5                          # tangent-as-color, linear-safe
    rgb, alpha = render_gaussians(f0["means"], f0["quats"], f0["scales"],
                                  f0["opacities"], col, c2w, K, W, H,
                                  bg=torch.zeros(3, device=dev), sh_degree=None)
    A = alpha[..., 0].clamp_min(1e-3)                          # [H,W]
    avg_t = (rgb / A[..., None]) * 2.0 - 1.0                   # weighted-avg world tangent
    cam_t = avg_t @ torch.linalg.inv(c2w)[:3, :3].T           # rotate world->camera (OpenCV)
    vx, vy = cam_t[..., 0], cam_t[..., 1]
    vn = torch.sqrt(vx * vx + vy * vy + 1e-8)                  # eps INSIDE sqrt (grad-safe at 0)
    ux, uy = vx / vn, vy / vn
    r_cos2, r_sin2 = ux * ux - uy * uy, 2 * ux * uy            # render double-angle
    with torch.no_grad():
        gray = (0.299 * gt[..., 0] + 0.587 * gt[..., 1] + 0.114 * gt[..., 2])
        t_cos2, t_sin2, coh = photo_orientation(gray, sigma)
        w = A.detach() * m[..., 0] * coh * (coh > coh_thr).float() * vn.detach().clamp(0, 1)
    per_px = 1.0 - (r_cos2 * t_cos2 + r_sin2 * t_sin2)         # 1 - cos(2*dTheta), [0,2]
    return (per_px * w).sum() / w.sum().clamp_min(1.0)


class PatchD(nn.Module):
    """PatchGAN discriminator on fur crops -- real (GT) vs rendered. Provides an
    adversarial texture loss that flows through the differentiable gaussian renderer
    into the gaussian params, pushing them to produce realistic high-freq fur (stable
    alternative to SDS, no diffusion drift)."""
    def __init__(self, ch=64):
        super().__init__()
        nrm = lambda o: nn.GroupNorm(8, o)
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nrm(ch * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nrm(ch * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 4, 1, 4, 1, 1))

    def forward(self, x):                                # x in [0,1] -> [-1,1]
        return self.net(x * 2 - 1)


def fur_patches(render_hwc, gt_hwc, mask_hw, sz, n):
    """n square crops centered on fur pixels; same box for render(fake)/GT(real),
    masked to the dog so the discriminator only sees fur. Returns fake[n,3,sz,sz]
    (with grad through render) and real[n,3,sz,sz]."""
    H, W = render_hwc.shape[:2]
    m = mask_hw[:, :, 0] > 0.5
    ys, xs = torch.where(m)
    if len(ys) < 50:
        return None, None
    rm = render_hwc * mask_hw                            # fur on black
    gm = gt_hwc * mask_hw
    fakes, reals = [], []
    for _ in range(n):
        i = int(torch.randint(len(ys), (1,)))
        cy, cx = int(ys[i]), int(xs[i])
        y0 = min(max(cy - sz // 2, 0), max(H - sz, 0))
        x0 = min(max(cx - sz // 2, 0), max(W - sz, 0))
        f = rm[y0:y0 + sz, x0:x0 + sz].permute(2, 0, 1)[None]
        r = gm[y0:y0 + sz, x0:x0 + sz].permute(2, 0, 1)[None]
        if f.shape[-2:] != (sz, sz):
            f = F.interpolate(f, (sz, sz), mode="bilinear", align_corners=False)
            r = F.interpolate(r, (sz, sz), mode="bilinear", align_corners=False)
        fakes.append(f); reals.append(r)
    return torch.cat(fakes), torch.cat(reals)

torch.multiprocessing.set_sharing_strategy("file_system")

ANC_KEYS = ("roots", "t", "b", "n", "L", "w_face", "gravity")


@torch.no_grad()
def nn_index(a, b, chunk=4096):
    """For each row of a, index of nearest row in b (chunked)."""
    out = []
    for i in range(0, a.shape[0], chunk):
        out.append(torch.cdist(a[i:i + chunk], b).argmin(1))
    return torch.cat(out)


def _face_box(scene, fr, s, W, H, min_px=24):
    """Face bbox in 1/s-res pixels from the cached s8 region mask (R = w_face prob),
    padded 15%; None when the face is absent/too small in this view."""
    from PIL import Image
    p = os.path.join(scene, "preprocess", "region_masks_s8",
                     os.path.splitext(fr["name"])[0] + ".png")
    if not os.path.exists(p):
        return None
    r = np.asarray(Image.open(p))[:, :, 0]
    ys, xs = np.nonzero(r > 76)
    if len(ys) < min_px * min_px:
        return None
    sc = 8.0 / s
    x0, x1 = xs.min() * sc, (xs.max() + 1) * sc
    y0, y1 = ys.min() * sc, (ys.max() + 1) * sc
    px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    x0, y0 = max(int(x0 - px), 0), max(int(y0 - py), 0)
    x1, y1 = min(int(x1 + px), W), min(int(y1 + py), H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def face_crop_518(scene, ref, ref_rgb_s4):
    """High-res face crop from the s4 ref RGB [3,H,W], resized to 518x518 for DINO re-encode.
    None when the face is absent/too small in the ref."""
    _, _, H, W = 0, 0, ref_rgb_s4.shape[1], ref_rgb_s4.shape[2]
    fb = _face_box(scene, ref, 4, W, H)
    if fb is None:
        return None
    x0, y0, x1, y1 = fb
    crop = ref_rgb_s4[:, y0:y1, x0:x1][None]
    return F.interpolate(crop, (518, 518), mode="bilinear", align_corners=False)[0]


def list_scenes(root):
    out = []
    for c in sorted(glob.glob(os.path.join(root, "*", "colmap"))):
        pre = os.path.join(c, "preprocess")
        if all(os.path.exists(os.path.join(pre, f)) for f in
               ("dsmal_anchors.npz", "fur_anchors.npz", "cameras.json")) \
                and os.path.isdir(os.path.join(pre, "region_masks_s8")):
            out.append(c)
    return out


class FurScenes(Dataset):
    """One item == one scene: ref view (input+label) + K supervision views (s4)
    + fur anchors. Held-out views (every 12th of the sorted list) never sampled."""

    def __init__(self, scene_dirs, scale_div, k_sup, n_root=26000, all_views=False,
                 anchors="fur_anchors.npz", v7=False):
        self.dirs, self.s, self.k, self.all_views = scene_dirs, scale_div, k_sup, all_views
        self.anchors, self.v7 = anchors, v7
        self.frames, self.train_ids, self.anc, self.canon, self.ref_ids = [], [], [], [], []
        for d in scene_dirs:
            fr = json.load(open(os.path.join(d, "preprocess", "cameras.json")))["frames"]
            fr = sorted(fr, key=lambda f: f["name"])
            self.frames.append(fr)
            held = set(range(1, len(fr), 12))
            tids = [i for i in range(len(fr)) if i not in held]
            self.train_ids.append(tids)
            # ref views restricted to face-visible ones (muzzle z-visibility from the
            # D-SMAL fit, cached by cache_fur_anchors) -- a random ref often shows no face
            ref_ids = tids
            fsp = os.path.join(d, "preprocess", "face_scores.json")
            if os.path.exists(fsp):
                fsc = json.load(open(fsp))
                sc = np.array([fsc.get(fr[i]["name"], 0.0) for i in tids])
                if sc.max() > 0:
                    keep = [t for t, s_ in zip(tids, sc) if s_ >= 0.3 * sc.max()]
                    ref_ids = keep or tids
            self.ref_ids.append(ref_ids)
            a = np.load(os.path.join(d, "preprocess", self.anchors))
            anc = {k: torch.from_numpy(a[k]) for k in ANC_KEYS}
            anc["diag"] = torch.tensor(float(a["diag"]))
            anc["curl_id"] = torch.tensor(int(a["curl_id"]))
            # r5 budgeted roots: cached shuffled, so a prefix is a uniform subsample
            for k in ("root_face", "root_bary", "root_phase", "root_tone"):
                anc[k] = torch.from_numpy(a[k][:n_root])
            if self.v7:        # v7 region recipe: face/nose/ear short fur lying FLAT
                wt = torch.from_numpy(a["w_tail"]) if "w_tail" in a else torch.zeros_like(anc["L"])
                we = torch.from_numpy(a["w_ear"]) if "w_ear" in a else torch.zeros_like(anc["L"])
                wh = torch.from_numpy(a["w_head"]) if "w_head" in a else torch.zeros_like(anc["L"])
                flat = torch.maximum(we, wh).clamp(0, 1)              # ears+face lie flat
                anc["tmix"] = (0.75 + 0.2 * wt - 0.5 * flat).clamp(0, 0.97)  # low tmix=no spray
                anc["droop"] = (0.6 + 0.3 * wt + 0.3 * we).clamp(0, 1)
                anc["L"] = anc["L"] * (1 - 0.3 * wt) * (1 - 0.6 * we) * (1 - 0.75 * wh)
            else:
                if "w_tail" in a:  # tail fix: flow along the bone + hang, shorter
                    wt = torch.from_numpy(a["w_tail"])
                    anc["tmix"] = (0.75 + 0.2 * wt).clamp(0, 0.97)
                    anc["droop"] = 0.6 + 0.3 * wt
                    anc["L"] = anc["L"] * (1 - 0.3 * wt)
                if "w_ear" in a:   # ear fix: ears borrow the head-fluff envelope in the fit
                    we = torch.from_numpy(a["w_ear"])     # pose and flare it when posed ->
                    base_t = anc.get("tmix", torch.full_like(we, 0.75))   # short + flow along
                    base_d = anc.get("droop", torch.full_like(we, 0.6))   # the ear + hang
                    anc["tmix"] = (base_t + 0.2 * we).clamp(0, 0.97)
                    anc["droop"] = (base_d + 0.3 * we).clamp(0, 1)
                    anc["L"] = anc["L"] * (1 - 0.5 * we)
            if "L_geo" in a:   # measured fur envelope: strands stay inside the GT
                anc["L_max"] = torch.from_numpy(a["L_geo"])  # mask in any pose
            self.anc.append(anc)
            da = np.load(os.path.join(d, "preprocess", "dsmal_anchors.npz"))
            self.canon.append(torch.from_numpy(da["canon"]))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d, frames, ids = self.dirs[i], self.frames[i], self.train_ids[i]
        ref = frames[int(np.random.choice(self.ref_ids[i]))]
        rgb_r, mask_r, _, _ = _load_rgb_mask(d, ref, 8)
        label = _label_grid(d, ref, mask_r)
        ref_in = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None],
                               size=(518, 518), mode="bilinear", align_corners=False)[0]
        rgb4, _, _, _ = _load_rgb_mask(d, ref, 4)                  # albedo sampling source
        ref_rgb = torch.from_numpy(rgb4).permute(2, 0, 1)
        fcrop = face_crop_518(d, ref, ref_rgb)                     # high-res face crop or None
        ref_K = intrinsics(ref["fx"] / 4, ref["fy"] / 4, ref["cx"] / 4, ref["cy"] / 4, "cpu")
        ref_c2w = torch.tensor(ref["c2w"]).float()
        sup = []
        sel = ids if self.all_views else np.random.choice(ids, self.k, replace=False)
        for j in sel:
            fr = frames[j]
            rgb, mask, W, H = _load_rgb_mask(d, fr, self.s)
            K = intrinsics(fr["fx"] / self.s, fr["fy"] / self.s,
                           fr["cx"] / self.s, fr["cy"] / self.s, "cpu")
            sup.append(dict(rgb=torch.from_numpy(rgb), mask=torch.from_numpy(mask), K=K,
                            c2w=torch.tensor(fr["c2w"]).float(), W=W, H=H,
                            face_box=_face_box(d, fr, self.s, W, H)))
        return dict(idx=i, ref_in=ref_in, label=label, canon=self.canon[i],
                    anc=self.anc[i], ref_rgb=ref_rgb, ref_K=ref_K, ref_c2w=ref_c2w,
                    face_crop=fcrop, sup=sup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default='received_data_from_Pinstudio_20260424/unzipped/0423')
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--k_sup", type=int, default=3)
    ap.add_argument("--all_views", action="store_true",
                    help="supervise on ALL train views each iter (not k_sup random); ref still random 1")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (use >0 with --all_views: each item loads ~30 NFS images)")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--scale_div", type=int, default=4)
    ap.add_argument("--n_root", type=int, default=26000)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--v7", action="store_true",
                    help="v7: triplane feature field + level-2 body + bigger backbone + region short-fur")
    ap.add_argument("--v8", action="store_true",
                    help="v8 (FUR_V8_PLAN): per-strand curl head + face/paw fur-kill + FREE body gaussians (implies v7)")
    ap.add_argument("--nofur_face_thr", type=float, default=0.3, help="v8: w_head>thr -> kill fur (face/nose)")
    ap.add_argument("--body_off_bound", type=float, default=0.05, help="v8: body gaussian canonical offset bound (x diag)")
    ap.add_argument("--body_base_sc", type=float, default=0.010, help="v8: body gaussian base scale floor (x diag)")
    ap.add_argument("--body_thin", type=float, default=0.1, help="v8: body gaussian normal-axis scale factor (flat surfel)")
    ap.add_argument("--render_only", default=None, help="render one dog (uses --only to pick scene) to a montage then exit")
    ap.add_argument("--render_decomp", action="store_true", help="render_only: dump full|body-only|fur-only|GT decomposition")
    ap.add_argument("--render_ref", action="store_true", help="render_only: also render at the REF camera pose (the seen view; GT=ref photo)")
    ap.add_argument("--v9", action="store_true", help="v9 (FUR_V9_PLAN): + single-image Splatter pixel-aligned branch (implies v8)")
    ap.add_argument("--v10", action="store_true", help="v10: v8 geometry + COLOR fully predicted (drop ref-sample blend + tone) (implies v8)")
    ap.add_argument("--proj_floor", type=float, default=0.0, help="v11: floor ref-projection weight where surface is visible (0=off; 0.7=projection dominates seen side)")
    ap.add_argument("--splat_res", type=int, default=128, help="v9 splat grid resolution (RxR pixel-gaussians)")
    ap.add_argument("--splat_base_sc", type=float, default=0.004, help="v9 splat base scale (x diag)")
    ap.add_argument("--splat_dres", type=float, default=0.05, help="v9 splat depth residual bound (x diag)")
    ap.add_argument("--anchors", default="fur_anchors.npz",
                    help="anchor npz name (v7 -> fur_anchors_v7.npz)")
    ap.add_argument("--dim", type=int, default=384, help="backbone width (v7: 768)")
    ap.add_argument("--n_layers", type=int, default=4, help="transformer depth (v7: 12)")
    ap.add_argument("--n_heads", type=int, default=6, help="attention heads (v7: 12)")
    ap.add_argument("--tri_res", type=int, default=64, help="v7 triplane resolution")
    ap.add_argument("--tri_ch", type=int, default=32, help="v7 triplane channels")
    ap.add_argument("--fur_op", type=float, default=0.7, help="opacity-gate init (max 0.9)")
    ap.add_argument("--radius_frac", type=float, default=0.0032)
    ap.add_argument("--fur_aspect", type=float, default=0.0,
                    help="cap fur-gaussian cross-section to half-seg/aspect (thin strands); 0=off")
    ap.add_argument("--w_adv", type=float, default=0.0,
                    help="fur-patch adversarial texture loss weight (PatchGAN, GANeRF-style)")
    ap.add_argument("--lr_d", type=float, default=2e-4, help="discriminator lr")
    ap.add_argument("--adv_start", type=int, default=0, help="iter to start adversarial loss")
    ap.add_argument("--adv_patch", type=int, default=160, help="fur crop size (px)")
    ap.add_argument("--adv_n", type=int, default=3, help="fur crops per iter")
    ap.add_argument("--w_mask", type=float, default=1.0)
    ap.add_argument("--w_lpips", type=float, default=0.15)
    ap.add_argument("--w_pen", type=float, default=5.0)
    ap.add_argument("--w_face_crop", type=float, default=1.0, help="0 disables the face crop loss")
    ap.add_argument("--w_sym", type=float, default=0.0,
                    help="left-right symmetry: occluded fur root color <- visible mirror twin's "
                         "ref-sampled color (canonical D-SMAL mirror across y); 0 disables")
    ap.add_argument("--w_sym_geo", type=float, default=0.0,
                    help="v6 B+: also pull occluded roots' GEOMETRY (length-factor/droop/gamma) "
                         "toward their visible Y-mirror twin (param[i]<-param[pi[i]].detach); "
                         "shares the --w_sym mirror map / visibility; 0 disables")
    ap.add_argument("--w_coat", type=float, default=0.0,
                    help="v6 C: CE loss on the coat-class head (coat_emb -> 6 logits) vs VLM curl_id, "
                         "making curl_emb a meaningful continuous coat code; 0 disables")
    ap.add_argument("--w_sds", type=float, default=0.0,
                    help="SDS: distill a frozen SD-2.1 prior on a random novel view; 0 disables")
    ap.add_argument("--sds_every", type=int, default=2, help="apply SDS every N iters (compute)")
    ap.add_argument("--sds_guidance", type=float, default=40.0, help="classifier-free guidance scale")
    ap.add_argument("--sds_start", type=int, default=0, help="iter to start SDS")
    ap.add_argument("--face_crop", action="store_true",
                    help="re-encode a high-res face crop as the face token set (sharper face)")
    ap.add_argument("--lpips_start", type=int, default=300)
    ap.add_argument("--vis_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--init_ckpt", default=None, help="optional r4 warm start (zero-padded heads)")
    ap.add_argument("--v6_repr", action="store_true",
                    help="v6 coat-adaptive length recipe (FUR_V6_PLAN s8.2): global-short + paw-short "
                         "+ offset-shell fed as FIXED anc knobs (length NOT freely optimized)")
    ap.add_argument("--len_short", type=float, default=0.5, help="v6 global length factor")
    ap.add_argument("--paw_len", type=float, default=0.3, help="v6 length factor on paw-mask verts")
    ap.add_argument("--offset_shell", type=float, default=0.2, help="v6 strand-origin lift (x L)")
    ap.add_argument("--paw_mask", default="synth_fur/paw_mask.npy")
    ap.add_argument("--w_len_prior", type=float, default=0.0,
                    help="v6: per-part length-prior reg keeping the network length factor near 1 "
                         "(length recipe stays fixed; this just discourages free length drift); 0 off")
    ap.add_argument("--w_orient", type=float, default=0.0,
                    help="v7: image-flow orientation loss -- align rendered fur strand-tangent "
                         "with the GT photo's coarse fur-flow (structure tensor, undirected mod-pi), "
                         "confidence-weighted; guides strand DIRECTION which RGB under-constrains. 0 off")
    ap.add_argument("--orient_sigma", type=float, default=16.0,
                    help="structure-tensor scale (px@s2). ~16=macro fur-flow; fine=texture noise")
    ap.add_argument("--orient_coh_thr", type=float, default=0.3,
                    help="only supervise where photo flow is coherent (anisotropy > thr)")
    ap.add_argument("--orient_start", type=int, default=0, help="iter to start orientation loss")
    ap.add_argument("--only", default=None)
    ap.add_argument("--split", default=None,
                    help="held-out split json (v6); train only on its dogs (see --split_half)")
    ap.add_argument("--split_half", default="train", choices=["train", "test"])
    ap.add_argument("--out", default="exps/dog_lrm_fur_v2")
    args = ap.parse_args()
    if args.v10:
        args.v8 = True                       # v10 = v8 geometry + fully-predicted color
    if args.v9:
        args.v8 = True                       # v9 = v8 + Splatter pixel-aligned branch
    if args.v8:
        args.v7 = True                       # v8 builds on the v7 backbone/triplane/level-2 body

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"
    if is_main():
        os.makedirs(args.out, exist_ok=True)

    scenes = list_scenes(args.root)
    if args.only:
        scenes = [s for s in scenes if args.only in s]
    if args.split:                                            # v6 held-out: train on split subset
        want = set(json.load(open(args.split))[args.split_half])
        scenes = [s for s in scenes if s.split("/")[-2] in want]
        if is_main():
            print(f"[split] {args.split} {args.split_half}: {len(scenes)} dogs", flush=True)
    da0 = np.load(os.path.join(scenes[0], "preprocess", "dsmal_anchors.npz"))
    w_face = torch.from_numpy(da0["w_face"])
    faces0 = torch.from_numpy(da0["faces"]).long()
    subdiv_M = build_subdiv(faces0, 1, dev)
    subdivide = lambda x: torch.stack([torch.sparse.mm(subdiv_M, x[b]) for b in range(x.shape[0])])

    ds = FurScenes(scenes, args.scale_div, args.k_sup, args.n_root, all_views=args.all_views,
                   anchors=args.anchors, v7=args.v7)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, collate_fn=collate, drop_last=True)

    w_face_s = torch.sparse.mm(subdiv_M, w_face[:, None].float().to(dev))[:, 0].cpu()
    Net = DogLRMFurV10 if args.v10 else (DogLRMFurV9 if args.v9 else (DogLRMFurV8 if args.v8 else (DogLRMFurV7 if args.v7 else DogLRMFurV2)))
    net_kw = dict(K=args.K, fur_op=args.fur_op, radius_frac=args.radius_frac,
                  dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
    if args.v7:
        net_kw.update(tri_res=args.tri_res, tri_ch=args.tri_ch)
    if args.v8:
        net_kw.update(body_off_bound=args.body_off_bound, body_base_sc=args.body_base_sc,
                      body_thin=args.body_thin)
    if args.v9:
        net_kw.update(splat_res=args.splat_res, splat_base_sc=args.splat_base_sc,
                      splat_dres=args.splat_dres)
    model = Net(w_face, faces_sub=subdivided_faces(faces0, 1), w_face_s=w_face_s,
                **net_kw).to(dev)
    if args.init_ckpt:
        if args.v7:                                           # v7->v7 resume: same arch, strict-safe
            sd = torch.load(args.init_ckpt, map_location=dev)  # ckpt excludes frozen dino.*
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if is_main():
                nondino = [k for k in missing if not k.startswith("dino.")]
                print(f"[init] v7 resume {args.init_ckpt}: {len(unexpected)} unexpected, "
                      f"{len(nondino)} non-dino missing {nondino[:5]}", flush=True)
        else:                                                 # v6/r5: zero-pad widened heads
            load_fur_ckpt(model, args.init_ckpt, dev)
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    core = model.module if ddp else model
    core.proj_floor = args.proj_floor                        # v11: floored ref-projection (0=off)
    if is_main():
        n_body = int(subdivided_faces(faces0, 2).max()) + 1 if args.v7 \
            else 15550 + core.face_edges.shape[0]
        n_fur = args.n_root * (args.K - 1)
        n_tot = n_fur + n_body
        print(f"{len(scenes)} scenes | n_root={args.n_root} K={args.K} v7={args.v7} | "
              f"fur {n_fur} + body {n_body} = {n_tot} gaussians/dog", flush=True)
        assert n_tot < 500_000, f"gaussian budget exceeded: {n_tot}"
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
    for p in lpips_fn.parameters():
        p.requires_grad = False
    white = torch.ones(3, device=dev)

    paw_short_v = None
    if args.v6_repr:                                          # per-vertex paw length factor [Vs]
        paw = torch.from_numpy(np.load(args.paw_mask)).float().to(dev)   # 1=paw
        paw_short_v = 1.0 - (1.0 - args.paw_len) * paw         # paw_len on paws, 1 elsewhere
        if is_main():
            print(f"[v6_repr] len_short={args.len_short} paw_len={args.paw_len} "
                  f"offset_shell={args.offset_shell} (length FIXED, not optimized)", flush=True)

    nofur_v = None
    if args.v8:                                   # v8: SMAL face(head+jaw)+paw -> KILL fur opacity
        paw = torch.from_numpy(np.load(args.paw_mask)).float().to(dev)
        a0 = np.load(os.path.join(scenes[0], "preprocess", args.anchors))
        wh0 = torch.from_numpy(a0["w_head"]).float().to(dev)            # head+jaw skin weight [Vs]
        nofur_v = ((wh0 > args.nofur_face_thr) | (paw > 0.5)).float()   # 1 = no fur (shared topo mask)
        if is_main():
            print(f"[v8] nofur(face+paw) {int(nofur_v.sum())}/{nofur_v.numel()} verts | "
                  f"body free-geo off={args.body_off_bound} sc={args.body_base_sc}", flush=True)

    if args.render_only:                                  # controlled static render of one dog -> montage, exit
        import imageio.v2 as imageio
        cm = model.module if ddp else model
        cm.eval()
        np.random.seed(1234)                              # deterministic ref+sup views -> matched across versions
        b = ds[0]
        ref_in = b["ref_in"][None].to(dev); label = b["label"][None].to(dev); canon = b["canon"][None].to(dev)
        anc = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in b["anc"].items()}
        anc["ref_rgb"] = b["ref_rgb"][None].to(dev); anc["ref_K"] = b["ref_K"][None].to(dev)
        anc["ref_c2w"] = b["ref_c2w"][None].to(dev)
        if b.get("face_crop") is not None:
            anc["face_crop"] = b["face_crop"][None].to(dev)
        if args.v6_repr:
            anc["len_short"] = args.len_short; anc["paw_short"] = paw_short_v[None]
            if nofur_v is not None:
                anc["nofur"] = nofur_v[None]
            anc["offset_shell"] = args.offset_shell
        if args.fur_aspect > 0:
            anc["fur_aspect"] = args.fur_aspect
        with torch.no_grad():
            fur, body = cm(ref_in, label, canon, anc, subdivide)
            f0, b0 = fur[0], body[0]
            full = {k: torch.cat([b0[k], f0[k]]) for k in ("means", "quats", "scales", "opacities", "sh")}
            KEYS = ("means", "quats", "scales", "opacities", "sh")
            def _ren(g, v):
                c2w, K = v["c2w"].to(dev), v["K"].to(dev)
                r, _ = render_gaussians(g["means"], g["quats"], g["scales"], g["opacities"], g["sh"],
                                        c2w, K, v["W"], v["H"], bg=white, sh_degree=1)
                return r
            if args.render_ref:                              # render at the REF pose (the seen view; GT=ref photo)
                H, W = b["ref_rgb"].shape[1:]
                rv = {"c2w": b["ref_c2w"], "K": b["ref_K"], "W": W, "H": H}
                rgb = _ren(full, rv)
                rb = _ren({k: b0[k] for k in KEYS}, rv)
                gt_w = b["ref_rgb"].permute(1, 2, 0).to(dev)
                panel = (torch.cat([rgb, rb, gt_w], 1).clamp(0, 1) * 255).byte().cpu().numpy()
                imageio.imwrite(os.path.join(args.out, f"refview_{args.render_only}.png"), panel)
            for vi, v in enumerate(b["sup"]):
                gt, m = v["rgb"].to(dev), v["mask"].to(dev)
                rgb = _ren(full, v)
                if args.render_decomp:                       # full | body-only | fur-only | GT
                    rb = _ren({k: b0[k] for k in KEYS}, v)
                    rf = _ren({k: f0[k] for k in KEYS}, v)
                    gt_w = gt * m + (1 - m) * white
                    panel = (torch.cat([rgb, rb, rf, gt_w], 1).clamp(0, 1) * 255).byte().cpu().numpy()
                    imageio.imwrite(os.path.join(args.out, f"decomp_{args.render_only}_v{vi}.png"), panel)
                    continue
                gt_w = gt * m + (1 - m) * white
                panel = (torch.cat([rgb, gt_w], 1).clamp(0, 1) * 255).byte().cpu().numpy()
                imageio.imwrite(os.path.join(args.out, f"render_{args.render_only}_v{vi}.png"), panel)
        print(f"RENDER_DONE -> {args.out}/render_{args.render_only}_v*.png ({len(b['sup'])} views)", flush=True)
        return

    sds = None
    if args.w_sds > 0:
        from dog_lrm.sds import SDSGuidance
        sds = SDSGuidance(dev)
        if is_main():
            print("SDS prior loaded (SD-1.5)", flush=True)
    netD = optD = None
    if args.w_adv > 0:
        netD = PatchD().to(dev)                  # NOT DDP-wrapped: D backward is conditional on
        # having valid fur crops per-rank -> DDP allreduce on netD would desync/timeout (the v8_adv
        # NCCL crash). Local-per-rank D (grads not synced) is fine; the generator/model DDP backward
        # stays symmetric (always gets rgb/lpips grad), so the model still syncs correctly.
        optD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.0, 0.99))
        if is_main():
            print(f"fur-patch adversarial loss ON (w_adv={args.w_adv}, patch={args.adv_patch})", flush=True)
    it, epoch, sym_cache = 0, 0, {}
    while it < args.iters:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if it >= args.iters:
                break
            b = batch[0]
            ref_in = b["ref_in"][None].to(dev)
            label = b["label"][None].to(dev)
            canon = b["canon"][None].to(dev)
            anc = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in b["anc"].items()}
            anc["ref_rgb"] = b["ref_rgb"][None].to(dev)
            anc["ref_K"] = b["ref_K"][None].to(dev)
            anc["ref_c2w"] = b["ref_c2w"][None].to(dev)
            if args.face_crop and b["face_crop"] is not None:
                anc["face_crop"] = b["face_crop"][None].to(dev)
            if args.v6_repr:                                  # fixed length recipe (s8.2)
                anc["len_short"] = args.len_short
                anc["paw_short"] = paw_short_v[None]
                if nofur_v is not None:
                    anc["nofur"] = nofur_v[None]
                anc["offset_shell"] = args.offset_shell
            if args.fur_aspect > 0:                           # thin-strand gaussian cap
                anc["fur_aspect"] = args.fur_aspect
            fur, body = model(ref_in, label, canon, anc, subdivide)
            f0, b0 = fur[0], body[0]
            full = dict(means=torch.cat([b0["means"], f0["means"]]),
                        quats=torch.cat([b0["quats"], f0["quats"]]),
                        scales=torch.cat([b0["scales"], f0["scales"]]),
                        opacities=torch.cat([b0["opacities"], f0["opacities"]]),
                        sh=torch.cat([b0["sh"], f0["sh"]]))
            loss_rgb = loss_mask = loss_perc = loss_face = 0.0
            loss_orient = torch.zeros((), device=dev)
            n_crops = 0
            for v in b["sup"]:
                c2w, K = v["c2w"].to(dev), v["K"].to(dev)
                gt, m = v["rgb"].to(dev), v["mask"].to(dev)
                rgb, alpha = render_gaussians(full["means"], full["quats"], full["scales"],
                                              full["opacities"], full["sh"], c2w, K,
                                              v["W"], v["H"], bg=white, sh_degree=1)
                area = m.sum().clamp_min(1)
                loss_rgb = loss_rgb + ((rgb - gt).abs() * m).sum() / (area * 3)
                loss_mask = loss_mask + F.l1_loss(alpha, m)
                if args.w_orient > 0 and it >= args.orient_start:
                    loss_orient = loss_orient + orient_loss(
                        f0, gt, m, c2w, K, v["W"], v["H"], dev,
                        args.orient_sigma, args.orient_coh_thr)
                if it >= args.lpips_start:
                    gt_w = gt * m + (1 - m) * white
                    r2 = F.interpolate(rgb.permute(2, 0, 1)[None] * 2 - 1, 256,
                                       mode="bilinear", align_corners=False)
                    g2 = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                       mode="bilinear", align_corners=False)
                    loss_perc = loss_perc + lpips_fn(r2, g2).mean()
                # high-res face crop supervision: cropping a pinhole image is a pure
                # principal-point shift, so render the crop at its native resolution
                if args.w_face_crop > 0 and n_crops < 2 and v["face_box"] is not None:
                    x0, y0, x1, y1 = v["face_box"]
                    Kc = K.clone()
                    Kc[0, 2] -= x0
                    Kc[1, 2] -= y0
                    rc, _ = render_gaussians(full["means"], full["quats"], full["scales"],
                                             full["opacities"], full["sh"], c2w, Kc,
                                             x1 - x0, y1 - y0, bg=white, sh_degree=1)
                    gc, mc = gt[y0:y1, x0:x1], m[y0:y1, x0:x1]
                    fc = 0.5 * ((rc - gc).abs() * mc).sum() / (mc.sum().clamp_min(1) * 3)
                    if it >= args.lpips_start:
                        gc_w = gc * mc + (1 - mc) * white
                        rc2 = F.interpolate(rc.permute(2, 0, 1)[None] * 2 - 1, 256,
                                            mode="bilinear", align_corners=False)
                        gc2 = F.interpolate(gc_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                            mode="bilinear", align_corners=False)
                        fc = fc + 0.15 * lpips_fn(rc2, gc2).mean()
                    loss_face = loss_face + fc
                    n_crops += 1
            pen = F.relu(-((f0["pts"] - f0["root"][:, None]) * f0["nrm"][:, None]).sum(-1)).mean()
            # left-right symmetry: occluded root's predicted color <- visible mirror twin's
            # observed (ref-sampled) color. Canonical D-SMAL mirror across y (exact pairing).
            loss_sym = torch.zeros((), device=dev)
            loss_sym_geo = torch.zeros((), device=dev)
            if args.w_sym > 0 or args.w_sym_geo > 0:
                si = b["idx"]
                if si not in sym_cache:
                    cr = core._interp(subdivide(canon), core.faces_sub[anc["root_face"]],
                                      anc["root_bary"])[0]                # [N,3] canonical roots
                    mir = cr * torch.tensor([1.0, -1.0, 1.0], device=dev)
                    sym_cache[si] = nn_index(cr, mir)
                pi = sym_cache[si]
                samp, vis = core.sample_ref(f0["root"][None], f0["nrm"][None], anc)
                samp, vis = samp[0], vis[0, :, 0]                          # [N,3], [N]
                occ = (vis < 0.5) & (vis[pi] > 0.5)                        # occluded, mirror seen
                if args.w_sym > 0 and occ.any():
                    rgb_root = f0["rgb"].reshape(pi.shape[0], core.Kp - 1, 3)[:, 0]
                    loss_sym = ((rgb_root[occ] - samp[pi][occ].detach()) ** 2).mean()
                # v6 B+: pull occluded roots' geometry toward the visible mirror twin's
                # (length-factor/droop/gamma). param[i] <- param[pi[i]].detach() so only the
                # occluded side moves; the visible side stays anchored by photometric.
                if args.w_sym_geo > 0 and occ.any():
                    for key in ("len_mult", "droop", "gamma"):
                        g = f0[key]                                       # [N]
                        loss_sym_geo = loss_sym_geo + ((g[occ] - g[pi][occ].detach()) ** 2).mean()
            # v6: length-prior reg -- keep the network's FREE length factor near neutral (1.0)
            # so length is governed by the fixed recipe, not freely optimized (FUR_V6_PLAN s8.2).
            loss_lenp = torch.zeros((), device=dev)
            if args.w_len_prior > 0:
                loss_lenp = ((f0["len_mult"] - 1.0) ** 2).mean()
            # v6 C: coat-class CE on the coat head -> makes curl_emb a meaningful coat code
            loss_coat = torch.zeros((), device=dev)
            if args.w_coat > 0:
                logits = core.coat_head(core.curl_emb(anc["curl_id"]))    # [B,6]
                loss_coat = F.cross_entropy(logits, anc["curl_id"].long())
            # SDS: distill SD-2.1 prior on a random novel view (Farm3D-style) -> hallucinate
            # plausible appearance where real views give no supervision. Inference unchanged.
            loss_sds = torch.zeros((), device=dev)
            if sds is not None and it >= args.sds_start and it % args.sds_every == 0:
                mu = full["means"]
                ctr = mu.mean(0).detach()
                rad = (mu - ctr).norm(dim=1).quantile(0.9).detach()
                az = float(torch.rand(1)) * 360.0
                el = -10.0 + float(torch.rand(1)) * 25.0
                c2w_n = look_at(ctr, az, el, 2.6 * float(rad), dev)
                Kn = intrinsics(560.0, 560.0, 256.0, 256.0, dev)
                rgb_n, _ = render_gaussians(full["means"], full["quats"], full["scales"],
                                            full["opacities"], full["sh"], c2w_n, Kn, 512, 512,
                                            bg=white, sh_degree=1)
                loss_sds = sds(rgb_n.permute(2, 0, 1)[None].clamp(0, 1), guidance=args.sds_guidance)
            n = len(b["sup"])
            # fur-patch adversarial texture loss (generator side): make rendered fur crops
            # look real to D. Uses the last sup view's render (has grad -> flows to gaussians).
            loss_adv = torch.zeros((), device=dev)
            fake_p = real_p = None
            if netD is not None and it >= args.adv_start:
                fake_p, real_p = fur_patches(rgb, gt, m, args.adv_patch, args.adv_n)
                if fake_p is not None:
                    loss_adv = -netD(fake_p).mean()
            loss = (loss_rgb + args.w_mask * loss_mask + args.w_lpips * loss_perc) / n \
                + args.w_pen * pen \
                + args.w_face_crop * loss_face / max(n_crops, 1) \
                + args.w_sym * loss_sym + args.w_sym_geo * loss_sym_geo \
                + args.w_coat * loss_coat + args.w_len_prior * loss_lenp \
                + args.w_orient * loss_orient / n \
                + args.w_sds * loss_sds + args.w_adv * loss_adv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            # discriminator step (hinge): real GT fur -> +1, rendered fur -> -1
            loss_d = torch.zeros((), device=dev)
            if fake_p is not None:
                optD.zero_grad()
                d_real = netD(real_p.detach()); d_fake = netD(fake_p.detach())
                loss_d = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
                loss_d.backward()
                optD.step()

            if is_main() and it % 50 == 0:
                lp = float(loss_perc) / n if it >= args.lpips_start else 0.0
                print(f"it{it:5d} loss={float(loss):.4f} rgb={float(loss_rgb)/n:.4f} "
                      f"mask={float(loss_mask)/n:.4f} lpips={lp:.4f} pen={float(pen):.5f} "
                      f"face={float(loss_face)/max(n_crops,1):.4f} sym={float(loss_sym):.4f} "
                      f"symg={float(loss_sym_geo):.4f} coat={float(loss_coat):.3f} "
                      f"orient={float(loss_orient)/n:.4f} "
                      f"sds={float(loss_sds):.3f} adv={float(loss_adv):.3f} d={float(loss_d):.3f}", flush=True)
            if is_main() and it % args.vis_every == 0:
                with torch.no_grad():
                    v = b["sup"][0]
                    rgb, _ = render_gaussians(full["means"], full["quats"], full["scales"],
                                              full["opacities"], full["sh"],
                                              v["c2w"].to(dev), v["K"].to(dev),
                                              v["W"], v["H"], bg=white, sh_degree=1)
                    pair = np.concatenate([(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                           (v["rgb"].numpy() * 255).astype(np.uint8)], 1)
                    Image.fromarray(pair).save(os.path.join(args.out, f"it{it:05d}.png"))
                    fb = next((vv for vv in b["sup"] if vv["face_box"] is not None), None)
                    if fb is not None:
                        x0, y0, x1, y1 = fb["face_box"]
                        Kc = fb["K"].to(dev).clone()
                        Kc[0, 2] -= x0
                        Kc[1, 2] -= y0
                        rc, _ = render_gaussians(full["means"], full["quats"], full["scales"],
                                                 full["opacities"], full["sh"],
                                                 fb["c2w"].to(dev), Kc, x1 - x0, y1 - y0,
                                                 bg=white, sh_degree=1)
                        pair2 = np.concatenate(
                            [(rc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                             (fb["rgb"][y0:y1, x0:x1].numpy() * 255).astype(np.uint8)], 1)
                        Image.fromarray(pair2).save(os.path.join(args.out, f"it{it:05d}_face.png"))
            if is_main() and args.save_every and it % args.save_every == 0 and it > 0:
                sd = {k: v for k, v in core.state_dict().items() if not k.startswith("dino.")}
                torch.save(sd, os.path.join(args.out, "model.pt"))
            it += 1
        epoch += 1

    if is_main():
        sd = {k: v for k, v in core.state_dict().items() if not k.startswith("dino.")}
        torch.save(sd, os.path.join(args.out, "model.pt"))
        print(f"done -> {args.out}/model.pt", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
