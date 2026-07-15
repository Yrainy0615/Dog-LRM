#!/usr/bin/env python3
"""Four-version feed-forward comparison: same dogs, same ref view, same render views.
Renders side-by-side strips + computes PSNR/LPIPS per (version, scene, view).

  PATH=$ENV/bin:$PATH TORCH_EXTENSIONS_DIR=.torch_ext_lhm python eval_ff_compare.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model_v2 import DogLRMv2
from dog_lrm.smal_model import SMALModel, load_pseudo_gt, build_surface_sampler
from dog_lrm.render import intrinsics, render_gaussians
from train_dog_lrm_ddp import _load_rgb_mask

DEV, S, R = "cuda", 2, 896
ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
SCENES = ["00086-tiara", "00002-nara", "00044-aura", "00110-choko"]
REF_IDX, VIEW_IDXS = 20, [0, 30, 60]
OUT = "exps/ff_comparison"

VERSIONS = [  # (tag, ckpt, anchors, rasterize_mode, pixel_aligned)
    ("A_62k_lowres", "exps/dog_lrm_v2_full/model.pt", "subdiv2", "classic", False),
    ("B_62k_4MP", "exps/dog_lrm_v2_hires62k/model.pt", "subdiv2", "classic", False),
    ("C_249k_4MP", "exps/dog_lrm_v2_hires300k/model.pt", "subdiv3", "classic", False),
    ("D_300k_PA", "exps/dog_lrm_v2_pa300k/model.pt", "surf300k", "antialiased", True),
]

os.makedirs(os.path.join(OUT, "imgs"), exist_ok=True)
smal2 = SMALModel(DEV, n_subdiv=2)
smal3 = SMALModel(DEV, n_subdiv=3)
w = smal2.smal.weights.float().cpu()
head_w = w[:, [15, 16, 32, 33, 34]].sum(1)
vert_w = 1.0 + 3.0 * (head_w / head_w.max()).clamp(0, 1)
with torch.no_grad():
    tmpl = smal2.canonical_verts(torch.zeros(1, smal2.num_betas, device=DEV),
                                 torch.zeros(1, 7, device=DEV))[0].cpu()
samp_M, samp_fi = build_surface_sampler(smal2.faces, tmpl, 300000, vert_weight=vert_w,
                                        seed=0, device=DEV)
SUBDIV = {"subdiv2": smal2.subdivide, "subdiv3": smal3.subdivide,
          "surf300k": lambda x: torch.stack([torch.sparse.mm(samp_M, x[b])
                                             for b in range(x.shape[0])])}
faces_t = torch.as_tensor(smal2.faces).long().to(DEV)

import lpips as lpips_lib
lpips_fn = lpips_lib.LPIPS(net="alex").to(DEV)
for p in lpips_fn.parameters():
    p.requires_grad = False
white = torch.ones(3, device=DEV)


def scene_data(name):
    scene = os.path.join(ROOT, name, "colmap")
    gt = load_pseudo_gt(scene, "preprocess", smal2.num_betas, DEV)
    canon = smal2.canonical_verts(gt["betas"], gt["limbs"])
    posed = smal2.posed_verts(gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])
    frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
    fr = frames[REF_IDX % len(frames)]
    rgb_r, _, _, _ = _load_rgb_mask(scene, fr, S)
    t = torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(DEV)
    ref_in = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    ref_hi = F.interpolate(t, size=(R, R), mode="bilinear", align_corners=False)
    W0, H0 = fr["width"], fr["height"]
    ref_K = intrinsics(fr["fx"] * R / W0, fr["fy"] * R / H0,
                       fr["cx"] * R / W0, fr["cy"] * R / H0, DEV)[None]
    ref_c2w = torch.tensor(fr["c2w"]).float().to(DEV)[None]
    tri = posed[:, faces_t]
    fn = F.normalize(torch.linalg.cross(tri[:, :, 1] - tri[:, :, 0],
                                        tri[:, :, 2] - tri[:, :, 0]), dim=-1)
    return dict(scene=scene, frames=frames, canon=canon, posed=posed, ref_in=ref_in,
                ref_hi=ref_hi, ref_K=ref_K, ref_c2w=ref_c2w, normals=fn[:, samp_fi])


metrics = {}
data = {n: scene_data(n) for n in SCENES}
for tag, ckpt, anchors, rmode, pa in VERSIONS:
    if not os.path.exists(ckpt):
        print(f"skip {tag}: {ckpt} missing")
        continue
    model = DogLRMv2(gaussians_per_point=1).to(DEV).eval()
    miss, _ = model.load_state_dict(torch.load(ckpt, map_location=DEV), strict=False)
    bad = [k for k in miss if not k.startswith(("dino.", "ref_cnn.", "proj_in."))]
    assert not bad, f"{tag}: unexpected missing {bad[:4]}"
    for name in SCENES:
        d = data[name]
        with torch.no_grad():
            pk = dict(ref_hi=d["ref_hi"], ref_K=d["ref_K"], ref_c2w=d["ref_c2w"],
                      anchor_normals=d["normals"]) if pa else {}
            gs = model(d["ref_in"], d["canon"], d["posed"], subdivide=SUBDIV[anchors], **pk)
            for vi in VIEW_IDXS:
                fr = d["frames"][vi % len(d["frames"])]
                K = intrinsics(fr["fx"] / S, fr["fy"] / S, fr["cx"] / S, fr["cy"] / S, DEV)
                gtrgb, mask, W, H = _load_rgb_mask(d["scene"], fr, S)
                c2w = torch.tensor(fr["c2w"]).float().to(DEV)
                rgb, _ = render_gaussians(gs["means"][0], gs["quats"][0], gs["scales"][0],
                                          gs["opacities"][0], gs["rgb"][0], c2w, K, W, H,
                                          bg=white, rasterize_mode=rmode)
                m = torch.from_numpy(mask).to(DEV)
                gt_w = torch.from_numpy(gtrgb).to(DEV) * m + (1 - m) * white
                psnr = float(-10 * torch.log10(((rgb.clamp(0, 1) - gt_w) ** 2).mean()))
                rr = F.interpolate(rgb.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                gg = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                lp = float(lpips_fn(rr, gg).mean())
                metrics[f"{tag}|{name}|{vi}"] = dict(psnr=psnr, lpips=lp)
                if vi == VIEW_IDXS[0]:
                    Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
                        os.path.join(OUT, "imgs", f"{tag}_{name}.png"))
        print(f"{tag} {name} done", flush=True)
    del model
    torch.cuda.empty_cache()

for name in SCENES:  # GT reference for the strip
    d = data[name]
    fr = d["frames"][VIEW_IDXS[0]]
    gtrgb, mask, W, H = _load_rgb_mask(d["scene"], fr, S)
    gt_w = gtrgb * mask + (1 - mask)
    Image.fromarray((gt_w * 255).astype(np.uint8)).save(
        os.path.join(OUT, "imgs", f"GT_{name}.png"))

json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print("metrics ->", os.path.join(OUT, "metrics.json"))
