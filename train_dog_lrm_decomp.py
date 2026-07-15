#!/usr/bin/env python3
"""P1a: face/body-decomposed Dog-LRM (FUR_V2_PLAN), body Gaussians only.

Same training protocol/losses as train_dog_lrm_ddp.py (clean A/B), with:
  * geometry anchors from the external D-SMAL fits (preprocess/dsmal_anchors.npz),
  * DogLRMDecomp: region-restricted cross-attention, image tokens split by the
    projected face/body label grid of the reference view (region_masks_s8 cache).

Launch:
  CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 train_dog_lrm_decomp.py \
      --root received_data_from_Pinstudio_20260424/unzipped/0423 --iters 12000 --workers 0
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model_decomp import DogLRMDecomp, GRID
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.smal_model import build_subdiv
from train_dog_lrm_ddp import _load_rgb_mask, collate, is_main

torch.multiprocessing.set_sharing_strategy("file_system")


def list_scenes(root):
    out = []
    for c in sorted(glob.glob(os.path.join(root, "*", "colmap"))):
        pre = os.path.join(c, "preprocess")
        if (os.path.exists(os.path.join(pre, "dsmal_anchors.npz"))
                and os.path.isdir(os.path.join(pre, "region_masks_s8"))
                and os.path.exists(os.path.join(pre, "cameras.json"))):
            out.append(c)
    return out


def _dilate_up(m, px):
    from scipy.ndimage import binary_dilation
    out = binary_dilation(m, iterations=max(px // 2, 1))
    up = out.copy()
    for k in range(1, px + 1):
        up[:-k] |= out[k:]
    return up


def _label_grid(scene_dir, fr, gt_mask, grid=GRID, dilate_frac=0.05):
    """0=bg 1=body 2=face at [grid,grid], from the cached s8 region mask + GT mask."""
    stem = os.path.splitext(fr["name"])[0]
    rm = np.asarray(Image.open(os.path.join(scene_dir, "preprocess", "region_masks_s8",
                                            stem + ".png")))
    face, mesh = rm[:, :, 0] > 76, rm[:, :, 1] > 127           # R=w_face*255, G=mesh
    dog = (gt_mask[:, :, 0] > 0.5) | mesh
    if dog.any():
        diag = np.sqrt((np.argwhere(dog).ptp(0) ** 2).sum())
        face = _dilate_up(face, max(int(dilate_frac * diag), 2)) & dog
    t = lambda m: F.interpolate(torch.from_numpy(m).float()[None, None], size=(grid, grid),
                                mode="area")[0, 0]
    f74, d74 = t(face), t(dog & ~face)
    label = torch.zeros(grid, grid, dtype=torch.long)
    label[d74 > 0.1] = 1
    label[f74 > 0.08] = 2
    return label


class DecompScenes(Dataset):
    """One item == one scene: reference view (input + region label grid) + K
    supervision views. Anchors come from the cached D-SMAL fits."""

    def __init__(self, scene_dirs, scale_div, k_sup):
        self.scene_dirs = scene_dirs
        self.s, self.k = scale_div, k_sup
        self.frames = [json.load(open(os.path.join(d, "preprocess", "cameras.json")))["frames"]
                       for d in scene_dirs]
        self.canon, self.posed = [], []
        for d in scene_dirs:
            a = np.load(os.path.join(d, "preprocess", "dsmal_anchors.npz"))
            self.canon.append(torch.from_numpy(a["canon"]))
            self.posed.append(torch.from_numpy(a["posed"]))

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, i):
        d, frames = self.scene_dirs[i], self.frames[i]
        n = len(frames)
        ref = np.random.randint(n)
        rgb_r, mask_r, _, _ = _load_rgb_mask(d, frames[ref], self.s)
        label = _label_grid(d, frames[ref], mask_r)
        ref_in = torch.from_numpy(rgb_r).permute(2, 0, 1)[None]
        ref_in = F.interpolate(ref_in, size=(518, 518), mode="bilinear",
                               align_corners=False)[0]
        others = [j for j in range(n) if j != ref]
        sel = np.random.choice(others, size=min(self.k, len(others)), replace=False)
        sup = []
        for j in sel:
            fr = frames[j]
            rgb, mask, W, H = _load_rgb_mask(d, fr, self.s)
            K = intrinsics(fr["fx"] / self.s, fr["fy"] / self.s,
                           fr["cx"] / self.s, fr["cy"] / self.s, "cpu")
            sup.append(dict(rgb=torch.from_numpy(rgb), mask=torch.from_numpy(mask), K=K,
                            c2w=torch.tensor(fr["c2w"]).float(), W=W, H=H))
        return dict(idx=i, ref_in=ref_in, label=label, canon=self.canon[i],
                    posed=self.posed[i], sup=sup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--b_local", type=int, default=4)
    ap.add_argument("--k_sup", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lpips_weight", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=500)
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vis_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out", default="exps/dog_lrm_decomp")
    ap.add_argument("--only", default=None, help="substring filter: train on matching scenes only")
    ap.add_argument("--n_subdiv", type=int, default=1)
    ap.add_argument("--gs_per_pt", type=int, default=2)
    args = ap.parse_args()

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
    a0 = np.load(os.path.join(scenes[0], "preprocess", "dsmal_anchors.npz"))
    w_face = torch.from_numpy(a0["w_face"])                     # template-level: same for all dogs
    subdiv_M = build_subdiv(torch.from_numpy(a0["faces"]), args.n_subdiv, dev)
    subdivide = lambda x: torch.stack([torch.sparse.mm(subdiv_M, x[b]) for b in range(x.shape[0])])
    if is_main():
        print(f"{len(scenes)} scenes | world={dist.get_world_size() if ddp else 1} "
              f"| b_local={args.b_local} k_sup={args.k_sup} | face verts {(w_face > 0.5).sum()}",
              flush=True)

    ds = DecompScenes(scenes, args.scale_div, args.k_sup)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=args.b_local, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, collate_fn=collate, drop_last=True,
                        persistent_workers=args.workers > 0,
                        multiprocessing_context="spawn" if args.workers > 0 else None)

    model = DogLRMDecomp(w_face, gaussians_per_point=args.gs_per_pt).to(dev)
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    core = model.module if ddp else model
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
    for p in lpips_fn.parameters():
        p.requires_grad = False
    white = torch.ones(3, device=dev)

    it, epoch = 0, 0
    while it < args.iters:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if it >= args.iters:
                break
            ref_in = torch.stack([b["ref_in"] for b in batch]).to(dev)
            label = torch.stack([b["label"] for b in batch]).to(dev)
            canon = torch.stack([b["canon"] for b in batch]).to(dev)
            posed = torch.stack([b["posed"] for b in batch]).to(dev)
            gs = model(ref_in, label, canon, posed, subdivide=subdivide)
            loss_rgb = loss_mask = loss_perc = 0.0
            n = 0
            for bi, b in enumerate(batch):
                for v in b["sup"]:
                    c2w, K = v["c2w"].to(dev), v["K"].to(dev)
                    gtrgb, mask = v["rgb"].to(dev), v["mask"].to(dev)
                    rgb, alpha = render_gaussians(gs["means"][bi], gs["quats"][bi],
                                                  gs["scales"][bi], gs["opacities"][bi],
                                                  gs["rgb"][bi], c2w, K, v["W"], v["H"], bg=white)
                    loss_rgb = loss_rgb + F.l1_loss(rgb * mask, gtrgb * mask)
                    loss_mask = loss_mask + F.l1_loss(alpha, mask)
                    if it >= args.lpips_start:
                        gt_w = gtrgb * mask + (1 - mask) * white
                        r = F.interpolate(rgb.permute(2, 0, 1)[None] * 2 - 1, 256,
                                          mode="bilinear", align_corners=False)
                        g = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                          mode="bilinear", align_corners=False)
                        loss_perc = loss_perc + lpips_fn(r, g).mean()
                    n += 1
            loss = (loss_rgb + loss_mask + args.lpips_weight * loss_perc) / n
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

            if is_main() and it % 50 == 0:
                lp = float(loss_perc) / n if it >= args.lpips_start else 0.0
                print(f"it{it:5d} loss={float(loss):.4f} rgb={float(loss_rgb)/n:.4f} "
                      f"mask={float(loss_mask)/n:.4f} lpips={lp:.4f}", flush=True)
            if is_main() and it % args.vis_every == 0:
                with torch.no_grad():
                    v = batch[0]["sup"][0]
                    rgb, _ = render_gaussians(gs["means"][0], gs["quats"][0], gs["scales"][0],
                                              gs["opacities"][0], gs["rgb"][0], v["c2w"].to(dev),
                                              v["K"].to(dev), v["W"], v["H"], bg=white)
                    pair = np.concatenate(
                        [(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                         (v["rgb"].numpy() * 255).astype(np.uint8)], axis=1)
                    Image.fromarray(pair).save(os.path.join(args.out, f"it{it:05d}.png"))
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
