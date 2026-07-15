#!/usr/bin/env python3
"""Full-dataset Dog-LRM training with DDP across GPUs.

Scales the v1 trainer to many scenes + multi-GPU:
  * lazy per-scene IO (a Dataset loads 1 reference view + K supervision views per item,
    instead of v1 holding every scene's every view in GPU memory),
  * data-parallel over GPUs via torchrun/DDP (each rank handles B_local scenes/step),
  * same model + masked-RGB/mask/LPIPS losses + teacher-forced posed-SMAL anchoring.

Launch (5 GPUs, physical 1..5):
  CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 train_dog_lrm_ddp.py \
      --root received_data_from_Pinstudio_20260424/unzipped/0423 --iters 20000
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
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt

# Module-level (so spawn-ed DataLoader workers, which re-import but never run main(),
# inherit it too): /dev/shm here is only 64M, so route worker->main tensor passing
# through tmp files instead of POSIX shm to avoid "Bus error".
torch.multiprocessing.set_sharing_strategy("file_system")


def list_scenes(root, min_img=50):
    out = []
    for c in sorted(glob.glob(os.path.join(root, "*", "colmap"))):
        sp = os.path.join(c, "sparse", "0")
        if not all(os.path.exists(os.path.join(sp, f))
                   for f in ("cameras.txt", "images.txt", "points3D.txt")):
            continue
        if not os.path.exists(os.path.join(c, "preprocess", "smal_params.json")):
            continue
        if len(glob.glob(os.path.join(c, "images", "*.jpg"))) >= min_img:
            out.append(c)
    return out


def _load_rgb_mask(scene_dir, fr, s):
    W, H = fr["width"] // s, fr["height"] // s
    stem = os.path.splitext(fr["name"])[0]
    cache = os.path.join(scene_dir, "preprocess", f"cache_s{s}")
    cj, cm = os.path.join(cache, stem + ".jpg"), os.path.join(cache, stem + ".png")
    if os.path.exists(cj) and os.path.exists(cm):     # pre-downscaled cache -> trivial IO
        rgb = np.asarray(Image.open(cj).convert("RGB"), np.float32) / 255.
        mask = np.asarray(Image.open(cm).convert("L"), np.float32) / 255.
    else:                                             # fall back to full-res + draft decode
        img = Image.open(os.path.join(scene_dir, fr["image_path"]))
        img.draft("RGB", (W, H))
        rgb = np.asarray(img.convert("RGB").resize((W, H)), np.float32) / 255.
        m = Image.open(os.path.join(scene_dir, "preprocess", "masks", stem + ".png")).convert("L")
        mask = np.asarray(m.resize((W, H)), np.float32) / 255.
    H, W = rgb.shape[:2]
    return rgb, mask[..., None], W, H


class DogScenes(Dataset):
    """One item == one scene: a random reference view (224 input) + K random
    supervision views (rgb/mask/intrinsics/c2w at 1/scale_div). Anchor verts are
    precomputed once from each scene's frozen SMAL fit."""

    def __init__(self, scene_dirs, smal, num_betas, scale_div, k_sup, ref_res=896):
        self.scene_dirs = scene_dirs
        self.s, self.k, self.ref_res = scale_div, k_sup, ref_res
        self.frames = [json.load(open(os.path.join(d, "preprocess", "cameras.json")))["frames"]
                       for d in scene_dirs]
        self.canon, self.posed = [], []           # cached CPU anchor verts per scene
        dev = smal.device
        with torch.no_grad():
            for d in scene_dirs:
                gt = load_pseudo_gt(d, "preprocess", num_betas, dev)
                self.canon.append(smal.canonical_verts(gt["betas"], gt["limbs"])[0].cpu())
                self.posed.append(smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"],
                                                   gt["trans"], gt["scale"])[0].cpu())

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, i):
        d, frames = self.scene_dirs[i], self.frames[i]
        n = len(frames)
        ref = np.random.randint(n)
        fr_ref = frames[ref]
        rgb_r, _, _, _ = _load_rgb_mask(d, fr_ref, self.s)
        t = torch.from_numpy(rgb_r).permute(2, 0, 1)[None]
        ref_in = F.interpolate(t, size=(224, 224), mode="bilinear",
                               align_corners=False)[0]
        R = self.ref_res                                   # hi-res ref for pixel-aligned sampling
        ref_hi = F.interpolate(t, size=(R, R), mode="bilinear", align_corners=False)[0]
        W0, H0 = fr_ref["width"], fr_ref["height"]
        ref_K = intrinsics(fr_ref["fx"] * R / W0, fr_ref["fy"] * R / H0,
                           fr_ref["cx"] * R / W0, fr_ref["cy"] * R / H0, "cpu")
        ref_c2w = torch.tensor(fr_ref["c2w"]).float()
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
        return dict(idx=i, ref_in=ref_in, ref_hi=ref_hi, ref_K=ref_K, ref_c2w=ref_c2w,
                    canon=self.canon[i], posed=self.posed[i], sup=sup)


def collate(batch):
    return batch  # list of per-scene dicts; rendering loops per scene anyway


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--b_local", type=int, default=4, help="scenes per GPU per step")
    ap.add_argument("--k_sup", type=int, default=6, help="supervision views per scene")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_iters", type=int, default=0, help="linear LR warmup iters (0=off)")
    ap.add_argument("--lr_final_ratio", type=float, default=1.0,
                    help="cosine-decay LR to lr*ratio at --iters (1.0=constant)")
    ap.add_argument("--lpips_weight", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=500)
    ap.add_argument("--n_subdiv", type=int, default=2, help="loop subdiv levels (2~62k verts)")
    ap.add_argument("--offset_reg", type=float, default=0.1, help="ACAP: pull Gaussians to anchor")
    ap.add_argument("--offset_free", type=float, default=0.02, help="free offset distance (no penalty)")
    ap.add_argument("--scale_reg", type=float, default=0.1, help="ball: penalize anisotropic Gaussians")
    ap.add_argument("--scale_ratio", type=float, default=4.0, help="free max/min axis ratio")
    ap.add_argument("--scale_clip_start", type=float, default=0.02, help="max scale at it 0")
    ap.add_argument("--scale_clip_end", type=float, default=0.08, help="max scale after warmup")
    ap.add_argument("--scale_clip_warmup", type=int, default=1000, help="iters to ramp max scale (0=off)")
    ap.add_argument("--K", type=int, default=1, help="gaussians per anchor (1 = no redundancy)")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1", help="v2 = modernized MM-DiT (dit_v2)")
    ap.add_argument("--surf_samples", type=int, default=0,
                    help=">0: anchors = N random surface points (breaks subdiv lattice) instead of subdiv verts")
    ap.add_argument("--head_boost", type=float, default=4.0,
                    help="sampling density multiplier on head/muzzle/ear (with --surf_samples)")
    ap.add_argument("--rasterize_mode", choices=["classic", "antialiased"], default="classic",
                    help="antialiased = mip-splatting opacity compensation (kills dilation speckle)")
    ap.add_argument("--proj_feat", type=int, default=0,
                    help="1: pixel-aligned conditioning - anchors sample a hi-res ref feature map (needs --surf_samples)")
    ap.add_argument("--ref_res", type=int, default=896, help="hi-res ref image size for --proj_feat")
    ap.add_argument("--opacity_reg", type=float, default=0.5, help="pull opacity ->1 (solid body, anti-collapse)")
    ap.add_argument("--init_ckpt", default=None, help="warm-start non-dino weights from this model.pt")
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vis_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out", default="exps/dog_lrm_full")
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
    smal = SMALModel(dev, n_subdiv=args.n_subdiv)
    if is_main():
        print(f"{len(scenes)} scenes | world={dist.get_world_size() if ddp else 1} "
              f"| b_local={args.b_local} k_sup={args.k_sup}", flush=True)

    if args.surf_samples > 0:
        from dog_lrm.smal_model import build_surface_sampler
        w = smal.smal.weights.float().cpu()                  # [3889, Nj] LBS skin weights
        head_w = w[:, [15, 16, 32, 33, 34]].sum(1)           # head+muzzle+ear joints
        vert_w = 1.0 + (args.head_boost - 1.0) * (head_w / head_w.max()).clamp(0, 1)
        with torch.no_grad():                                # template (mean-shape) rest verts
            tmpl = smal.canonical_verts(torch.zeros(1, smal.num_betas, device=dev),
                                        torch.zeros(1, 7, device=dev))[0].cpu()
        samp_M, samp_fi = build_surface_sampler(smal.faces, tmpl, args.surf_samples,
                                                vert_weight=vert_w, seed=0, device=dev)
        subdiv_fn = lambda x: torch.stack([torch.sparse.mm(samp_M, x[b]) for b in range(x.shape[0])])
        if is_main():
            print(f"surface anchors: {args.surf_samples} (head_boost {args.head_boost})", flush=True)
    else:
        subdiv_fn = smal.subdivide

    if args.proj_feat:
        assert args.surf_samples > 0, "--proj_feat needs --surf_samples (face ids for normals)"
        faces_t = torch.as_tensor(smal.faces).long().to(dev)

    ds = DogScenes(scenes, smal, smal.num_betas, args.scale_div, args.k_sup, args.ref_res)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None
    # 'spawn' workers: the main process has already initialised CUDA (SMAL/model on GPU),
    # and forked DataLoader workers inheriting that context hang. spawn sidesteps it and
    # keeps async IO so the heavy JPG decodes overlap the render/forward.
    loader = DataLoader(ds, batch_size=args.b_local, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, collate_fn=collate, drop_last=True,
                        persistent_workers=args.workers > 0,
                        multiprocessing_context="spawn" if args.workers > 0 else None)

    if args.arch == "v2":
        from dog_lrm.model_v2 import DogLRMv2
        model = DogLRMv2(gaussians_per_point=args.K).to(dev)
    else:
        model = DogLRM(gaussians_per_point=args.K).to(dev)
    if args.init_ckpt:                                   # warm-start (non-dino weights)
        isd = torch.load(args.init_ckpt, map_location=dev)
        miss, unexp = model.load_state_dict(isd, strict=False)
        if is_main():
            print(f"warm-start {args.init_ckpt}: "
                  f"{len([k for k in miss if not k.startswith('dino.')])} non-dino missing, "
                  f"{len(unexp)} unexpected", flush=True)
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    core = model.module if ddp else model
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    import math
    def lr_lambda(step):                                 # warmup -> cosine to lr*final_ratio
        if args.warmup_iters > 0 and step < args.warmup_iters:
            return (step + 1) / args.warmup_iters
        if args.lr_final_ratio >= 1.0:
            return 1.0
        t = (step - args.warmup_iters) / max(1, args.iters - args.warmup_iters)
        return args.lr_final_ratio + (1 - args.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

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
            if args.scale_clip_warmup > 0:                        # ramp max scale (kill early floaters)
                t = min(1.0, it / args.scale_clip_warmup)
                scale_clip = args.scale_clip_start + t * (args.scale_clip_end - args.scale_clip_start)
            else:
                scale_clip = args.scale_clip_end
            ref_in = torch.stack([b["ref_in"] for b in batch]).to(dev)
            canon = torch.stack([b["canon"] for b in batch]).to(dev)
            posed = torch.stack([b["posed"] for b in batch]).to(dev)
            pk = {}
            if args.proj_feat:
                tri = posed[:, faces_t]                               # [B,F,3,3] posed base mesh
                fn = F.normalize(torch.linalg.cross(tri[:, :, 1] - tri[:, :, 0],
                                                    tri[:, :, 2] - tri[:, :, 0]), dim=-1)
                pk = dict(ref_hi=torch.stack([b["ref_hi"] for b in batch]).to(dev),
                          ref_K=torch.stack([b["ref_K"] for b in batch]).to(dev),
                          ref_c2w=torch.stack([b["ref_c2w"] for b in batch]).to(dev),
                          anchor_normals=fn[:, samp_fi])              # [B,N,3]
            gs = model(ref_in, canon, posed, subdivide=subdiv_fn, scale_clip=scale_clip, **pk)
            # Per-view backward through a DETACHED copy of the Gaussians so only one
            # view's rasterization graph is alive at a time (required for high-res
            # supervision); photometric grads collect on the detached leaves and are
            # routed through the model in a single final backward (one DDP allreduce).
            gkeys = ("means", "quats", "scales", "opacities", "rgb")
            det = {k: gs[k].detach().requires_grad_(True) for k in gkeys}
            n = sum(len(b["sup"]) for b in batch)
            loss_rgb = loss_mask = loss_perc = 0.0                # floats, logging only
            for bi, b in enumerate(batch):
                for v in b["sup"]:
                    c2w, K = v["c2w"].to(dev), v["K"].to(dev)
                    gtrgb, mask = v["rgb"].to(dev), v["mask"].to(dev)
                    rgb, alpha = render_gaussians(det["means"][bi], det["quats"][bi],
                                                  det["scales"][bi], det["opacities"][bi],
                                                  det["rgb"][bi], c2w, K, v["W"], v["H"], bg=white,
                                                  rasterize_mode=args.rasterize_mode)
                    l_rgb = F.l1_loss(rgb * mask, gtrgb * mask)
                    l_mask = F.l1_loss(alpha, mask)
                    l_view = l_rgb + l_mask
                    if it >= args.lpips_start:
                        gt_w = gtrgb * mask + (1 - mask) * white
                        r = F.interpolate(rgb.permute(2, 0, 1)[None] * 2 - 1, 256,
                                          mode="bilinear", align_corners=False)
                        g = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                          mode="bilinear", align_corners=False)
                        l_perc = lpips_fn(r, g).mean()
                        l_view = l_view + args.lpips_weight * l_perc
                        loss_perc += float(l_perc)
                    (l_view / n).backward()                       # frees this view's raster graph
                    loss_rgb += float(l_rgb)
                    loss_mask += float(l_mask)
            off = gs["offset"].norm(dim=-1)                       # ACAP: keep near anchor
            loss_off = (off.clamp(min=args.offset_free) - args.offset_free).mean()
            sc = gs["scales"]                                     # ball: discourage spiky Gaussians
            ratio = sc.max(dim=-1).values / (sc.min(dim=-1).values + 1e-6)
            loss_ball = (ratio.clamp(min=args.scale_ratio) - args.scale_ratio).mean()
            loss_op = (1.0 - gs["opacities"]).mean()             # keep anchors opaque (solid body)
            reg = (args.offset_reg * loss_off + args.scale_reg * loss_ball
                   + args.opacity_reg * loss_op)
            opt.zero_grad()
            # single backward through the model: inject accumulated photometric grads
            # at the gs outputs, plus the (still-attached) regularizers
            torch.autograd.backward([gs[k] for k in gkeys] + [reg],
                                    [det[k].grad for k in gkeys] + [torch.ones_like(reg)])
            loss = (loss_rgb + loss_mask + args.lpips_weight * loss_perc) / n + float(reg)
            # grad norm is post-allreduce -> identical on all ranks, so the skip is DDP-consistent
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(gn):
                opt.step()
            elif is_main():
                print(f"it{it:5d} SKIP step: non-finite grad norm", flush=True)
            sched.step()

            if is_main() and it % 50 == 0:
                lp = float(loss_perc) / n if it >= args.lpips_start else 0.0
                print(f"it{it:5d} loss={float(loss):.4f} rgb={float(loss_rgb)/n:.4f} "
                      f"mask={float(loss_mask)/n:.4f} lpips={lp:.4f} "
                      f"off={float(loss_off):.4f} ball={float(loss_ball):.4f} "
                      f"op={float(loss_op):.4f} sclip={scale_clip:.3f}",
                      flush=True)
            if is_main() and it % args.vis_every == 0:        # render-vs-gt tile (monitoring)
                with torch.no_grad():
                    v = batch[0]["sup"][0]
                    rgb, _ = render_gaussians(gs["means"][0], gs["quats"][0], gs["scales"][0],
                                              gs["opacities"][0], gs["rgb"][0], v["c2w"].to(dev),
                                              v["K"].to(dev), v["W"], v["H"], bg=white,
                                              rasterize_mode=args.rasterize_mode)
                    pair = np.concatenate(
                        [(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                         (v["rgb"].numpy() * 255).astype(np.uint8)], axis=1)
                    Image.fromarray(pair).save(os.path.join(args.out, f"it{it:05d}.png"))
                    gt = v["rgb"].to(dev); msk = v["mask"].to(dev)   # PSNR/LPIPS on this view
                    gt_w = gt * msk + (1 - msk) * white
                    mse = ((rgb.clamp(0, 1) - gt_w) ** 2).mean()
                    psnr = -10.0 * torch.log10(mse + 1e-8)
                    rr = F.interpolate(rgb.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1, 256,
                                       mode="bilinear", align_corners=False)
                    gg = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                       mode="bilinear", align_corners=False)
                    print(f"metric it{it:5d} PSNR={float(psnr):.2f} LPIPS={float(lpips_fn(rr, gg).mean()):.4f}",
                          flush=True)
            if is_main() and args.save_every and it % args.save_every == 0 and it > 0:
                sd = {k: v for k, v in core.state_dict().items() if not k.startswith("dino.")}
                torch.save(sd, os.path.join(args.out, "model.pt"))
                torch.save(sd, os.path.join(args.out, f"model_it{it:06d}.pt"))  # collapse-safe snapshot
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
