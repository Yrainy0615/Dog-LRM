#!/usr/bin/env python3
"""Eval the feedforward strand-fur model: held-out comp PSNR + IoU per dog.

Same convention as eval_decomp (GT-bg composited PSNR, mask IoU); held-out views
= every 12th of the sorted frame list (excluded from training by FurScenes).

  PATH=$ENV/bin:$PATH python eval_fur_ff.py --ckpt exps/dog_lrm_fur_v2_r1/model.pt --S 12 --K 14
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model_fur import DogLRMFurV2, load_fur_ckpt
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_dog_lrm_decomp import _label_grid
from train_dog_lrm_fur_v2 import FurScenes, list_scenes, _face_box, face_crop_518


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--ckpt", default="exps/dog_lrm_fur_v2_r1/model.pt")
    ap.add_argument("--n_root", type=int, default=26000)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--fur_op", type=float, default=0.7)
    ap.add_argument("--radius_frac", type=float, default=0.0032)
    ap.add_argument("--vis", default=None, help="save first held-out render/GT pair here")
    ap.add_argument("--face_crop", action="store_true", help="use high-res face-crop tokens")
    ap.add_argument("--scale_div", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out_csv", default="exps/eval_fur_ff.csv")
    args = ap.parse_args()
    dev = "cuda"

    scenes = list_scenes(args.root)
    if args.only:
        scenes = [s for s in scenes if args.only in s]
    ds = FurScenes(scenes, args.scale_div, 1, args.n_root)
    da = np.load(os.path.join(scenes[0], "preprocess", "dsmal_anchors.npz"))
    w_face = torch.from_numpy(da["w_face"])
    faces0 = torch.from_numpy(da["faces"]).long()
    subdiv_M = build_subdiv(faces0, 1, dev)
    sub = lambda x: torch.stack([torch.sparse.mm(subdiv_M, x[b]) for b in range(x.shape[0])])
    w_face_s = torch.sparse.mm(subdiv_M, w_face[:, None].float().to(dev))[:, 0].cpu()
    model = DogLRMFurV2(w_face, faces_sub=subdivided_faces(faces0, 1), w_face_s=w_face_s,
                        K=args.K, fur_op=args.fur_op, radius_frac=args.radius_frac).to(dev)
    load_fur_ckpt(model, args.ckpt, dev)
    model.eval()
    n_tot = args.n_root * (args.K - 1) + 15550 + model.face_edges.shape[0]
    print(f"gaussians/dog: {n_tot}")
    assert n_tot < 300_000, n_tot
    white = torch.ones(3, device=dev)
    s = args.scale_div
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
    for p_ in lpips_fn.parameters():
        p_.requires_grad = False

    rows = []
    for i, scene in enumerate(scenes):
        frames = ds.frames[i]
        held = sorted(set(range(1, len(frames), 12)))
        # fixed face-best ref view (muzzle-visibility score from cache_fur_anchors)
        fsp = os.path.join(scene, "preprocess", "face_scores.json")
        if os.path.exists(fsp):
            fsc = json.load(open(fsp))
            ref = frames[max(ds.train_ids[i], key=lambda t: fsc.get(frames[t]["name"], 0.0))]
        else:
            ref = frames[0]
        rgb_r, mask_r, _, _ = _load_rgb_mask(scene, ref, 8)
        label = _label_grid(scene, ref, mask_r)[None].to(dev)
        inp = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(dev),
                            (518, 518), mode="bilinear", align_corners=False)
        canon = ds.canon[i][None].to(dev)
        anc = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in ds.anc[i].items()}
        rgb4, _, _, _ = _load_rgb_mask(scene, ref, 4)
        anc["ref_rgb"] = torch.from_numpy(rgb4).permute(2, 0, 1)[None].to(dev)
        anc["ref_K"] = intrinsics(ref["fx"] / 4, ref["fy"] / 4, ref["cx"] / 4, ref["cy"] / 4, dev)[None]
        anc["ref_c2w"] = torch.tensor(ref["c2w"], device=dev).float()[None]
        if args.face_crop:
            fcrop = face_crop_518(scene, ref, torch.from_numpy(rgb4).permute(2, 0, 1))
            if fcrop is not None:
                anc["face_crop"] = fcrop[None].to(dev)
        ps, ious, lps, fps_ = [], [], [], []
        with torch.no_grad():
            fur, body = model(inp, label, canon, anc, sub)
            f0, b0 = fur[0], body[0]
            full = {k: torch.cat([b0[k], f0[k]]) for k in
                    ("means", "quats", "scales", "opacities", "sh")}
            for vi in held:
                fr = frames[vi]
                rgb, mask, W, H = _load_rgb_mask(scene, fr, s)
                gt = torch.from_numpy(rgb).to(dev)
                m = torch.from_numpy(mask).to(dev)
                K = intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, dev)
                pred, alpha = render_gaussians(full["means"], full["quats"], full["scales"],
                                               full["opacities"], full["sh"],
                                               torch.tensor(fr["c2w"], device=dev).float(),
                                               K, W, H, bg=white, sh_degree=1)
                if m.sum() < 10:                                    # dog out of view
                    continue
                if args.vis and vi == held[0]:
                    from PIL import Image
                    pair = np.concatenate([(pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                           (rgb * 255).astype(np.uint8)], 1)
                    Image.fromarray(pair).save(args.vis)
                comp = (pred - (1 - alpha) * white) + gt * (1 - alpha)
                mse = (((comp - gt) ** 2) * m).sum() / (m.sum() * 3)
                ps.append(float(10 * torch.log10(1 / mse.clamp_min(1e-10))))
                am, gm = alpha[:, :, 0] > 0.5, m[:, :, 0] > 0.5
                ious.append(float((am & gm).sum()) / max(float((am | gm).sum()), 1))
                c2 = F.interpolate(comp.permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                g2 = F.interpolate(gt.permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                lps.append(float(lpips_fn(c2, g2).mean()))
                # face-region PSNR (s8 region mask upsampled; skip tiny/absent faces)
                fp = os.path.join(scene, "preprocess", "region_masks_s8",
                                  os.path.splitext(fr["name"])[0] + ".png")
                if os.path.exists(fp):
                    from PIL import Image
                    rch = np.asarray(Image.open(fp))[:, :, 0]
                    fm = torch.from_numpy((rch > 76).astype(np.float32))[None, None].to(dev)
                    fm = F.interpolate(fm, size=(H, W), mode="nearest")[0, 0, :, :, None] * m
                    if fm.sum() >= 100:
                        msef = (((comp - gt) ** 2) * fm).sum() / (fm.sum() * 3)
                        fps_.append(float(10 * torch.log10(1 / msef.clamp_min(1e-10))))
        dog = scene.split("/")[-2]
        fpsnr = np.mean(fps_) if fps_ else float("nan")
        rows.append((dog, np.mean(ps), np.mean(ious), np.mean(lps), fpsnr))
        print(f"[{i+1}/{len(scenes)}] {dog}: comp {np.mean(ps):.2f}dB iou {np.mean(ious):.3f} "
              f"lpips {np.mean(lps):.4f} face {fpsnr:.2f}dB", flush=True)

    with open(args.out_csv, "w") as f:
        f.write("dog,comp_psnr,iou,lpips,face_psnr\n")
        for dog, p, io, lp, fpn in rows:
            f.write(f"{dog},{p:.3f},{io:.4f},{lp:.4f},{fpn:.3f}\n")
    print(f"== mean comp {np.mean([r[1] for r in rows]):.2f}dB "
          f"iou {np.mean([r[2] for r in rows]):.3f} "
          f"lpips {np.mean([r[3] for r in rows]):.4f} "
          f"face {np.nanmean([r[4] for r in rows]):.2f}dB (n={len(rows)}) -> {args.out_csv}")


if __name__ == "__main__":
    main()
