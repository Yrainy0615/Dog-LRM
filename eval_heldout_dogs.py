#!/usr/bin/env python3
"""v6 stage-0: held-out DOG evaluation -- the only meter that can show a prior's
benefit on occluded/unseen regions (FUR_V6_PLAN s2.5/s4.6).

A model trained WITHOUT the test dogs is fed ONE reference image of an unseen dog
and must reconstruct it feed-forward. Because the whole dog is held out, EVERY
other captured view is genuinely unseen -- the non-reference views are exactly the
"occluded-from-ref side" that multi-view-of-a-trained-dog eval cannot probe. We
report comp-PSNR / IoU / LPIPS / face-PSNR over the non-ref views, and dump a novel
orbit montage (front/side/back/top) per dog for visual inspection of phantom ear
fur / occluded-leg quality.

NOTE: run on r5 (trained on all 69) it is only a HARNESS smoke -- r5 saw these dogs,
so the numbers are not the real held-out signal. The real signal needs a model
trained on the split's 60-dog train set (the upcoming v6 baseline).

  PATH=$ENV/bin:$PATH python eval_heldout_dogs.py --ckpt exps/dog_lrm_fur_v2_r5/model.pt \
      --split v6_heldout_split.json --vis_dir exps/v6_heldout/r5
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
from dog_lrm.motion import look_at
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_dog_lrm_decomp import _label_grid
from train_dog_lrm_fur_v2 import FurScenes, list_scenes, _face_box, face_crop_518


def orbit_montage(full, ctr, diag, dev, res=400, azims=(0, 90, 180, 270), elev=-8.0):
    """Render the fitted-pose gaussians from novel azimuths -> [res, res*len, 3] uint8.
    No GT (purely visual): surfaces the occluded-from-ref side (ear fur, legs)."""
    white = torch.ones(3, device=dev)
    K = intrinsics(res * 1.15, res * 1.15, res / 2, res / 2, dev)
    panes = []
    for az in azims:
        c2w = look_at(ctr, float(az), elev, 1.5 * diag, dev)
        img, _ = render_gaussians(full["means"], full["quats"], full["scales"],
                                  full["opacities"], full["sh"], c2w, K, res, res,
                                  bg=white, sh_degree=1)
        panes.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate(panes, 1)


def _kparts(K):
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def ref_occluded_mask(depth_e, c2w_e, K_e, depth_r, alpha_r, c2w_r, K_r, tol):
    """Per-pixel mask [H,W] of the eval view: 1 where the surface seen by that pixel was
    NOT visible in the reference image (projects off-frame, onto ref background, or behind
    the ref's first surface). This is the genuinely-occluded-from-ref region where a prior
    has to hallucinate -- the only place over-fluff is penalised instead of rewarded."""
    H, W = depth_e.shape[:2]
    dev = depth_e.device
    fxe, fye, cxe, cye = _kparts(K_e)
    vv, uu = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    d = depth_e[..., 0]
    cam = torch.stack([(uu - cxe) / fxe * d, (vv - cye) / fye * d, d], -1)   # [H,W,3] eval cam
    world = cam @ c2w_e[:3, :3].T + c2w_e[:3, 3]                             # -> world
    w2c_r = torch.inverse(c2w_r)
    camr = world @ w2c_r[:3, :3].T + w2c_r[:3, 3]                            # -> ref cam
    zr = camr[..., 2]
    fxr, fyr, cxr, cyr = _kparts(K_r)
    ur = camr[..., 0] / zr.clamp_min(1e-6) * fxr + cxr
    vr = camr[..., 1] / zr.clamp_min(1e-6) * fyr + cyr
    Hr, Wr = depth_r.shape[:2]
    inside = (zr > 1e-4) & (ur >= 0) & (ur <= Wr - 1) & (vr >= 0) & (vr <= Hr - 1)
    gx = (ur / (Wr - 1)) * 2 - 1
    gy = (vr / (Hr - 1)) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]                                   # [1,H,W,2]
    dr = F.grid_sample(depth_r.permute(2, 0, 1)[None], grid, mode="nearest",
                       align_corners=True)[0, 0]
    ar = F.grid_sample(alpha_r.permute(2, 0, 1)[None], grid, mode="nearest",
                       align_corners=True)[0, 0]
    ref_saw = inside & (ar > 0.5) & (zr <= dr + tol)                         # visible in ref
    return (~ref_saw)                                                        # occluded-from-ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="v6_heldout_split.json")
    ap.add_argument("--which", default="test", choices=["test", "train"],
                    help="which split half to evaluate (default: the held-out test dogs)")
    ap.add_argument("--n_root", type=int, default=26000)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--fur_op", type=float, default=0.7)
    ap.add_argument("--radius_frac", type=float, default=0.0032)
    ap.add_argument("--scale_div", type=int, default=4)
    ap.add_argument("--view_stride", type=int, default=3,
                    help="eval every Nth non-ref view (bounds cost)")
    ap.add_argument("--face_crop", action="store_true")
    ap.add_argument("--density_npz", default=None,
                    help="canonical per-vertex fur density (blender_fur_groom output); "
                         "sampled at each dog's roots and applied as a fur-opacity gate")
    ap.add_argument("--vis_dir", default=None, help="save per-dog orbit montage here")
    ap.add_argument("--out_csv", default="exps/eval_heldout.csv")
    ap.add_argument("--v6_repr", action="store_true",
                    help="apply v6 coat-adaptive representation (FUR_V6_PLAN s8.2): global-short "
                         "+ paw-short + offset-shell; fed as fixed anc knobs (length not optimized)")
    ap.add_argument("--len_short", type=float, default=0.5, help="v6 global length factor")
    ap.add_argument("--paw_len", type=float, default=0.3, help="v6 length factor on paw-mask verts")
    ap.add_argument("--offset_shell", type=float, default=0.2, help="v6 strand-origin lift (x L)")
    ap.add_argument("--fur_aspect", type=float, default=0.0,
                    help="cap fur-gaussian cross-section to half-seg/aspect (thin strands); 0=off")
    ap.add_argument("--paw_mask", default="synth_fur/paw_mask.npy")
    ap.add_argument("--no_occ", action="store_true",
                    help="disable the ref-occluded-region metric (default: on)")
    ap.add_argument("--occ_tol", type=float, default=0.02,
                    help="depth-test tolerance as fraction of body diag")
    args = ap.parse_args()
    do_occ = not args.no_occ
    dev = "cuda"

    split = json.load(open(args.split))
    want = set(split[args.which])
    scenes = [s for s in list_scenes(args.root) if s.split("/")[-2] in want]
    assert scenes, f"no scenes matched {args.which} set of {args.split}"
    print(f"eval {args.which}: {len(scenes)}/{len(want)} dogs found", flush=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

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
    white = torch.ones(3, device=dev)
    s = args.scale_div
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
    for p_ in lpips_fn.parameters():
        p_.requires_grad = False

    dens_canon = None
    if args.density_npz:                                            # canonical per-vertex density
        dens_canon = torch.from_numpy(np.load(args.density_npz)["density"]).float().to(dev)
        print(f"[density] gating fur opacity with {args.density_npz} "
              f"(mean {float(dens_canon.mean()):.3f})", flush=True)

    paw_short_v = None
    if args.v6_repr:                                                # per-vertex paw length factor
        paw = torch.from_numpy(np.load(args.paw_mask)).float().to(dev)  # [Vs] 1=paw
        paw_short_v = 1.0 - (1.0 - args.paw_len) * paw              # paw_len on paws, 1 elsewhere
        print(f"[v6_repr] len_short={args.len_short} paw_len={args.paw_len} "
              f"offset_shell={args.offset_shell}", flush=True)

    rows = []
    for i, scene in enumerate(scenes):
        frames = ds.frames[i]
        # ref = best face-visibility view (same convention as eval_fur_ff)
        fsp = os.path.join(scene, "preprocess", "face_scores.json")
        if os.path.exists(fsp):
            fsc = json.load(open(fsp))
            ref_id = max(ds.train_ids[i], key=lambda t: fsc.get(frames[t]["name"], 0.0))
        else:
            ref_id = 0
        ref = frames[ref_id]
        # held-out-dog eval views: ALL views except the ref, subsampled
        eval_ids = [v for v in range(len(frames)) if v != ref_id][::args.view_stride]

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
        if args.v6_repr:
            anc["len_short"] = args.len_short
            anc["paw_short"] = paw_short_v[None]
            anc["offset_shell"] = args.offset_shell
        if args.fur_aspect > 0:
            anc["fur_aspect"] = args.fur_aspect

        ps, ious, lps, fps_ = [], [], [], []
        prs, rcs = [], []                                           # silhouette precision / recall
        hpred, hgt = [], []                                         # halo: coat area beyond bald body
        ps_o, lps_o, occf = [], [], []                              # ref-occluded-region metrics
        with torch.no_grad():
            fur, body = model(inp, label, canon, anc, sub)
            f0, b0 = fur[0], body[0]
            if dens_canon is not None:                              # canonical density -> per-root gate
                fsub = model.faces_sub.to(dev)
                droot = (dens_canon[fsub[anc["root_face"][0]]] * anc["root_bary"][0]).sum(-1).clamp(0, 1)
                f0 = {**f0, "opacities": f0["opacities"] * droot.repeat_interleave(model.Kp - 1)}
            full = {k: torch.cat([b0[k], f0[k]]) for k in
                    ("means", "quats", "scales", "opacities", "sh")}
            if do_occ:                                              # ref-view depth/alpha for occlusion test
                diag = float((b0["means"].max(0).values - b0["means"].min(0).values).norm())
                tol = args.occ_tol * diag
                _, _, Wr, Hr = _load_rgb_mask(scene, ref, s)
                c2w_ref = torch.tensor(ref["c2w"], device=dev).float()
                K_ref = intrinsics(ref["fx"] / s, ref["fy"] / s, ref["cx"] / s, ref["cy"] / s, dev)
                _, alpha_ref, depth_ref = render_gaussians(
                    full["means"], full["quats"], full["scales"], full["opacities"], full["sh"],
                    c2w_ref, K_ref, Wr, Hr, bg=None, sh_degree=1, return_depth=True)
            for vi in eval_ids:
                fr = frames[vi]
                rgb, mask, W, H = _load_rgb_mask(scene, fr, s)
                gt = torch.from_numpy(rgb).to(dev)
                m = torch.from_numpy(mask).to(dev)
                if m.sum() < 10:
                    continue
                K = intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, dev)
                c2w_e = torch.tensor(fr["c2w"], device=dev).float()
                pred, alpha, depth_e = render_gaussians(
                    full["means"], full["quats"], full["scales"], full["opacities"], full["sh"],
                    c2w_e, K, W, H, bg=white, sh_degree=1, return_depth=True)
                comp = (pred - (1 - alpha) * white) + gt * (1 - alpha)
                mse = (((comp - gt) ** 2) * m).sum() / (m.sum() * 3)
                ps.append(float(10 * torch.log10(1 / mse.clamp_min(1e-10))))
                if do_occ:                                          # restrict error to ref-occluded pixels
                    occ = ref_occluded_mask(depth_e, c2w_e, K, depth_ref, alpha_ref,
                                            c2w_ref, K_ref, tol)            # [H,W] bool
                    occ_gt = (occ & (m[:, :, 0] > 0.5)).float()[:, :, None]
                    npx = float(occ_gt.sum())
                    if npx >= 200:
                        occf.append(npx / max(float((m[:, :, 0] > 0.5).sum()), 1))
                        mse_o = (((comp - gt) ** 2) * occ_gt).sum() / (npx * 3)
                        ps_o.append(float(10 * torch.log10(1 / mse_o.clamp_min(1e-10))))
                        co = comp * occ_gt + white * (1 - occ_gt)
                        go = gt * occ_gt + white * (1 - occ_gt)
                        c2o = F.interpolate(co.permute(2, 0, 1)[None] * 2 - 1, 256,
                                            mode="bilinear", align_corners=False)
                        g2o = F.interpolate(go.permute(2, 0, 1)[None] * 2 - 1, 256,
                                            mode="bilinear", align_corners=False)
                        lps_o.append(float(lpips_fn(c2o, g2o).mean()))
                am, gm = alpha[:, :, 0] > 0.5, m[:, :, 0] > 0.5
                inter = float((am & gm).sum())
                ious.append(inter / max(float((am | gm).sum()), 1))
                prs.append(inter / max(float(am.sum()), 1))         # precision: fur outside GT -> over-fluff
                rcs.append(inter / max(float(gm.sum()), 1))         # recall: missing fur -> under-coat
                _, ab = render_gaussians(b0["means"], b0["quats"], b0["scales"], b0["opacities"],
                                         b0["sh"], c2w_e, K, W, H, bg=None, sh_degree=1)  # bald body
                a_body = max(float((ab[:, :, 0] > 0.5).sum()), 1.0)
                hpred.append((float(am.sum()) - a_body) / a_body)   # predicted coat beyond body
                hgt.append((float(gm.sum()) - a_body) / a_body)     # real coat beyond body (target)
                c2 = F.interpolate(comp.permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                g2 = F.interpolate(gt.permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                lps.append(float(lpips_fn(c2, g2).mean()))
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
            if args.vis_dir:                                        # novel-view occluded-side check
                ctr = b0["means"].mean(0)
                diag = float((b0["means"].max(0).values - b0["means"].min(0).values).norm())
                mont = orbit_montage(full, ctr, diag, dev)
                from PIL import Image
                Image.fromarray(mont).save(os.path.join(args.vis_dir, f"{scene.split('/')[-2]}_orbit.png"))

        dog = scene.split("/")[-2]
        fpsnr = np.mean(fps_) if fps_ else float("nan")
        co_p = np.mean(ps_o) if ps_o else float("nan")
        co_l = np.mean(lps_o) if lps_o else float("nan")
        of = np.mean(occf) if occf else float("nan")
        pr, rc = np.mean(prs), np.mean(rcs)
        hp, hgm = np.mean(hpred), np.mean(hgt)
        rows.append((dog, np.mean(ps), np.mean(ious), np.mean(lps), fpsnr, co_p, co_l, of, pr, rc, hp, hgm))
        print(f"[{i+1}/{len(scenes)}] {dog}: comp {np.mean(ps):.2f}dB iou {np.mean(ious):.3f} "
              f"P {pr:.3f} R {rc:.3f} halo {hp:.3f}/gt {hgm:.3f}(err {hp-hgm:+.3f}) "
              f"lpips {np.mean(lps):.4f} | OCC comp {co_p:.2f}dB frac {of:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("dog,comp_psnr,iou,lpips,face_psnr,comp_psnr_occ,lpips_occ,occ_frac,"
                "precision,recall,halo_pred,halo_gt\n")
        for dog, p, io, lp, fpn, cop, col, of, pr, rc, hp, hgm in rows:
            f.write(f"{dog},{p:.3f},{io:.4f},{lp:.4f},{fpn:.3f},{cop:.3f},{col:.4f},{of:.4f},"
                    f"{pr:.4f},{rc:.4f},{hp:.4f},{hgm:.4f}\n")
    mp = np.mean([r[10] for r in rows]); mg = np.mean([r[11] for r in rows])
    print(f"== {args.which} mean comp {np.mean([r[1] for r in rows]):.2f}dB "
          f"iou {np.mean([r[2] for r in rows]):.3f} "
          f"P {np.mean([r[8] for r in rows]):.3f} R {np.mean([r[9] for r in rows]):.3f} "
          f"halo {mp:.3f}/gt {mg:.3f} (err {mp-mg:+.3f}) "
          f"lpips {np.mean([r[3] for r in rows]):.4f} "
          f"face {np.nanmean([r[4] for r in rows]):.2f}dB | "
          f"OCC comp {np.nanmean([r[5] for r in rows]):.2f}dB "
          f"frac {np.nanmean([r[7] for r in rows]):.2f} (n={len(rows)}) -> {args.out_csv}")


if __name__ == "__main__":
    main()
