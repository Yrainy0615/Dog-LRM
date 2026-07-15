#!/usr/bin/env python3
"""Bake per-dog STUDENT training data from the fur-teacher results.

Per dog: rebuild the refined D-SMAL surface from teacher.pt['rig'] (SmalRig deltas),
FPS-subsample teacher roots -> anchor tokens, encode each anchor's strand with the
synth-fitted codec (root-TBN local), and cache per-view conditioning: pixel-aligned
DINOv2 features + vertex-splat depth visibility + CLS. Output: one npz per dog that
train_fur_diff can consume alongside the synthetic grooms.

  python prep_student_data.py --views 24 --out synth_fur/student_data
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from PIL import Image
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from optimize_fur_dog import SmalRig, load_scene, scale_K
from train_fur_diff import Codec, splat_depth, anchor_visibility, fps
from train_strand_predictor import vnormals, tbn_frames

dev = "cuda"


def resample_polyline(pts, K):
    """arc-length linear resample [N,K0,3] -> [N,K,3] (teacher K=10 vs codec K=12)"""
    seg = torch.diff(pts, dim=1).norm(dim=-1)                          # [N,K0-1]
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)  # [N,K0]
    total = cum[:, -1:].clamp(min=1e-9)
    t = cum / total                                                    # [N,K0] in [0,1]
    ts = torch.linspace(0, 1, K, device=pts.device)[None].expand(pts.shape[0], -1)
    idx = torch.searchsorted(t.contiguous(), ts.contiguous(), right=True).clamp(1, pts.shape[1] - 1)
    t0 = torch.gather(t, 1, idx - 1); t1 = torch.gather(t, 1, idx)
    w = ((ts - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0, 1)[..., None]
    p0 = torch.gather(pts, 1, (idx - 1)[..., None].expand(-1, -1, 3))
    p1 = torch.gather(pts, 1, idx[..., None].expand(-1, -1, 3))
    return p0 + w * (p1 - p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--teacher", default="exps/fur_teacher")
    ap.add_argument("--codec", default="synth_fur/strand_codec.npz")
    ap.add_argument("--nanchor", type=int, default=4096)
    ap.add_argument("--views", type=int, default=24, help="views cached per dog (evenly spaced)")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--out", default="synth_fur/student_data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    codec = Codec(args.codec)
    from transformers import AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    nsd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    ph = args.res // 14

    @torch.no_grad()
    def dfeat(img):
        x = img.to(dev).permute(2, 0, 1)[None].float()
        x = (F.interpolate(x, (args.res, args.res), mode="bilinear", align_corners=False) - nm) / nsd
        o = dino(pixel_values=x).last_hidden_state
        return o[:, 1:].transpose(1, 2).reshape(1, -1, ph, ph), o[0, 0]

    for tp in sorted(glob.glob(os.path.join(args.teacher, "*", "teacher.pt"))):
        dog = os.path.basename(os.path.dirname(tp))
        op = os.path.join(args.out, f"{dog}.npz")
        if os.path.exists(op):
            print(f"[skip] {dog}", flush=True)
            continue
        T = torch.load(tp, map_location=dev)
        strands, roots = T["strands"].to(dev), T["roots"].to(dev)

        # refined surface from rig deltas -> subdiv-1 verts + normals (anchor TBN frames)
        rig = SmalRig(dog, args.root).to(dev)
        if T.get("rig"):
            with torch.no_grad():
                for k, v in T["rig"].items():
                    if hasattr(rig, k):                                # skip seg_dalb/body_* (not rig geo params)
                        getattr(rig, k).copy_(v.to(dev))
        with torch.no_grad():
            base = rig.verts()
        da = np.load(os.path.join(args.root, dog, "colmap", "preprocess", "dsmal_anchors.npz"))
        faces0 = torch.from_numpy(da["faces"].astype(np.int64))
        M = build_subdiv(faces0, 1, "cpu").to(dev)
        verts = torch.sparse.mm(M, base)
        faces = subdivided_faces(faces0, 1).to(dev)
        vn = vnormals(verts, faces)
        TBN = tbn_frames(verts, vn)
        diag = float((verts.max(0).values - verts.min(0).values).norm())

        aidx = fps(roots, args.nanchor)
        R = roots[aidx]
        nnv = torch.cdist(R, verts).argmin(1)
        Rtbn, Rn = TBN[nnv], vn[nnv]
        S = resample_polyline(strands[aidx], codec.K)                  # teacher K=10 -> codec K=12
        loc = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), S - R[:, None]) / diag
        z = codec.encode(loc)                                          # [nanchor, C]
        rec_err = float((codec.decode(z) - loc).norm(dim=-1).mean())

        imgs, masks, K0, c2ws, H, W = load_scene(args.root, dog)
        full = np.asarray(Image.open(glob.glob(os.path.join(args.root, dog, "colmap/preprocess/masks/*"))[0]))
        Ks = scale_K(K0, H, W, full.shape[0], full.shape[1]).to(dev)
        c2ws = c2ws.to(dev)
        vids = np.linspace(0, len(imgs) - 1, args.views).astype(int)
        feats, viss, clss = [], [], []
        for vi in vids:
            img = imgs[vi]
            fm, cls = dfeat(img)
            w2c = torch.inverse(c2ws[vi])
            cam = (w2c[:3, :3] @ R.T + w2c[:3, 3:4]).T
            zc = cam[:, 2].clamp(min=1e-4)
            uv = (Ks[vi] @ (cam / zc[:, None]).T).T[:, :2]
            gu = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)[None, :, None, :]
            f = F.grid_sample(fm, gu, align_corners=False)[0, :, :, 0].T
            dmap = splat_depth(verts, Ks[vi], c2ws[vi], max(H, W), diag, blender_cam=False)
            # splat_depth renders square; re-project with square-padded convention:
            vis_uv_ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (zc > 0)
            gu_sq = (uv / max(H, W) * 2 - 1).clamp(-1, 1)
            dsamp = F.grid_sample(dmap[None, None], gu_sq[None, :, None, :], align_corners=False)[0, 0, :, 0]
            vis = vis_uv_ok & (zc <= dsamp + 0.02 * diag)
            feats.append(f.half().cpu()); viss.append(vis.cpu()); clss.append(cls.half().cpu())
        np.savez(op,
                 z=z.cpu().numpy().astype(np.float32),
                 anchors=R.cpu().numpy().astype(np.float32),
                 normals=Rn.cpu().numpy().astype(np.float32),
                 albedo=T["albedo"][aidx].cpu().numpy().astype(np.float32),
                 diag=diag, rec_err=rec_err,
                 feats=torch.stack(feats).numpy(),                     # [V,nanchor,1024] fp16
                 vis=torch.stack(viss).numpy(),
                 cls=torch.stack(clss).numpy(),
                 view_ids=vids)
        print(f"[prep] {dog}: codec recon {rec_err:.4f} xdiag | vis/view "
              f"{torch.stack(viss).float().mean(1).mul(100).int().tolist()[:6]}...", flush=True)
    print("[prep] DONE", flush=True)


if __name__ == "__main__":
    main()
