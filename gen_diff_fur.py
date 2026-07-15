#!/usr/bin/env python3
"""Route A: diffusion AS the fur generator, teacher-free. For each dog:
image -> D-SMAL surface -> DINOv2 anchor conditioning -> diffusion DDIM -> strands
-> densify -> gaussian render + OBJ export. No v8 teacher used anywhere.

  python gen_diff_fur.py --dogs 00100-kokoa,00148-uta --views 2
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from PIL import Image
from transformers import AutoModel
from dog_lrm.render import render_gaussians
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from optimize_fur_dog import SmalRig, load_scene, scale_K, project_color
from train_fur_strand import quat_align_z
from train_fur_diff import DiT, Codec, build_geo, splat_depth, fps
from train_strand_predictor import vnormals, tbn_frames
from sample_fur_diff import predict, interp_latent, dense_roots, export_obj

dev = "cuda"
ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
NANCHOR = 4096
NDENSE = 80000
HELDOUT = ["00062-bear", "00089-huu", "00148-uta", "00188-willy", "00029-oto",
           "00100-kokoa", "00174-tete", "00215-teo", "00256-nagi"]


def fur_gauss(world, rgb, diag, rf=0.0006):
    p0, p1 = world[:, :-1].reshape(-1, 3), world[:, 1:].reshape(-1, 3)
    mid = (p0 + p1) / 2
    seg = p1 - p0
    sl = seg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    quat = quat_align_z(seg / sl)
    r = rf * diag
    scales = torch.cat([torch.full_like(sl, r), torch.full_like(sl, r), sl * 0.6], -1)
    col = rgb[:, None].expand(-1, world.shape[1] - 1, -1).reshape(-1, 3)
    return mid, quat, scales, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exps/fur_diff_v2/diff.pt")
    ap.add_argument("--codec", default="synth_fur/strand_codec.npz")
    ap.add_argument("--dogs", default=",".join(HELDOUT))
    ap.add_argument("--views", type=int, default=2, help="condition views (k=2 best per eval)")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--out", default="exps/fur_diff_v2/routeA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    codec = Codec(args.codec)
    ck = torch.load(args.ckpt, map_location=dev)
    A = ck["args"]
    net = DiT(zdim=codec.C, geo=ck["net"]["cgeo.weight"].shape[1], d=A["d"],
              depth=A["depth"], heads=max(A["d"] // 64, 4)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()

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

    white = torch.ones(3, device=dev)
    for dog in args.dogs.split(","):
        rig = SmalRig(dog, ROOT).to(dev)
        with torch.no_grad():
            base = rig.verts()
        da = np.load(os.path.join(ROOT, dog, "colmap", "preprocess", "dsmal_anchors.npz"))
        faces0 = torch.from_numpy(da["faces"].astype(np.int64))
        wface0 = torch.from_numpy(da["w_face"].astype(np.float32))
        M1 = build_subdiv(faces0, 1, "cpu").to(dev)
        verts = torch.sparse.mm(M1, base)
        faces = subdivided_faces(faces0, 1).to(dev)
        wface = torch.sparse.mm(M1, wface0.to(dev)[:, None])[:, 0]      # no-fur on face/muzzle
        vn = vnormals(verts, faces)
        TBN = tbn_frames(verts, vn)
        diag = float((verts.max(0).values - verts.min(0).values).norm())

        aidx = fps(verts, NANCHOR)
        R, Rn, Rtbn = verts[aidx], vn[aidx], TBN[aidx]

        imgs, masks, K0, c2ws, H, W = load_scene(ROOT, dog)
        full = np.asarray(Image.open(glob.glob(os.path.join(ROOT, dog, "colmap/preprocess/masks/*"))[0]))
        Ks = scale_K(K0, H, W, full.shape[0], full.shape[1]).to(dev)
        c2ws = c2ws.to(dev)

        # conditioning from k evenly-spaced views (pixel-aligned DINO + gsplat visibility)
        vids = np.linspace(0, len(imgs) - 1, args.views).astype(int)
        feats, viss, clss = [], [], []
        for vi in vids:
            fm, cls = dfeat(imgs[vi])
            w2c = torch.inverse(c2ws[vi])
            cam = (w2c[:3, :3] @ R.T + w2c[:3, 3:4]).T
            zc = cam[:, 2].clamp(min=1e-4)
            uv = (Ks[vi] @ (cam / zc[:, None]).T).T[:, :2]
            gu = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)[None, :, None, :]
            f = F.grid_sample(fm, gu, align_corners=False)[0, :, :, 0].T
            dmap = splat_depth(verts, Ks[vi], c2ws[vi], max(H, W), diag, blender_cam=False)
            okuv = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (zc > 0)
            gsq = (uv / max(H, W) * 2 - 1).clamp(-1, 1)
            ds = F.grid_sample(dmap[None, None], gsq[None, :, None, :], align_corners=False)[0, 0, :, 0]
            vis = okuv & (zc <= ds + 0.02 * diag)
            feats.append(f); viss.append(vis.float()); clss.append(cls)
        v = torch.stack(viss); f = torch.stack(feats)
        w = v[:, :, None]
        feat = (f * w).sum(0) / w.sum(0).clamp(min=1)
        visfrac = v.max(0).values
        feat = feat * visfrac[:, None]
        geo = build_geo(R, Rn, diag, visfrac)
        gl = torch.stack(clss).mean(0)
        zs = predict(net, (feat, geo, gl), codec.C, A)

        # densify: fur everywhere except face/muzzle
        lgeo_v = (wface < 0.5).float()
        pos, nrm, ptbn = dense_roots(verts, faces, lgeo_v, NDENSE, seed=0)
        dloc = codec.decode(interp_latent(zs, R, pos))
        fur_world = pos[:, None] + torch.einsum("rij,rkj->rki", ptbn, dloc * diag)
        drgb = project_color(pos, imgs, masks, Ks, c2ws)

        # per-strand clump+individual tone jitter (teacher's trick, optimize_fur_dog.py:120-122)
        # a flat root-colour projected from photos reads as a painted blob, not fibres -- break
        # it up so neighbouring strands are visibly distinguishable.
        g = torch.Generator(device="cpu").manual_seed(0)
        nseed = 2500
        seed_idx = torch.randperm(pos.shape[0], generator=g)[:nseed].to(dev)
        gid = torch.cdist(pos, pos[seed_idx]).argmin(1)
        tone_clump = (1 + 0.15 * torch.randn(nseed, 1, device=dev)).clamp(0.7, 1.3)
        tone_strand = (1 + 0.06 * torch.randn(pos.shape[0], 1, device=dev)).clamp(0.85, 1.15)
        drgb = (drgb * tone_clump[gid] * tone_strand).clamp(0, 1)

        # body skin gaussians (subdiv-2 for density), image-coloured
        M2 = build_subdiv(faces0, 2, "cpu").to(dev)
        bverts = torch.sparse.mm(M2, base)
        faces2 = subdivided_faces(faces0, 2).to(dev)
        bnrm = vnormals(bverts, faces2)
        bcol = project_color(bverts, imgs, masks, Ks, c2ws)
        bq = torch.zeros(bverts.shape[0], 4, device=dev); bq[:, 0] = 1
        bs = torch.full((bverts.shape[0], 3), 0.004 * diag, device=dev)
        bo = torch.ones(bverts.shape[0], device=dev)

        mid, quat, scales, col = fur_gauss(fur_world, drgb, diag)
        means = torch.cat([mid, bverts]); quats = torch.cat([quat, bq])
        scl = torch.cat([scales, bs]); cols = torch.cat([col, bcol])
        ops = torch.cat([torch.full((mid.shape[0],), 0.9, device=dev), bo])
        pt_nrm = torch.cat([nrm[:, None].expand(-1, fur_world.shape[1] - 1, -1).reshape(-1, 3), bnrm])

        for vi in np.linspace(0, len(imgs) - 1, 4).astype(int):
            # headlamp-style shading (matches the ring-light rig): light ~ toward camera per point
            light = F.normalize(c2ws[vi][:3, 3][None] - means, dim=-1)
            shade = (0.55 + 0.45 * (pt_nrm * light).sum(-1).clamp(min=0))[:, None]
            rgb, _ = render_gaussians(means, quats, scl, ops, (cols * shade).clamp(0, 1),
                                       c2ws[vi], Ks[vi], W, H, bg=white)
            im = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(im).save(os.path.join(args.out, f"{dog}_v{vi}.png"))
        export_obj(os.path.join(args.out, f"{dog}_fur.obj"), fur_world[::5].cpu().numpy())
        print(f"[routeA] {dog} done | {NDENSE} strands | vis@k={args.views} {float(visfrac.mean()):.2f}", flush=True)
    print("[routeA] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
