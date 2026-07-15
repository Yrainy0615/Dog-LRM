#!/usr/bin/env python3
"""Render the v9 feed-forward fur model from FRONT views (top face-score frames) for a dog.
Output: [v9 render | GT] for each of the top-N front frames. Reuses render_v9_nofur loading."""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
from dog_lrm.model_fur import DogLRMFurV9, load_fur_ckpt
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_dog_lrm_decomp import _label_grid
from train_dog_lrm_fur_v2 import FurScenes, list_scenes

ap = argparse.ArgumentParser()
ap.add_argument("--dog", default="00031-itsuki")
ap.add_argument("--ckpt", default="exps/dog_lrm_fur_v9/model.pt")
ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
ap.add_argument("--nofur_thr", type=float, default=0.25)
ap.add_argument("--n", type=int, default=2, help="top-N front views")
ap.add_argument("--out", default="exps/v9_front")
args = ap.parse_args(); dev = "cuda"; os.makedirs(args.out, exist_ok=True)

scenes = [s for s in list_scenes(args.root) if args.dog in s]
ds = FurScenes(scenes, 4, 1, 26000)
da = np.load(os.path.join(scenes[0], "preprocess", "dsmal_anchors.npz"))
w_face = torch.from_numpy(da["w_face"]); faces0 = torch.from_numpy(da["faces"]).long()
subdiv_M = build_subdiv(faces0, 1, dev)
sub = lambda x: torch.stack([torch.sparse.mm(subdiv_M, x[b]) for b in range(x.shape[0])])
w_face_s = torch.sparse.mm(subdiv_M, w_face[:, None].float().to(dev))[:, 0].cpu()
model = DogLRMFurV9(w_face, faces_sub=subdivided_faces(faces0, 1), w_face_s=w_face_s,
                    K=11, fur_op=0.7, radius_frac=0.0032, dim=768, n_layers=12, n_heads=12,
                    tri_res=64, tri_ch=32, splat_res=128, splat_base_sc=0.004, splat_dres=0.05).to(dev)
load_fur_ckpt(model, args.ckpt, dev); model.eval()
white = torch.ones(3, device=dev); s = 4
scene = scenes[0]; frames = ds.frames[0]

# reference frame (most front-facing) for the image conditioning
fsc = json.load(open(os.path.join(scene, "preprocess", "face_scores.json")))
ref = frames[max(ds.train_ids[0], key=lambda t: fsc.get(frames[t]["name"], 0.0))]
rgb_r, mask_r, _, _ = _load_rgb_mask(scene, ref, 8)
label = _label_grid(scene, ref, mask_r)[None].to(dev)
inp = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(dev), (518, 518), mode="bilinear", align_corners=False)
canon = ds.canon[0][None].to(dev)
anc = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in ds.anc[0].items()}
rgb4, _, _, _ = _load_rgb_mask(scene, ref, 4)
anc["ref_rgb"] = torch.from_numpy(rgb4).permute(2, 0, 1)[None].to(dev)
anc["ref_K"] = intrinsics(ref["fx"]/4, ref["fy"]/4, ref["cx"]/4, ref["cy"]/4, dev)[None]
anc["ref_c2w"] = torch.tensor(ref["c2w"], device=dev).float()[None]

fa = np.load(os.path.join(scene, "preprocess", "fur_anchors.npz"))
wf = torch.from_numpy(fa["w_face"]).to(dev).float()
wh = torch.from_numpy(fa["w_head"]).to(dev).float() if "w_head" in fa else torch.zeros_like(wf)
anc["nofur"] = ((wf + wh) > args.nofur_thr).float()[None]

# top-N FRONT frames by face score
front = sorted(frames, key=lambda fr: fsc.get(fr["name"], 0.0), reverse=True)[:args.n]
print(f"[v9-front] {args.dog}: front frames {[f['name'] for f in front]}", flush=True)

def render_view(fr):
    rgb, mask, W, H = _load_rgb_mask(scene, fr, s)
    K = intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev); c2w = torch.tensor(fr["c2w"], device=dev).float()
    with torch.no_grad():
        fur, body = model(inp, label, canon, anc, sub)
        f0, b0 = fur[0], body[0]
        full = {k: torch.cat([b0[k], f0[k]]) for k in ("means", "quats", "scales", "opacities", "sh")}
        img = render_gaussians(full["means"], full["quats"], full["scales"], full["opacities"], full["sh"],
                               c2w, K, W, H, bg=white, sh_degree=1)[0].clamp(0, 1).cpu().numpy()
    m = mask[:, :, 0] > 0.5; ys, xs = np.where(m)
    cr = lambda x: x[max(ys.min()-15, 0):ys.max()+15, max(xs.min()-15, 0):xs.max()+15]
    a, g = cr(img), cr(rgb); h = min(a.shape[0], g.shape[0])
    return np.concatenate([a[:h], g[:h]], 1)

for i, fr in enumerate(front):
    out = (render_view(fr) * 255).astype(np.uint8)
    Image.fromarray(out).save(f"{args.out}/{args.dog}_front{i}.png")
    print(f"[v9-front] saved {args.dog}_front{i}.png [v9 | GT] ({fr['name']})", flush=True)
