#!/usr/bin/env python3
"""NeuralFur-style export for v6 dogs at the fitted (GT) pose:
  - {dog}_full.ply : full Gaussian splat (body+fur, real colors)
  - {dog}_fur.ply  : fur-only Gaussian splat
  - {dog}_neuralfur.png : [ gray untextured SMAL mesh | brown fur | combined ] at a side view
SMAL body = pytorch3d gray mesh; fur = gsplat strands tinted uniform brown."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
from dog_lrm.model_fur import DogLRMFurV2, load_fur_ckpt
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from dog_lrm.motion import look_at
from train_dog_lrm_ddp import _load_rgb_mask
from train_dog_lrm_decomp import _label_grid
from train_dog_lrm_fur_v2 import FurScenes, list_scenes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (RasterizationSettings, MeshRenderer, MeshRasterizer,
                                 SoftPhongShader, PointLights, Materials, TexturesVertex)
from pytorch3d.utils import cameras_from_opencv_projection

ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
CKPT = "exps/dog_lrm_fur_v6/model.pt"
DOGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["00062-bear", "00148-uta", "00174-tete"]
OUT = "exps/v6_heldout/neuralfur"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"; s = 4; RES = 900
BROWN = torch.tensor([0.46, 0.30, 0.17], device=dev)
GRAY = (0.72, 0.72, 0.72)

scenes = [sc for sc in list_scenes(ROOT) if sc.split("/")[-2] in set(DOGS)]
ds = FurScenes(scenes, s, 1, 26000)
da = np.load(os.path.join(scenes[0], "preprocess", "dsmal_anchors.npz"))
w_face = torch.from_numpy(da["w_face"]); faces0 = torch.from_numpy(da["faces"]).long()
subM = build_subdiv(faces0, 1, dev)
sub = lambda x: torch.stack([torch.sparse.mm(subM, x[b]) for b in range(x.shape[0])])
w_face_s = torch.sparse.mm(subM, w_face[:, None].float().to(dev))[:, 0].cpu()
model = DogLRMFurV2(w_face, faces_sub=subdivided_faces(faces0, 1), w_face_s=w_face_s, K=11).to(dev)
load_fur_ckpt(model, CKPT, dev); model.eval()
faces_sub = model.faces_sub.to(dev)
white = torch.ones(3, device=dev)
paw = torch.from_numpy(np.load("synth_fur/paw_mask.npy")).float().to(dev)
paw_short = (1.0 - 0.7 * paw)[None]


def best_az(verts):
    c = verts.mean(0); diag = float((verts.max(0).values - verts.min(0).values).norm())
    best, ba = -1, 0.0
    for az in range(0, 360, 20):
        c2w = look_at(c, float(az), -8.0, 1.45 * diag, dev)
        cam = (torch.linalg.inv(c2w)[:3, :3] @ verts.T).T
        ext = cam.max(0).values - cam.min(0).values
        if float(ext[0] * ext[1]) > best:
            best, ba = float(ext[0] * ext[1]), float(az)
    return ba, c, diag


def render_mesh_gray(verts, faces, c2w, K):
    w2c = torch.linalg.inv(c2w)
    cam = cameras_from_opencv_projection(w2c[:3, :3][None], w2c[:3, 3][None], K[None],
                                         torch.tensor([[RES, RES]], device=dev))
    tex = TexturesVertex(verts_features=torch.tensor(GRAY, device=dev).expand(verts.shape[0], 3)[None])
    mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
    lights = PointLights(device=dev, location=[[0, -2, -2]],
                         ambient_color=[[0.55, 0.55, 0.55]], diffuse_color=[[0.55, 0.55, 0.55]],
                         specular_color=[[0.0, 0.0, 0.0]])
    rs = RasterizationSettings(image_size=RES, blur_radius=0.0, faces_per_pixel=1)
    rend = MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=rs),
                        SoftPhongShader(device=dev, cameras=cam, lights=lights,
                                        materials=Materials(device=dev, shininess=0.0)))
    img = rend(mesh)[0, :, :, :3]                                   # [H,W,3], white bg
    frag = MeshRasterizer(cameras=cam, raster_settings=rs)(mesh)
    alpha = (frag.pix_to_face[0, :, :, 0] >= 0).float()[:, :, None]
    return img.clamp(0, 1), alpha


def comp_white(img, alpha):
    return img * alpha + white * (1 - alpha)


for i, scene in enumerate(scenes):
    dog = scene.split("/")[-2]; frames = ds.frames[i]
    fsp = os.path.join(scene, "preprocess", "face_scores.json")
    fsc = json.load(open(fsp)) if os.path.exists(fsp) else {}
    ref_id = max(ds.train_ids[i], key=lambda t: fsc.get(frames[t]["name"], 0.0)) if fsc else 0
    ref = frames[ref_id]
    rgb_r, mask_r, _, _ = _load_rgb_mask(scene, ref, 8)
    label = _label_grid(scene, ref, mask_r)[None].to(dev)
    inp = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(dev), (518, 518),
                        mode="bilinear", align_corners=False)
    canon = ds.canon[i][None].to(dev)
    anc = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in ds.anc[i].items()}
    rgb4, _, _, _ = _load_rgb_mask(scene, ref, 4)
    anc["ref_rgb"] = torch.from_numpy(rgb4).permute(2, 0, 1)[None].to(dev)
    anc["ref_K"] = intrinsics(ref["fx"]/4, ref["fy"]/4, ref["cx"]/4, ref["cy"]/4, dev)[None]
    anc["ref_c2w"] = torch.tensor(ref["c2w"], device=dev).float()[None]
    anc["len_short"] = 0.5; anc["paw_short"] = paw_short; anc["offset_shell"] = 0.2
    with torch.no_grad():
        fur, body = model(inp, label, canon, anc, sub)
        f0, b0 = fur[0], body[0]
        # ---- GS .ply exports ----
        save_ply(f"{OUT}/{dog}_fur.ply", f0["means"], f0["scales"], f0["quats"], f0["opacities"], f0["rgb"])
        save_ply(f"{OUT}/{dog}_full.ply",
                 torch.cat([b0["means"], f0["means"]]), torch.cat([b0["scales"], f0["scales"]]),
                 torch.cat([b0["quats"], f0["quats"]]), torch.cat([b0["opacities"], f0["opacities"]]),
                 torch.cat([b0["rgb"], f0["rgb"]]))
        # ---- NeuralFur-style render ----
        verts = anc["roots"][0]                                     # [Vs,3] posed SMAL subdiv verts (GT pose)
        az, ctr, diag = best_az(verts)
        c2w = look_at(ctr, az, -8.0, 1.45 * diag, dev)
        K = intrinsics(RES * 1.2, RES * 1.2, RES / 2, RES / 2, dev)
        mesh_img, mesh_a = render_mesh_gray(verts, faces_sub, c2w, K)
        brown = BROWN.expand(f0["means"].shape[0], 3)
        fur_img, fur_a = render_gaussians(f0["means"], f0["quats"], f0["scales"], f0["opacities"],
                                          brown, c2w, K, RES, RES, bg=white, sh_degree=None)
        mesh_c = comp_white(mesh_img, mesh_a)
        fur_c = comp_white(fur_img - (1 - fur_a) * white, fur_a)    # fur already white-bg; recomp clean
        combined = fur_img * fur_a + mesh_c * (1 - fur_a)           # brown fur over gray mesh
        panes = [mesh_c, fur_c, combined.clamp(0, 1)]
        mont = torch.cat(panes, 1)
        Image.fromarray((mont.cpu().numpy() * 255).astype(np.uint8)).save(f"{OUT}/{dog}_neuralfur.png")
    print(f"{dog}: saved fur.ply ({f0['means'].shape[0]} g), full.ply, neuralfur.png (az={az:.0f})", flush=True)
