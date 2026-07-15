#!/usr/bin/env python3
"""Run BARC's pretrained model on dog crops -> initial SMAL params + overlay.

Produces per-crop {betas, betas_limbs, pose_rotmat, trans, flength} (BARC's
input-camera weak-perspective frame) and a red-silhouette overlay so you can eyeball
BARC's raw quality before deciding to fine-tune or use it fixed.

Requires the BARC checkpoints (registration, see SETUP.md):
  third_party/barc_release/checkpoint/barc_complete/model_best.pth.tar
  third_party/barc_release/checkpoint/cvpr_normflow_pret/rgbddog_v3_model.pt

Input : <scene>/preprocess/barc_crops/*.jpg   (from crop_dogs.py)
Output: <scene>/preprocess/barc_init/params.json, overlays/<name>.png

Deps: torch, pytorch3d, chumpy, BARC. Run on GPU.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

BARC = os.path.join(os.path.dirname(__file__), "..", "third_party", "barc_release")


class _StubRenderer(torch.nn.Module):
    """Replaces BARC's pytorch3d-version-locked SilhRenderer. All SMAL params are
    computed before the renderer call, so a dummy output is sufficient."""

    def __init__(self, image_size, *a, **k):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, int) else 256

    def forward(self, vertices, points, faces, focal_lengths, color=None):
        bs, n, s = vertices.shape[0], points.shape[1], self.image_size
        return (torch.zeros(bs, 1, s, s, device=vertices.device),
                torch.zeros(bs, n, 2, device=vertices.device))

    def get_torch_meshes(self, vertices, faces):
        return None


def build_model(device):
    sys.path.insert(0, os.path.abspath(os.path.join(BARC, "src")))
    import combined_model.model_shape_v7 as msv
    msv.SilhRenderer = _StubRenderer  # avoid the version-locked renderer at construction/forward
    from configs.barc_cfg_defaults import (get_cfg_defaults,
                                           update_cfg_global_with_yaml,
                                           get_cfg_global_updated)
    update_cfg_global_with_yaml(os.path.join(get_cfg_defaults().barc_dir, "src",
                                             "configs", "barc_cfg_visualization.yaml"))
    cfg = get_cfg_global_updated()
    p = cfg.params
    model = msv.ModelImageTo3d_withshape_withproj(
        num_stage_comb=p.NUM_STAGE_COMB, num_stage_heads=p.NUM_STAGE_HEADS,
        num_stage_heads_pose=p.NUM_STAGE_HEADS_POSE, trans_sep=p.TRANS_SEP,
        arch=p.ARCH, n_joints=p.N_JOINTS, n_classes=p.N_CLASSES,
        n_keyp=p.N_KEYP, n_bones=p.N_BONES, n_betas=p.N_BETAS, n_betas_limbs=p.N_BETAS_LIMBS,
        n_breeds=p.N_BREEDS, n_z=p.N_Z, image_size=p.IMG_SIZE,
        silh_no_tail=p.SILH_NO_TAIL, thr_keyp_sc=p.KP_THRESHOLD, add_z_to_3d_input=p.ADD_Z_TO_3D_INPUT,
        n_segbps=p.N_SEGBPS, add_segbps_to_3d_input=p.ADD_SEGBPS_TO_3D_INPUT,
        add_partseg=p.ADD_PARTSEG, n_partseg=p.N_PARTSEG,
        fix_flength=p.FIX_FLENGTH, structure_z_to_betas=p.STRUCTURE_Z_TO_B,
        structure_pose_net=p.STRUCTURE_POSE_NET, nf_version=p.NF_VERSION)
    ckpt = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, "barc_complete", "model_best.pth.tar")
    assert os.path.isfile(ckpt), f"missing {ckpt} -- download BARC checkpoints (see SETUP.md)"
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded BARC; missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval(), cfg


def make_norm_dict(device):
    from configs.data_info import COMPLETE_DATA_INFO_24 as di
    keys = ["pose_rot6d_mean", "trans_mean", "trans_std", "flength_mean", "flength_std"]
    return {k: torch.from_numpy(getattr(di, k)).float().to(device) for k in keys}


def barc_camera(flength, device, image_size=256):
    """BARC's fixed weak-perspective camera (differentiable_renderer.SilhRenderer)."""
    from pytorch3d.renderer import PerspectiveCameras
    bs = flength.shape[0]
    R = torch.eye(3, device=device)[None].repeat(bs, 1, 1)
    R[:, 0, 0] = -1
    R[:, 1, 1] = -1
    T = torch.zeros(bs, 3, device=device)
    pp = torch.tensor([[image_size / 2.0, image_size / 2.0]], device=device).repeat(bs, 1)
    imsz = torch.tensor([[image_size, image_size]], device=device).float().repeat(bs, 1)
    return PerspectiveCameras(device=device, in_ndc=False,
                              focal_length=flength.repeat(1, 2), principal_point=pp,
                              R=R, T=T, image_size=imsz)


@torch.no_grad()
def render_overlay(verts, faces, flength, device, image_size=256):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                    RasterizationSettings, SoftSilhouetteShader)
    cams = barc_camera(flength, device, image_size)
    blend = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(image_size=image_size,
                                   blur_radius=float(np.log(1.0 / 1e-4 - 1.0) * blend.sigma),
                                   faces_per_pixel=50)
    renderer = MeshRenderer(MeshRasterizer(cameras=cams, raster_settings=raster),
                            SoftSilhouetteShader(blend_params=blend))
    bs = verts.shape[0]
    meshes = Meshes(verts=verts, faces=faces[None].repeat(bs, 1, 1).long())
    return renderer(meshes, cameras=cams)[..., 3]  # [bs, H, W] alpha


RGB_MEAN = (0.4404, 0.4440, 0.4327)  # BARC color_normalize (subtract mean only)


def load_barc_crop(path):
    """Replicate ImgCrops preprocessing without BARC's Pillow-incompatible pilutil:
    center-pad to square, resize 256 bilinear, [0,1], subtract rgb_mean."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w != h:
        s = max(w, h)
        sq = Image.new("RGB", (s, s))
        sq.paste(im, ((s - w) // 2, (s - h) // 2))
        im = sq
    im = im.resize((256, 256), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)
    for c in range(3):
        t[c] -= RGB_MEAN[c]
    return t


def save_obj(path, verts, faces):
    """Write a posed SMAL mesh (verts in BARC camera frame, faces 0-indexed)."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def process_scene(scene_dir, out_subdir, model, cfg, norm_dict, device, batch_size):
    crops_dir = os.path.join(scene_dir, out_subdir, "barc_crops")
    if not os.path.isdir(crops_dir):
        raise FileNotFoundError(f"{crops_dir} not found (run crop_dogs.py first)")
    names = sorted(f for f in os.listdir(crops_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    out_dir = os.path.join(scene_dir, out_subdir, "barc_init")
    ov_dir = os.path.join(out_dir, "overlays")
    os.makedirs(ov_dir, exist_ok=True)
    faces = model.smal.faces.to(device)
    faces_np = model.smal.faces.cpu().numpy()

    results = {}
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        inp = torch.stack([load_barc_crop(os.path.join(crops_dir, n)) for n in batch]).to(device)
        with torch.no_grad():
            out, out_unnorm, out_reproj = model(inp, norm_dict=norm_dict)
        V = out_reproj["vertices_smal"]
        sil = render_overlay(V, faces, out_unnorm["flength"], device)
        for b, name in enumerate(batch):
            results[name] = {
                "betas": out_reproj["betas"][b].cpu().tolist(),
                "betas_limbs": out_reproj["betas_limbs"][b].cpu().tolist(),
                "pose_rotmat": out_unnorm["pose_rotmat"][b].cpu().tolist(),  # (35,3,3)
                "trans": out_unnorm["trans"][b].cpu().tolist(),
                "flength": float(out_unnorm["flength"][b].cpu().reshape(-1)[0]),
                # hourglass 2D keypoint detections (24), crop-256 frame; normalized [-1,1]
                # + per-keypoint confidence. Used as semantic supervision in fit_smal.
                "keypoints_norm": out["keypoints_norm"][b].cpu().tolist(),     # (24,2)
                "keypoints_scores": out["keypoints_scores"][b].cpu().reshape(-1).tolist(),  # (24,)
            }
            crop = Image.open(os.path.join(crops_dir, name)).convert("RGB").resize((256, 256))
            ov = np.array(crop).astype(np.float32)
            m = sil[b].cpu().numpy()
            ov[..., 0] = np.clip(ov[..., 0] + 120 * m, 0, 255)
            Image.fromarray(ov.astype(np.uint8)).save(
                os.path.join(ov_dir, os.path.splitext(name)[0] + ".png"))
    json.dump(results, open(os.path.join(out_dir, "params.json"), "w"))

    # One canonical + one posed mesh per scene. Shape = median betas across views
    # (robust to per-view wobble); canonical = rest pose, posed = representative view.
    keys = list(results)
    betas = np.median([results[k]["betas"] for k in keys], axis=0)
    limbs = np.median([results[k]["betas_limbs"] for k in keys], axis=0)
    rep_pose = np.array(results[keys[len(keys) // 2]]["pose_rotmat"])  # (35,3,3)
    bt = torch.tensor(betas, dtype=torch.float32, device=device)[None]
    lt = torch.tensor(limbs, dtype=torch.float32, device=device)[None]
    z = torch.zeros(1, 3, device=device)
    with torch.no_grad():
        Vc, _, _ = model.smal(beta=bt, betas_limbs=lt, theta=torch.zeros(1, 35, 3, device=device),
                              trans=z, get_skin=True)
        Vp, _, _ = model.smal(beta=bt, betas_limbs=lt,
                              pose=torch.tensor(rep_pose, dtype=torch.float32, device=device)[None],
                              trans=z, get_skin=True)
    save_obj(os.path.join(out_dir, "canonical.obj"), Vc[0].cpu().numpy(), faces_np)
    save_obj(os.path.join(out_dir, "posed.obj"), Vp[0].cpu().numpy(), faces_np)

    print(f"[ok] {scene_dir}: {len(results)} crops, canonical.obj + posed.obj -> {out_dir}")
    return len(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir")
    g.add_argument("--root")
    ap.add_argument("--out_subdir", default="preprocess")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, cfg = build_model(args.device)
    norm_dict = make_norm_dict(args.device)
    print("BARC ready.")

    scenes = ([args.scene_dir] if args.scene_dir else
              [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
               if os.path.isdir(os.path.join(args.root, d))])
    for scene in scenes:
        try:
            process_scene(scene, args.out_subdir, model, cfg, norm_dict, args.device, args.batch_size)
        except Exception as e:
            print(f"[skip] {scene}: {e}")


if __name__ == "__main__":
    main()
