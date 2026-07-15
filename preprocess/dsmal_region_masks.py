"""P0-2: face/body region masks from D-SMAL fits, via z-buffered rasterization.

face = soft skinning mass of joints 16 (skull) + 32 (muzzle), using D-SMAL's own
weights (BARC weights differ, P0-1). Projected face prob is dilated toward the
image-up direction (SMAL head verts sit low on fluffy dogs). --viz renders QC
overlays; default mode caches per-view masks to <scene>/preprocess/region_masks/.
"""
import os, sys, json, argparse
import numpy as np
import torch

DSMAL_ROOT = "/home/yyang/mnt/workspace/received_data_from_Pinstudio_20260424/InterPet2026/dsmal_dataset"
SCENE_ROOT = "/home/yyang/mnt/workspace/received_data_from_Pinstudio_20260424/unzipped/0423"
FACE_JOINTS = [16, 32]


def load_smal():
    os.chdir(DSMAL_ROOT)
    sys.path.insert(0, os.path.join(DSMAL_ROOT, "dsmal_code"))
    import _compat_shim  # noqa: F401
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    from configs.SMAL_configs import SMAL_MODEL_CONFIG
    return SMAL(smal_model_type="39dogs_norm_newv3", template_name="neutral",
                logscale_part_list=SMAL_MODEL_CONFIG["39dogs_norm_newv3"]["logscale_part_list"])


def scene_verts(smal, dog, stage="offset"):
    """Posed verts in our normalized camera space + per-vertex soft face weight."""
    d = np.load(os.path.join(DSMAL_ROOT, "params", f"{dog}.npz"))
    t = lambda k: torch.tensor(d[f"{stage}_{k}"])
    verts = smal(beta=t("betas"), betas_limbs=t("betas_limbs"), pose=t("pose"),
                 trans=t("trans"), vert_off_compact=t("vert_off_compact"),
                 get_skin=True, uniform_scale=torch.exp(t("log_scale")))[0][0]
    sn = json.load(open(os.path.join(SCENE_ROOT, dog, "colmap/preprocess/scene_norm.json")))
    verts = (verts - torch.tensor(sn["center"], dtype=verts.dtype)) * sn["scale"]
    w = smal.weights if torch.is_tensor(getattr(smal, "weights", None)) else torch.tensor(np.asarray(smal.weights))
    w_face = w[:, FACE_JOINTS].sum(1).float()                    # [V] soft
    faces = smal.faces if torch.is_tensor(smal.faces) else torch.tensor(np.asarray(smal.faces))
    return verts.float(), faces.long(), w_face


def rasterize_face_prob(verts, faces, w_face, fr, ds, device="cuda"):
    """Returns (render_mask, face_prob) [H,W] at 1/ds res, z-buffered."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.ops import interpolate_face_attributes

    H, W = fr["height"] // ds, fr["width"] // ds
    s = 1.0 / ds
    K = torch.tensor([[fr["fx"] * s, 0, fr["cx"] * s], [0, fr["fy"] * s, fr["cy"] * s], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]
    w2c = torch.tensor(np.linalg.inv(np.array(fr["c2w"])), dtype=torch.float32, device=device)
    cam = cameras_from_opencv_projection(w2c[None, :3, :3], w2c[None, :3, 3], K,
                                         torch.tensor([[H, W]], device=device))
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    frag = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
        image_size=(H, W), faces_per_pixel=1, bin_size=None)).forward(mesh)
    mask = (frag.pix_to_face[0, :, :, 0] >= 0)
    attr = w_face.to(device)[faces.to(device)][:, :, None]       # [F,3,1]
    prob = interpolate_face_attributes(frag.pix_to_face, frag.bary_coords, attr)[0, :, :, 0, 0]
    return mask.cpu().numpy(), prob.clamp(0, 1).cpu().numpy()


def dilate_up(face_mask, px):
    """Isotropic dilation + extra upward growth (image-up ≈ head-top in studio rig)."""
    from scipy.ndimage import binary_dilation
    out = binary_dilation(face_mask, iterations=max(px // 2, 1))
    up = out.copy()
    for k in range(1, px + 1):                                   # shift up and accumulate
        up[:-k] |= out[k:]
    return up


def cache_all(args):
    """Cache per-view region masks for every dog with D-SMAL params, at 1/ds res.
    Output <scene>/preprocess/region_masks_s{ds}/<view>.png: R=face_prob*255, G=mesh mask."""
    from PIL import Image
    smal = load_smal()
    dogs = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(DSMAL_ROOT, "params")))
    for di, dog in enumerate(dogs):
        cam_path = os.path.join(SCENE_ROOT, dog, "colmap/preprocess/cameras.json")
        if not os.path.exists(cam_path):
            print(f"[skip] {dog}: no scene")
            continue
        out_dir = os.path.join(SCENE_ROOT, dog, f"colmap/preprocess/region_masks_s{args.ds}")
        os.makedirs(out_dir, exist_ok=True)
        verts, faces, w_face = scene_verts(smal, dog, args.stage)
        frames = json.load(open(cam_path))["frames"]
        for fr in frames:
            mask, prob = rasterize_face_prob(verts, faces, w_face, fr, args.ds)
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[:, :, 0] = (prob * 255).astype(np.uint8)
            rgb[:, :, 1] = mask.astype(np.uint8) * 255
            Image.fromarray(rgb).save(os.path.join(out_dir, fr["name"].replace(".jpg", ".png")))
        print(f"[{di+1}/{len(dogs)}] {dog}: {len(frames)} views", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogs", nargs="+", default=["00003-nara", "00062-bear", "00211-pon"])
    ap.add_argument("--stage", default="offset")
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--dilate_frac", type=float, default=0.05, help="upward dilation / projected diag")
    ap.add_argument("--viz", default="/home/yyang/mnt/workspace/exps/_p0_region_masks.png")
    ap.add_argument("--cache", action="store_true", help="batch-cache masks for all dogs instead of viz")
    args = ap.parse_args()
    if args.cache:
        cache_all(args)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    smal = load_smal()
    fig, axes = plt.subplots(len(args.dogs), 2, figsize=(10, 6.5 * len(args.dogs)))
    axes = np.atleast_2d(axes)

    for r, dog in enumerate(args.dogs):
        verts, faces, w_face = scene_verts(smal, dog, args.stage)
        print(f"{dog}: {int((w_face > 0.5).sum())} face verts")
        cams = json.load(open(os.path.join(SCENE_ROOT, dog, "colmap/preprocess/cameras.json")))
        frames = sorted(cams["frames"], key=lambda f: f["name"])
        picks = [frames[len(frames) // 3], frames[2 * len(frames) // 3]]
        for c, fr in enumerate(picks):
            mask, prob = rasterize_face_prob(verts, faces, w_face, fr, args.ds)
            uv = None
            diag_px = np.sqrt((np.argwhere(mask).ptp(0) ** 2).sum()) if mask.any() else 100
            face_dil = dilate_up(prob > 0.5, max(int(args.dilate_frac * diag_px), 2))
            img = np.asarray(Image.open(os.path.join(SCENE_ROOT, dog, "colmap", fr["image_path"]))
                             .resize((fr["width"] // args.ds, fr["height"] // args.ds))) / 255.0
            ov = img.copy()
            body = mask & ~face_dil
            ov[body] = ov[body] * 0.55 + np.array([0.1, 0.3, 1.0]) * 0.45
            ov[face_dil] = ov[face_dil] * 0.55 + np.array([1.0, 0.15, 0.1]) * 0.45
            ax = axes[r, c]
            ax.imshow(ov)
            ax.contour(prob, levels=[0.5], colors="yellow", linewidths=0.8)
            ax.set_title(f"{dog} view {os.path.splitext(fr['name'])[0]} (red=face+dilate, yellow=raw w>0.5)", fontsize=9)
            ax.axis("off")

    fig.suptitle("P0-2: D-SMAL face/body region masks (z-buffered, soft w_face)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.viz, dpi=110)
    print("saved", args.viz)


if __name__ == "__main__":
    main()
