"""Overlay externally-fitted D-SMAL meshes onto scene images (QC for dsmal_dataset)."""
import os, sys, json, argparse
import numpy as np
import torch

DSMAL_ROOT = "/home/yyang/mnt/workspace/received_data_from_Pinstudio_20260424/InterPet2026/dsmal_dataset"
SCENE_ROOT = "/home/yyang/mnt/workspace/received_data_from_Pinstudio_20260424/unzipped/0423"


def load_smal():
    os.chdir(DSMAL_ROOT)  # dsmal_code resolves data/smal_data relative to cwd
    sys.path.insert(0, os.path.join(DSMAL_ROOT, "dsmal_code"))
    import _compat_shim  # noqa: F401
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    from configs.SMAL_configs import SMAL_MODEL_CONFIG
    smal = SMAL(smal_model_type="39dogs_norm_newv3", template_name="neutral",
                logscale_part_list=SMAL_MODEL_CONFIG["39dogs_norm_newv3"]["logscale_part_list"])
    return smal


def fit_verts(smal, npz_path, stage="offset"):
    d = np.load(npz_path)
    t = lambda k: torch.tensor(d[f"{stage}_{k}"])
    verts = smal(beta=t("betas"), betas_limbs=t("betas_limbs"), pose=t("pose"),
                 trans=t("trans"), vert_off_compact=t("vert_off_compact"),
                 get_skin=True, uniform_scale=torch.exp(t("log_scale")))[0]
    return verts[0].numpy(), smal.faces.cpu().numpy() if torch.is_tensor(smal.faces) else np.asarray(smal.faces)


def project(verts, c2w, fx, fy, cx, cy):
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ verts.T + w2c[:3, 3:4]).T
    z = cam[:, 2:3]
    uv = np.stack([cam[:, 0] / z[:, 0] * fx + cx, cam[:, 1] / z[:, 0] * fy + cy], 1)
    return uv, z[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogs", nargs="+", default=["00003-nara", "00062-bear", "00211-pon"])
    ap.add_argument("--out", default="/home/yyang/mnt/workspace/exps/_dsmal_overlay.png")
    ap.add_argument("--stage", default="offset", choices=["offset", "param"])
    ap.add_argument("--ds", type=int, default=4, help="image downscale")
    ap.add_argument("--raw_world", action="store_true",
                    help="treat fit as raw COLMAP world; apply scene_norm to verts")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from PIL import Image

    smal = load_smal()
    n_views = 2
    fig, axes = plt.subplots(len(args.dogs), n_views, figsize=(5 * n_views, 6.5 * len(args.dogs)))
    axes = np.atleast_2d(axes)

    for r, dog in enumerate(args.dogs):
        npz = os.path.join(DSMAL_ROOT, "params", f"{dog}.npz")
        verts, faces = fit_verts(smal, npz, args.stage)
        if args.raw_world:
            sn = json.load(open(os.path.join(SCENE_ROOT, dog, "colmap/preprocess/scene_norm.json")))
            verts = (verts - np.array(sn["center"])) * sn["scale"]
        cams = json.load(open(os.path.join(SCENE_ROOT, dog, "colmap/preprocess/cameras.json")))
        frames = {os.path.splitext(f["name"])[0]: f for f in cams["frames"]}
        d = np.load(npz)
        holdout = [v for v in d[f"{args.stage}_holdout_views"] if v in frames]
        names = sorted(frames)
        picks = [holdout[len(holdout) // 2] if holdout else names[0], names[len(names) // 3]]

        for c, vn in enumerate(picks[:n_views]):
            fr = frames[vn]
            img = Image.open(os.path.join(SCENE_ROOT, dog, "colmap", fr["image_path"]))
            img = img.resize((fr["width"] // args.ds, fr["height"] // args.ds))
            s = 1.0 / args.ds
            uv, z = project(verts, np.array(fr["c2w"]), fr["fx"] * s, fr["fy"] * s, fr["cx"] * s, fr["cy"] * s)
            order = np.argsort(-z[faces].mean(1))  # back-to-front
            ax = axes[r, c]
            ax.imshow(img)
            ax.add_collection(PolyCollection(uv[faces[order]], facecolors=(0.2, 0.9, 1.0, 0.25),
                                             edgecolors=(0.0, 0.4, 0.6, 0.15), linewidths=0.2))
            tag = "holdout" if vn in holdout else "train"
            ax.set_title(f"{dog}  view {vn} ({tag})", fontsize=10)
            ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0); ax.axis("off")

    fig.suptitle(f"D-SMAL external fit overlay — stage={args.stage}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
