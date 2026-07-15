"""QC: visualize the exact face/body label grids the P1a trainer consumes.

Random-samples dogs/views, rebuilds the label via train_dog_lrm_decomp._label_grid
(same code path as training) and overlays: blue=body, red=face at the 74x74 token
grid (nearest-upsampled), white contour = GT mask.
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, "/home/yyang/mnt/workspace")
from train_dog_lrm_decomp import _label_grid, list_scenes
from train_dog_lrm_ddp import _load_rgb_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/yyang/mnt/workspace/received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/yyang/mnt/workspace/exps/_p1a_label_samples.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import torch

    rng = np.random.RandomState(args.seed)
    scenes = list_scenes(args.root)
    picks = rng.choice(len(scenes), size=args.n, replace=False)
    cols = 3
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6.5 * rows))

    for k, si in enumerate(picks):
        scene = scenes[si]
        frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
        fr = frames[rng.randint(len(frames))]
        rgb, mask, W, H = _load_rgb_mask(scene, fr, 8)
        label = _label_grid(scene, fr, mask)                       # [74,74] as in training
        lab_up = F.interpolate(label[None, None].float(), size=(H, W), mode="nearest")[0, 0].numpy()
        ov = rgb.copy()
        ov[lab_up == 1] = ov[lab_up == 1] * 0.55 + np.array([0.1, 0.3, 1.0]) * 0.45
        ov[lab_up == 2] = ov[lab_up == 2] * 0.55 + np.array([1.0, 0.15, 0.1]) * 0.45
        ax = axes.flat[k]
        ax.imshow(ov)
        ax.contour(mask[:, :, 0], levels=[0.5], colors="white", linewidths=0.6)
        nf, nb = int((label == 2).sum()), int((label == 1).sum())
        ax.set_title(f"{scene.split('/')[-2]} v{os.path.splitext(fr['name'])[0]} "
                     f"(face {nf} / body {nb} cells)", fontsize=9)
        ax.axis("off")
    for k in range(args.n, rows * cols):
        axes.flat[k].axis("off")
    fig.suptitle("P1a token-routing labels (74x74, trainer code path): red=face, blue=body, white=GT mask", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
