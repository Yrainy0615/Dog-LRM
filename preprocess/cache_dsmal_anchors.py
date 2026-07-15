"""P1a prep: cache per-scene D-SMAL anchors so trainers never import dsmal_code.

Writes <scene>/preprocess/dsmal_anchors.npz:
  canon  [V,3] rest-pose shape (identity pose, no trans/scale) — PE input
  posed  [V,3] fitted pose in our normalized camera space — Gaussian anchors
  w_face [V]   soft skull+muzzle skinning mass (D-SMAL's own weights)
  faces  [F,3]
"""
import os, json
import numpy as np
import torch

from dsmal_region_masks import DSMAL_ROOT, SCENE_ROOT, load_smal, scene_verts, FACE_JOINTS


def main():
    smal = load_smal()
    eye = torch.eye(3)[None, None].repeat(1, 35, 1, 1)
    dogs = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(DSMAL_ROOT, "params")))
    for dog in dogs:
        scene = os.path.join(SCENE_ROOT, dog, "colmap")
        if not os.path.exists(os.path.join(scene, "preprocess/cameras.json")):
            print(f"[skip] {dog}: no scene")
            continue
        d = np.load(os.path.join(DSMAL_ROOT, "params", f"{dog}.npz"))
        t = lambda k: torch.tensor(d["offset_" + k])
        canon = smal(beta=t("betas"), betas_limbs=t("betas_limbs"), pose=eye,
                     trans=torch.zeros(1, 3), vert_off_compact=t("vert_off_compact"),
                     get_skin=True)[0][0]
        posed, faces, w_face = scene_verts(smal, dog, "offset")
        np.savez(os.path.join(scene, "preprocess/dsmal_anchors.npz"),
                 canon=canon.numpy().astype(np.float32),
                 posed=posed.numpy().astype(np.float32),
                 w_face=w_face.numpy().astype(np.float32),
                 faces=faces.numpy().astype(np.int64))
        print(f"[ok] {dog}")


if __name__ == "__main__":
    main()
