#!/usr/bin/env python3
"""Stage 1 preprocessing: COLMAP sparse model -> per-image cameras + scene normalization.

Assumptions (see ANIMAL_LHM_PLAN.md):
  - One pet == one COLMAP scene dir, laid out as:
        <scene>/images/...
        <scene>/sparse/0/{cameras,images,points3D}.{bin|txt}   (or <scene>/sparse/...)
  - Output goes to <scene>/preprocess/{cameras.json, scene_norm.json}.

Camera convention of the output c2w is OpenCV / COLMAP: +X right, +Y down, +Z forward
(into the scene). Matching this to the renderer's convention is a separate downstream step.

Dependency: numpy only.
"""
import argparse
import json
import os
import struct
from collections import namedtuple

import numpy as np

# ---------------------------------------------------------------------------
# COLMAP model reader (compact, supports .bin and .txt). Adapted from the
# canonical COLMAP scripts/python/read_write_model.py (numpy-only subset).
# ---------------------------------------------------------------------------

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
_CAMERA_MODELS = [
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
]
_MODEL_ID_TO_MODEL = {m.model_id: m for m in _CAMERA_MODELS}
_MODEL_NAME_TO_MODEL = {m.model_name: m for m in _CAMERA_MODELS}

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])


def _read(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        (num,) = _read(fid, 8, "Q")
        for _ in range(num):
            cam_id, model_id, w, h = _read(fid, 24, "iiQQ")
            n = _MODEL_ID_TO_MODEL[model_id].num_params
            params = _read(fid, 8 * n, "d" * n)
            cameras[cam_id] = Camera(cam_id, _MODEL_ID_TO_MODEL[model_id].model_name,
                                     w, h, np.array(params))
    return cameras


def read_cameras_text(path):
    cameras = {}
    with open(path) as fid:
        for line in fid:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            e = line.split()
            cam_id = int(e[0])
            cameras[cam_id] = Camera(cam_id, e[1], int(e[2]), int(e[3]),
                                     np.array([float(x) for x in e[4:]]))
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        (num,) = _read(fid, 8, "Q")
        for _ in range(num):
            p = _read(fid, 64, "idddddddi")
            image_id = p[0]
            qvec = np.array(p[1:5])
            tvec = np.array(p[5:8])
            camera_id = p[8]
            name = b""
            c = fid.read(1)
            while c != b"\x00":
                name += c
                c = fid.read(1)
            (num2d,) = _read(fid, 8, "Q")
            fid.read(24 * num2d)  # skip 2D points
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name.decode())
    return images


def read_images_text(path):
    images = {}
    with open(path) as fid:
        lines = [l.strip() for l in fid if l.strip() and l[0] != "#"]
    for i in range(0, len(lines), 2):  # every 2nd line is the 2D-points line
        e = lines[i].split()
        image_id = int(e[0])
        qvec = np.array([float(x) for x in e[1:5]])
        tvec = np.array([float(x) for x in e[5:8]])
        camera_id = int(e[8])
        name = e[9]
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name)
    return images


def read_points3d_xyz_binary(path):
    pts = []
    with open(path, "rb") as fid:
        (num,) = _read(fid, 8, "Q")
        for _ in range(num):
            b = _read(fid, 43, "QdddBBBd")
            pts.append(b[1:4])
            (track_len,) = _read(fid, 8, "Q")
            fid.read(8 * track_len)
    return np.array(pts, dtype=np.float64).reshape(-1, 3)


def read_points3d_xyz_text(path):
    pts = []
    with open(path) as fid:
        for line in fid:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            e = line.split()
            pts.append([float(e[1]), float(e[2]), float(e[3])])
    return np.array(pts, dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def w2c_to_c2w(qvec, tvec):
    R = qvec2rotmat(qvec)  # world->cam rotation
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ tvec
    return c2w


# Map a COLMAP camera model to (fx, fy, cx, cy) + whether distortion params exist.
def intrinsics_from_params(model, params):
    p = params
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2], False
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3], False
    if model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        return p[0], p[0], p[1], p[2], True
    if model in ("RADIAL", "RADIAL_FISHEYE"):
        return p[0], p[0], p[1], p[2], True
    if model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
        return p[0], p[1], p[2], p[3], True
    raise ValueError(f"Unsupported camera model: {model}")


# ---------------------------------------------------------------------------
# Scene normalization (center + isotropic scale from sparse points)
# ---------------------------------------------------------------------------

def compute_scene_norm(xyz, percentile):
    center = np.median(xyz, axis=0)
    r = np.linalg.norm(xyz - center, axis=1)
    radius = float(np.percentile(r, percentile))
    scale = 1.0 / radius if radius > 0 else 1.0
    return center, scale, radius


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def find_sparse_dir(scene_dir):
    for cand in (os.path.join(scene_dir, "sparse", "0"),
                 os.path.join(scene_dir, "sparse")):
        if os.path.isdir(cand) and (
            os.path.exists(os.path.join(cand, "cameras.bin"))
            or os.path.exists(os.path.join(cand, "cameras.txt"))
        ):
            return cand
    raise FileNotFoundError(f"No COLMAP sparse model under {scene_dir}/sparse[/0]")


def load_model(sparse_dir):
    if os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        cams = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        imgs = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        xyz = read_points3d_xyz_binary(os.path.join(sparse_dir, "points3D.bin"))
    else:
        cams = read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
        imgs = read_images_text(os.path.join(sparse_dir, "images.txt"))
        xyz = read_points3d_xyz_text(os.path.join(sparse_dir, "points3D.txt"))
    return cams, imgs, xyz


def process_scene(scene_dir, out_dir, percentile, normalize):
    sparse_dir = find_sparse_dir(scene_dir)
    cameras, images, xyz = load_model(sparse_dir)

    if normalize and len(xyz) > 0:
        center, scale, radius = compute_scene_norm(xyz, percentile)
    else:
        center, scale, radius = np.zeros(3), 1.0, 0.0

    any_distortion = False
    frames = []
    for img in sorted(images.values(), key=lambda i: i.name):
        cam = cameras[img.camera_id]
        fx, fy, cx, cy, has_dist = intrinsics_from_params(cam.model, cam.params)
        any_distortion = any_distortion or has_dist
        c2w = w2c_to_c2w(img.qvec, img.tvec)
        c2w[:3, 3] = (c2w[:3, 3] - center) * scale  # apply scene normalization
        frames.append({
            "name": img.name,
            "image_path": os.path.join("images", img.name),
            "camera_id": img.camera_id,
            "model": cam.model,
            "width": cam.width,
            "height": cam.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "has_distortion": bool(has_dist),
            "c2w": c2w.tolist(),
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump({
            "convention": "opencv",  # +X right, +Y down, +Z forward
            "normalized": bool(normalize),
            "num_frames": len(frames),
            "frames": frames,
        }, f, indent=2)
    with open(os.path.join(out_dir, "scene_norm.json"), "w") as f:
        json.dump({
            "center": center.tolist(),
            "scale": float(scale),
            "percentile": percentile,
            "radius": radius,
            "num_points": int(len(xyz)),
        }, f, indent=2)

    note = "  [WARN: lens distortion params present, ignored]" if any_distortion else ""
    print(f"[ok] {scene_dir}: {len(frames)} frames, scale={scale:.4g}{note}")
    return len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir", help="single COLMAP scene dir")
    g.add_argument("--root", help="parent dir; each immediate subdir is one scene")
    ap.add_argument("--out_subdir", default="preprocess",
                    help="output subdir inside each scene (default: preprocess)")
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="radius percentile for scene scale (default: 90)")
    ap.add_argument("--no_normalize", action="store_true",
                    help="skip scene normalization (keep raw COLMAP world)")
    args = ap.parse_args()

    normalize = not args.no_normalize
    if args.scene_dir:
        scenes = [args.scene_dir]
    else:
        scenes = [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
                  if os.path.isdir(os.path.join(args.root, d))]

    total = 0
    for scene in scenes:
        try:
            out_dir = os.path.join(scene, args.out_subdir)
            total += process_scene(scene, out_dir, args.percentile, normalize)
        except Exception as e:  # keep batch going; report the offender
            print(f"[skip] {scene}: {e}")
    print(f"[done] {len(scenes)} scene(s), {total} frames total")


if __name__ == "__main__":
    main()
