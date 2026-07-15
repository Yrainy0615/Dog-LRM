#!/usr/bin/env python3
"""Template-mesh motion-comparison renders: same skeletal clip (tail wag / head
sway / ears via make_pose_deltas), same 360-degree orbit, resolution and encoder
settings as the GS videos — but rasterizing the rigged mesh itself (clay shading,
pytorch3d). MODE=dynamic|static, BG=white|black, HIRES=4k|1 env, like
animate_fur_wind.py. Output: exps/coatdepth_demo/mesh_{mode}_{bg}_{res}.mp4"""
import math, os, subprocess, sys
import numpy as np
import torch
import trimesh

sys.path.insert(0, ".")
import animate_gs_coatdepth as base
from pytorch3d.renderer import (BlendParams, HardPhongShader, MeshRasterizer, MeshRenderer,
                                PerspectiveCameras, PointLights, RasterizationSettings,
                                TexturesVertex)
from pytorch3d.structures import Meshes

dev = "cuda"
DATA, OUT = "train_data", "exps/coatdepth_demo"
FPS = 24
NF = int(os.environ.get("NF", "480"))


def main():
    mode = os.environ.get("MODE", "dynamic")
    bgname = os.environ.get("BG", "white")
    four_k = os.environ.get("HIRES", "4k") == "4k"
    W, H = (3840, 2160) if four_k else (1920, 1080)
    FX = 4700.0 if four_k else 2350.0
    bg = (1.0, 1.0, 1.0) if bgname == "white" else (0.0, 0.0, 0.0)

    nodes, joint_ids, ibm, verts, vj, vw = base.parse_glb(f"{DATA}/shba701_mesh_7k_noroot.glb")
    name2id = {nodes[i].get("name", ""): i for i in range(len(nodes))}
    pose_fn = base.make_pose_deltas(nodes, name2id)
    verts_t = torch.from_numpy(verts).to(dev)
    vj_t = torch.from_numpy(vj).to(dev)
    vw_t = torch.from_numpy(vw).to(dev)
    tm = trimesh.load(f"{DATA}/shba701_mesh_7k_noroot.glb", process=False, force="mesh")
    faces = torch.from_numpy(np.asarray(tm.faces, np.int64)).to(dev)
    vcol = torch.full((1, len(verts), 3), 0.73, device=dev)     # clay gray

    raster = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
    ctr = [0.0, 0.24, -0.06]
    path = f"{OUT}/mesh_{mode}_{bgname}_{'4k' if four_k else 'hires'}.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "medium" if four_k else "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    with torch.no_grad():
        for f in range(NF):
            th = 2 * math.pi * f / NF
            eye = [ctr[0] + 1.30 * math.cos(th), 0.46, ctr[2] + 1.30 * math.sin(th)]
            c2w, _ = __import__("animate_fur_wind").lookat(eye, ctr, FX, W, H)
            w2c = torch.inverse(c2w)
            R_cv, t_cv = w2c[:3, :3], w2c[:3, 3]
            R = (R_cv.T @ torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=dev)))[None]
            T = (t_cv * torch.tensor([-1.0, -1.0, 1.0], device=dev))[None]
            cam = PerspectiveCameras(focal_length=((FX, FX),), principal_point=((W / 2, H / 2),),
                                     R=R, T=T, in_ndc=False, image_size=((H, W),), device=dev)
            if mode == "static":
                v = verts_t
            else:
                Sk = base.vertex_affines(nodes, joint_ids, ibm, pose_fn(f / FPS))
                v, _ = base.skin_verts(Sk, verts_t, vj_t, vw_t)
            mesh = Meshes(verts=[v], faces=[faces], textures=TexturesVertex(vcol))
            lights = PointLights(device=dev, location=[eye],
                                 ambient_color=((0.45,) * 3,), diffuse_color=((0.55,) * 3,),
                                 specular_color=((0.05,) * 3,))
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
                shader=HardPhongShader(device=dev, cameras=cam, lights=lights,
                                       blend_params=BlendParams(background_color=bg)))
            img = renderer(mesh)[0, ..., :3]
            proc.stdin.write((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("saved", path)


if __name__ == "__main__":
    main()
