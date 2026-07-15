#!/usr/bin/env python3
"""Standalone re-render of fur dynamics from a trained FurDogLRM checkpoint.

Decouples dynamics tuning from training: load model.pt, forward once to get the
body/fur Gaussians (with exact fur roots + rest offsets), then run an
amplitude-controlled spring so each fur tip sways by a chosen fraction of its
own length. No retraining.

The old save_fur_dynamics used absolute stiffness/wind/gravity constants whose
equilibrium displacement (F/stiffness) came out to ~0.02-0.3% of body size --
sub-pixel, i.e. visually static. Here the forcing is scaled by k*target so the
steady-state sway is amp_frac * strand_length regardless of stiffness.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.render import render_gaussians
from dog_lrm.smal_model import SMALModel
import train_dog_lrm_fur as T


def sway_dynamics(body, fur, view, amp_frac=0.4, stiffness=60.0, zeta=0.2,
                  wind_frac=0.8, gravity_frac=0.4, gust_hz=0.4,
                  fps=15, frames=60, seed=0, fur_opacity_floor=0.0, fur_scale_mult=1.0,
                  mobility=None, wind_exposure=True):
    """Amplitude-controlled cantilever sway. body/fur are per-branch tensor dicts
    for a single scene. Returns a list of uint8 frames.

    fur_opacity_floor lifts the (typically collapsed) fur opacity at render time so
    the strand geometry + motion can be inspected independently of the unsolved
    body/fur appearance-split. It does NOT touch geometry."""
    device = fur["means"].device
    bg = torch.ones(3, device=device)
    fur_op = fur["opacities"].clamp_min(fur_opacity_floor)
    fur_sc = fur["scales"] * fur_scale_mult
    roots = fur["roots"]                                   # [M,3]
    rest = fur["delta"]                                    # [M,3] rest offset from root
    strand_len = rest.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target = amp_frac * strand_len                         # desired sway magnitude [M,1]

    wind_dir = F.normalize(torch.tensor([1.0, 0.1, 0.2], device=device), dim=0)
    down = torch.tensor([0.0, 0.0, -1.0], device=device)
    k = stiffness
    c = 2.0 * (k ** 0.5) * zeta                            # damping for chosen ratio
    dt = 1.0 / fps
    g = torch.Generator().manual_seed(seed)
    phi = (torch.rand(roots.shape[0], 1, generator=g) * 2 * np.pi).to(device)  # per-strand phase

    # windward strands (growth dir facing the wind) catch more force; leeward catch ~none
    strand_dir = F.normalize(rest, dim=-1, eps=1e-6)
    exposure = ((strand_dir * wind_dir.view(1, 3)).sum(-1, keepdim=True).clamp_min(0.0)
                if wind_exposure else torch.ones_like(target))
    # per-strand mobility in [0,1]: 0 pins the strand (e.g. face fur), 1 = free
    mob = torch.ones_like(target) if mobility is None else mobility.view(-1, 1).to(device)

    x = rest.clone()
    v = torch.zeros_like(x)
    out = []
    with torch.no_grad():
        for t in range(frames):
            ph = torch.sin(2 * np.pi * gust_hz * t / fps + phi)               # [M,1]
            drive = (wind_frac * ph * exposure * wind_dir.view(1, 3)
                     + gravity_frac * down.view(1, 3))
            force = k * (rest - x) - c * v + k * (target * mob) * drive
            v = v + dt * force
            x = x + dt * v
            fur_means = roots + x
            rgb, _ = render_gaussians(
                torch.cat([body["means"], fur_means], dim=0),
                torch.cat([body["quats"], fur["quats"]], dim=0),
                torch.cat([body["scales"], fur_sc], dim=0),
                torch.cat([body["opacities"], fur_op], dim=0),
                torch.cat([body["rgb"], fur["rgb"]], dim=0),
                view["c2w"], view["K"], view["W"], view["H"], bg=bg)
            out.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="exp dir containing model.pt")
    ap.add_argument("--scene_dirs", nargs="+", required=True)
    ap.add_argument("--preset", action="append", default=[])
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--body_offset_max", type=float, default=0.035)
    ap.add_argument("--fur_len_floor_frac", type=float, default=0.0)
    ap.add_argument("--body_k", type=int, default=1)
    ap.add_argument("--fur_k", type=int, default=1)
    ap.add_argument("--dino_name", default="facebook/dinov2-large")
    ap.add_argument("--device", default="cuda")
    # dynamics knobs
    ap.add_argument("--amp_frac", type=float, default=0.4, help="tip sway as fraction of strand length")
    ap.add_argument("--stiffness", type=float, default=60.0)
    ap.add_argument("--zeta", type=float, default=0.2, help="damping ratio (<1 = wobbly)")
    ap.add_argument("--wind_frac", type=float, default=0.8)
    ap.add_argument("--gravity_frac", type=float, default=0.4)
    ap.add_argument("--gust_hz", type=float, default=0.4)
    ap.add_argument("--fur_opacity_floor", type=float, default=0.0,
                    help="lift fur opacity at render time to inspect strand motion (viz only)")
    ap.add_argument("--fur_scale_mult", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--tag", default="rerender")
    args = ap.parse_args()

    dev = args.device
    overrides = T.parse_preset_overrides(args.preset)
    smal = SMALModel(dev)
    faces_sub = T.subdivided_faces(smal.faces, 1).to(dev)
    scenes = [T.load_scene(sd, smal, args.scale_div, dev,
                           overrides.get(T.scene_name(sd))) for sd in args.scene_dirs]

    canonical = torch.stack([s["canonical"] for s in scenes])
    posed = torch.stack([s["posed"] for s in scenes])
    posed_sub = smal.subdivide(posed)
    normals_sub = T.orient_normals_outward(posed_sub, T.vertex_normals(posed_sub, faces_sub))
    body_diag = torch.stack([s["body_diag"] for s in scenes]).to(dev)
    fur_lmax = torch.tensor([T.FUR_PRESETS[s["preset"]]["lmax"] for s in scenes], device=dev) * body_diag
    fur_latmax = torch.tensor([T.FUR_PRESETS[s["preset"]]["lat"] for s in scenes], device=dev) * body_diag
    fur_floor_frac = torch.full((len(scenes),), args.fur_len_floor_frac, device=dev)

    model = T.FurDogLRM(dino_name=args.dino_name, body_offset_max=args.body_offset_max,
                        body_k=args.body_k, fur_k=args.fur_k).to(dev)
    sd = torch.load(os.path.join(args.out, "model.pt"), map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("dino.")]
    if missing or unexpected:
        print(f"load_state_dict: missing(non-dino)={missing} unexpected={unexpected}", flush=True)
    model.eval()

    ref = [len(s["views"]) // 2 for s in scenes]
    inputs = torch.stack([scenes[b]["inputs_all"][ref[b]] for b in range(len(scenes))])
    with torch.no_grad():
        gs = model(inputs, canonical, posed, normals_sub, fur_lmax, fur_latmax,
                   fur_floor_frac=fur_floor_frac, subdivide=smal.subdivide)

    import imageio
    for b, scene in enumerate(scenes):
        body = {k: gs["body"][k][b].detach() for k in ("means", "quats", "scales", "opacities", "rgb")}
        fur = {k: gs["fur"][k][b].detach() for k in
               ("means", "quats", "scales", "opacities", "rgb", "roots", "delta")}
        view = scene["views"][len(scene["views"]) // 2]
        frames = sway_dynamics(body, fur, view, amp_frac=args.amp_frac,
                               stiffness=args.stiffness, zeta=args.zeta,
                               wind_frac=args.wind_frac, gravity_frac=args.gravity_frac,
                               gust_hz=args.gust_hz, fps=args.fps, frames=args.frames,
                               fur_opacity_floor=args.fur_opacity_floor,
                               fur_scale_mult=args.fur_scale_mult)
        path = os.path.join(args.out, f"{args.tag}_{scene['name']}_{scene['preset']}.mp4")
        imageio.mimsave(path, frames, fps=args.fps, quality=8)
        # quick motion stat
        arr = np.stack([f.astype(np.float32) for f in frames])
        d = np.abs(arr - arr[0]).mean(axis=(1, 2, 3)).max()
        print(f"saved {path}  max|frame-frame0|={d:.3f}", flush=True)


if __name__ == "__main__":
    main()
