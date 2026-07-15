#!/usr/bin/env python3
"""Dynamic-fur demo: static Stage-1 body vs body + wind-swaying strands.
Loads a trained Strands (strands.pt), drives sway_vec over time, renders a fixed view ->
sway.gif + a [static | sway frames] strip. Shows the app's static/dynamic decoupling."""
import os, sys, json, math
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, ".")
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.smal_model import SMALModel, load_pseudo_gt, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_fur_stage2 import Strands

dev = "cuda"; s = 4
dog = "00010-hanabi"
scene = f"received_data_from_Pinstudio_20260424/unzipped/0423/{dog}/colmap"
out = "exps/fur_demo"; RK = ["means", "quats", "scales", "opacities", "rgb"]

# Stage-1 body
smal = SMALModel(dev, n_subdiv=2)
model = DogLRM(gaussians_per_point=1).to(dev).eval()
model.load_state_dict(torch.load("/tmp/stage1_for_fur.pt", map_location=dev), strict=False)
gt = load_pseudo_gt(scene, "preprocess", smal.num_betas, dev)
canon = smal.canonical_verts(gt["betas"], gt["limbs"])
posed = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])
frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
rr, _, _, _ = _load_rgb_mask(scene, frames[0], s)
ref = F.interpolate(torch.from_numpy(rr).permute(2, 0, 1)[None].to(dev), (224, 224), mode="bilinear", align_corners=False)
with torch.no_grad():
    bf = {k: v[0].detach() for k, v in model(ref, canon, posed, subdivide=smal.subdivide).items()}
body = {k: bf[k] for k in RK}
posed1 = smal.subdivide(posed)[0]
diag = float((posed1.max(0).values - posed1.min(0).values).norm())

# reconstruct trained Strands (buffers come from state_dict; pass real diag + cfg attrs)
ck = torch.load(os.path.join(out, "cf_mid", "strands.pt"), map_location=dev)
N, cfg = ck["N"], ck["cfg"]
z3, z1, g = torch.zeros(N, 3, device=dev), torch.zeros(N, device=dev), torch.zeros(3, device=dev)
st = Strands(z3, z3, z1, z1, z3, diag, g, Kp=cfg["Kp"], radius_frac=cfg.get("radius_frac", 0.0032),
             op_init=cfg.get("op_init", 1.25), curl_amp=cfg.get("curl_amp", 0.0),
             curl_freq=cfg.get("curl_freq", 2.0)).to(dev)
st.load_state_dict(ck["sd"]); st.eval()

# fixed side view
v = frames[20]
K = intrinsics(v["fx"]/s, v["fy"]/s, v["cx"]/s, v["cy"]/s, dev)
c2w = torch.tensor(v["c2w"], device=dev).float(); W, H = v["width"]//s, v["height"]//s
white = torch.ones(3, device=dev)


def render(gs):
    return render_gaussians(gs["means"], gs["quats"], gs["scales"], gs["opacities"], gs["rgb"],
                            c2w, K, W, H, bg=white)[0].clamp(0, 1).cpu().numpy()


static = render(body)                                            # static mode = Stage-1 body only
amp, T = 0.5, 30
frames_img = []
with torch.no_grad():
    for f in range(T):
        t = f / T
        sway = amp * torch.sin(2 * math.pi * t + 0.4 * st.root_phase)[:, None] * st.root_b
        st.sway_vec.copy_(sway)
        S = st()
        gs = {k: torch.cat([body[k], S[k]]) for k in RK}
        frames_img.append((render(gs) * 255).astype(np.uint8))

os.makedirs(out, exist_ok=True)
# gif of swaying fur
Image.fromarray(frames_img[0]).save(os.path.join(out, "sway.gif"), save_all=True,
                                     append_images=[Image.fromarray(f) for f in frames_img[1:]],
                                     duration=60, loop=0)
# strip: static body | sway peak A | sway peak B
strip = np.concatenate([(static * 255).astype(np.uint8), frames_img[T // 4], frames_img[3 * T // 4]], 1)
Image.fromarray(strip).save(os.path.join(out, "static_vs_dynamic.png"))
print(f"saved {out}/sway.gif ({T} frames) + static_vs_dynamic.png [static body | dyn peakA | dyn peakB]")
