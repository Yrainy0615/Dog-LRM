"""Trainable BARC predictor: dog crop -> differentiable SMAL shape + pose.

Wraps BARC's ModelImageTo3d (built via preprocess.barc_infer.build_model, which patches
out the version-locked silhouette renderer with a stub). The backbone -- keypoint/seg
hourglass and the breed ResNet feature extractor -- is frozen; only the shape heads
(breed_model.linear_betas / linear_betas_limbs) and the 3D pose regressor (model_3d) are
fine-tuned, so SMAL geometry can move under the LRM rendering loss without the whole
predictor drifting.

forward(crop) -> (betas[B,30], betas_limbs[B,7], body_pose[B,34,3] axis-angle), all
differentiable. Root orientation / global trans / scale are NOT read from BARC: those are
the per-scene world placement, kept frozen from fit_smal.
"""
import os
import sys

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_axis_angle

RGB_MEAN = (0.4404, 0.4440, 0.4327)  # BARC color_normalize (subtract mean only)

# Fine-tune only these heads; everything else (backbone) stays frozen.
_TRAINABLE = ("breed_model.linear_betas", "breed_model.linear_betas_limbs", "model_3d")
# ...but keep the normalizing-flow pose prior frozen even inside model_3d: fine-tuning it
# destabilised pose in the first joint run (geometry good ~5k then appearance collapsed).
_KEEP_FROZEN = ("model_3d.pose_normflow_model",)


class BarcPredictor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        sys.path.insert(0, os.path.abspath("preprocess"))
        import barc_infer as bi
        model, _ = bi.build_model(device)   # returns model.to(device).eval()
        self.model = model
        self._norm = bi.make_norm_dict(device)  # dict of tensors already on `device`

        for p in self.model.parameters():
            p.requires_grad = False
        for name in _TRAINABLE:
            for p in self._submodule(name).parameters():
                if p.is_floating_point() or p.is_complex():  # normflow holds int buffers-as-params
                    p.requires_grad = True
        for name in _KEEP_FROZEN:                            # re-freeze normflow pose prior
            for p in self._submodule(name).parameters():
                p.requires_grad = False

        self.register_buffer("rgb_mean", torch.tensor(RGB_MEAN, device=device).view(1, 3, 1, 1))

    def _submodule(self, dotted):
        m = self.model
        for part in dotted.split("."):
            m = getattr(m, part)
        return m

    def forward(self, crop):
        """crop [B,3,256,256] in [0,1] -> (betas, betas_limbs, body_pose[B,34,3])."""
        x = crop - self.rgb_mean
        _, out_unnorm, out_reproj = self.model(x, norm_dict=self._norm)
        body_pose = matrix_to_axis_angle(out_unnorm["pose_rotmat"][:, 1:])  # drop root
        return out_reproj["betas"], out_reproj["betas_limbs"], body_pose
