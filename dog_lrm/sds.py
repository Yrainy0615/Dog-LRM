"""Score Distillation Sampling (DreamFusion/Farm3D-style) from a frozen Stable
Diffusion 2.1-base, used to supervise novel/occluded views in feed-forward training.

The 2D diffusion prior pulls a rendered novel view toward the distribution of
plausible dog images; gradient flows render -> model params. Inference unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline


class SDSGuidance(nn.Module):
    def __init__(self, device, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                 prompt="a photo of a dog, full body, studio white background, sharp, high detail",
                 neg="blurry, low quality, deformed, extra limbs, disfigured"):
        super().__init__()
        self.device = device
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.tok, self.text_encoder = pipe.tokenizer, pipe.text_encoder.to(device).eval()
        self.sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.alphas = self.sched.alphas_cumprod.to(device)
        self.n_train = self.sched.config.num_train_timesteps
        self.scaling = self.vae.config.scaling_factor
        for m in (self.vae, self.unet, self.text_encoder):
            for p in m.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            self.text_emb = self._embed([neg, prompt])             # [2,L,D]: [neg, pos] for CFG

    def _embed(self, prompts):
        ids = self.tok(prompts, padding="max_length", max_length=self.tok.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0]

    def __call__(self, rgb, guidance=100.0, t_min=0.02, t_max=0.98):
        """rgb [1,3,H,W] in [0,1], differentiable. Returns scalar SDS loss."""
        x = F.interpolate(rgb, (512, 512), mode="bilinear", align_corners=False)
        lat = self.vae.encode((x * 2 - 1).half()).latent_dist.sample() * self.scaling
        t = torch.randint(int(t_min * self.n_train), int(t_max * self.n_train), (1,), device=self.device)
        noise = torch.randn_like(lat)
        lat_noisy = self.sched.add_noise(lat, noise, t)
        with torch.no_grad():
            pred = self.unet(torch.cat([lat_noisy] * 2), t,
                             encoder_hidden_states=self.text_emb).sample
            neg, pos = pred.chunk(2)
            noise_pred = neg + guidance * (pos - neg)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = torch.nan_to_num(w * (noise_pred - noise))
        target = (lat - grad).detach()                             # SpecifyGradient trick
        return 0.5 * F.mse_loss(lat.float(), target.float(), reduction="sum") / lat.shape[0]
