"""Dog-LRM network: frozen DINOv2 image tokens + canonical point tokens -> cross-attn
-> per-point Gaussian attributes. Minimal v1 to validate the training pipeline.

DINOv2 is a swappable stand-in for DINOv3 (transformers 4.46 lacks DINOv3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_embed(x, n_freq=8):
    freqs = (2.0 ** torch.arange(n_freq, device=x.device)) * torch.pi
    xf = x[..., None] * freqs                       # [...,3,F]
    return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], dim=-1)


class CrossBlock(nn.Module):
    """Cross-attention only (no self-attn) so it scales O(N*M) to a high-resolution
    subdivided point set: each point queries the image tokens for its own appearance."""
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, q, kv):
        kv = self.norm_kv(kv)
        # need_weights=False keeps the memory-efficient SDPA path (no [N,M] score matrix)
        q = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.norm_m(q))


class DogLRM(nn.Module):
    def __init__(self, dim=384, n_layers=4, n_heads=6, n_freq=8, n_refine=2,
                 dino_name="facebook/dinov2-large", offset_max=0.15, base_scale=0.02,
                 gaussians_per_point=2):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(dino_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        self.offset_max, self.base_scale, self.K = offset_max, base_scale, gaussians_per_point

        self.img_proj = nn.Linear(self.dino.config.hidden_size, dim)
        pe = 3 + 3 * 2 * n_freq
        self.pt_proj = nn.Sequential(nn.Linear(pe, dim), nn.SiLU(), nn.Linear(dim, dim))
        # SD3 MM-DiT: joint (bidirectional) attention between points and image tokens,
        # adaLN-Zero modulation. temb (no motion/timestep here) = pooled global image code.
        from .mmdit import TransformerDecoder
        self.transformer = TransformerDecoder(block_type="sd3_mm_cond", num_layers=n_layers,
                                              num_heads=n_heads, inner_dim=dim, cond_dim=dim,
                                              mod_dim=None)
        self.temb_proj = nn.Linear(dim, dim)
        # per-point refinement on the high-res subdivided surface (cross-attn only)
        self.pt_proj_fine = nn.Sequential(nn.Linear(pe, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.refine = nn.ModuleList([CrossBlock(dim, n_heads) for _ in range(n_refine)])

        K = self.K
        self.head_offset = nn.Linear(dim, K * 3)
        self.head_scale = nn.Linear(dim, K * 3)
        self.head_quat = nn.Linear(dim, K * 4)
        self.head_opacity = nn.Linear(dim, K * 1)
        self.head_rgb = nn.Linear(dim, K * 3)
        for h in (self.head_offset, self.head_scale, self.head_quat,
                  self.head_opacity, self.head_rgb):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def encode_image(self, img, res=224):
        x = F.interpolate((img - self.mean) / self.std, size=(res, res),
                          mode="bilinear", align_corners=False)
        return self.dino(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS

    def forward(self, img, canonical_pts, posed_pts, subdivide=None, scale_clip=None, return_feat=False):
        img_tok = self.img_proj(self.encode_image(img))            # [B,M,D]
        temb = self.temb_proj(img_tok.mean(dim=1))                 # [B,D] global cond for adaLN
        pt = self.pt_proj(fourier_embed(canonical_pts))            # [B,N,D]
        x = self.transformer(pt, cond=img_tok, temb=temb)          # [B,N,D] joint attention
        canon = canonical_pts
        if subdivide is not None:                                  # densify output surface
            x = subdivide(x)                                       # [B,Vsub,D] (interpolated)
            canon = subdivide(canonical_pts)                       # [B,Vsub,3]
            posed_pts = subdivide(posed_pts)                       # [B,Vsub,3]
        # individuate each subdivided point by its own position, then pull appearance
        # from the image tokens -> appearance frequency no longer capped at base verts
        q = x + self.pt_proj_fine(fourier_embed(canon))
        for blk in self.refine:
            q = blk(q, img_tok)
        x = q
        B, N, _ = x.shape
        K = self.K

        offset = torch.tanh(self.head_offset(x)).view(B, N, K, 3) * self.offset_max
        scales = torch.exp(self.head_scale(x).view(B, N, K, 3).clamp(-6, 2)) * self.base_scale
        if scale_clip is not None:                                 # warmup: cap max scale
            scales = scales.clamp(max=scale_clip)
        q0 = torch.tensor([1.0, 0, 0, 0], device=x.device)
        quats = F.normalize(self.head_quat(x).view(B, N, K, 4) + q0, dim=-1)
        opacities = torch.sigmoid(self.head_opacity(x).view(B, N, K) + 2.0)
        rgb = torch.sigmoid(self.head_rgb(x).view(B, N, K, 3))
        means = posed_pts[:, :, None, :] + offset                  # [B,N,K,3]
        out = dict(means=means.reshape(B, N * K, 3),
                   scales=scales.reshape(B, N * K, 3),
                   quats=quats.reshape(B, N * K, 4),
                   opacities=opacities.reshape(B, N * K),
                   rgb=rgb.reshape(B, N * K, 3),
                   offset=offset.reshape(B, N * K, 3))
        if return_feat:
            out["feat"] = x                                        # [B,N,D] per-anchor feature (for Stage-2 fur head)
        return out
