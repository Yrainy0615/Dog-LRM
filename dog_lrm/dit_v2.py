"""Modernized MM-DiT backbone for Dog-LRM v2 (self-contained, no diffusers).

Upgrades over dog_lrm/mmdit.py + transformer_dit.py (SD3 blocks):
  * adaLN-Zero modulation actually applied on BOTH streams (the v1 blocks had the
    gate/shift/scale lines commented out, so temb conditioning was inert),
  * RMSNorm everywhere + per-head QK RMSNorm (training stability at higher dim),
  * SwiGLU FFN (same param budget as 4x GELU via 8/3 hidden ratio),
  * 2D axial RoPE on image tokens; point tokens get identity rotation (Flux-style
    text treatment) and keep their Fourier absolute PE from the model wrapper,
  * final RMSNorm on the point stream before the Gaussian heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out * self.weight if self.weight is not None else out


def rope_2d(h, w, head_dim, device, base=100.0):
    """Axial 2D RoPE table for an h*w token grid -> cos/sin [h*w, head_dim].
    head_dim is split half for rows, half for columns."""
    assert head_dim % 4 == 0
    d4 = head_dim // 4                                  # freqs per axis
    freqs = 1.0 / (base ** (torch.arange(d4, device=device).float() / d4))
    ang_h = torch.arange(h, device=device).float()[:, None] * freqs   # [h,d4]
    ang_w = torch.arange(w, device=device).float()[:, None] * freqs   # [w,d4]
    ang = torch.cat([ang_h[:, None, :].expand(h, w, d4),
                     ang_w[None, :, :].expand(h, w, d4)], dim=-1).reshape(h * w, -1)
    ang = ang.repeat_interleave(2, dim=-1)              # pairwise rotation layout
    return ang.cos(), ang.sin()                         # [h*w, head_dim] each


def apply_rope(x, cos, sin):
    """x [B,H,L,D] with cos/sin [L,D] (identity where cos=1,sin=0)."""
    x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + x2 * sin


class SwiGLU(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio * 2 / 3 / 64) * 64     # match 4x-GELU param budget
        self.w12 = nn.Linear(dim, hidden * 2)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * up)


class Modulation(nn.Module):
    """adaLN-Zero: temb -> (shift/scale/gate) x (attn, mlp). Zero-init => identity."""
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, temb):
        return self.lin(F.silu(temb))[:, None].chunk(6, dim=-1)  # each [B,1,D]


class JointBlockV2(nn.Module):
    """MM-DiT joint block: separate qkv/out/ffn weights per stream (points x, image c),
    one joint attention over the concatenated sequence."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        assert self.head_dim * n_heads == dim
        self.mod_x, self.mod_c = Modulation(dim), Modulation(dim)
        self.norm1_x, self.norm1_c = RMSNorm(dim, affine=False), RMSNorm(dim, affine=False)
        self.qkv_x, self.qkv_c = nn.Linear(dim, 3 * dim), nn.Linear(dim, 3 * dim)
        self.qnorm, self.knorm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.out_x, self.out_c = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm2_x, self.norm2_c = RMSNorm(dim, affine=False), RMSNorm(dim, affine=False)
        self.ff_x, self.ff_c = SwiGLU(dim), SwiGLU(dim)

    def _heads(self, t):                                # [B,L,3D] -> 3x [B,H,L,d]
        B, L, _ = t.shape
        q, k, v = t.view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return self.qnorm(q), self.knorm(k), v

    def forward(self, x, c, temb, rope=None):
        shift_x, scale_x, gate_x, shift2_x, scale2_x, gate2_x = self.mod_x(temb)
        shift_c, scale_c, gate_c, shift2_c, scale2_c, gate2_c = self.mod_c(temb)

        qx, kx, vx = self._heads(self.qkv_x(self.norm1_x(x) * (1 + scale_x) + shift_x))
        qc, kc, vc = self._heads(self.qkv_c(self.norm1_c(c) * (1 + scale_c) + shift_c))
        if rope is not None:                            # rotate image tokens only
            cos, sin = rope
            qc, kc = apply_rope(qc, cos, sin), apply_rope(kc, cos, sin)
        q = torch.cat([qx, qc], dim=2)
        k = torch.cat([kx, kc], dim=2)
        v = torch.cat([vx, vc], dim=2)
        o = F.scaled_dot_product_attention(q, k, v)
        B, _, L, _ = o.shape
        o = o.transpose(1, 2).reshape(B, L, -1)
        ox, oc = o[:, :x.shape[1]], o[:, x.shape[1]:]

        x = x + gate_x * self.out_x(ox)
        x = x + gate2_x * self.ff_x(self.norm2_x(x) * (1 + scale2_x) + shift2_x)
        c = c + gate_c * self.out_c(oc)
        c = c + gate2_c * self.ff_c(self.norm2_c(c) * (1 + scale2_c) + shift2_c)
        return x, c


class CrossBlockV2(nn.Module):
    """Refinement cross-attention (points query image tokens), modernized: RMSNorm,
    per-head QK RMSNorm, RoPE on image k, SwiGLU. O(N*M) like v1's CrossBlock."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.norm_q, self.norm_kv = RMSNorm(dim), RMSNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2 * dim)
        self.qnorm, self.knorm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim)
        self.norm_m = RMSNorm(dim)
        self.ff = SwiGLU(dim)

    def forward(self, q, kv, rope=None):
        B, N, D = q.shape
        M = kv.shape[1]
        qh = self.qnorm(self.to_q(self.norm_q(q)).view(B, N, self.n_heads, self.head_dim).transpose(1, 2))
        k, v = self.to_kv(self.norm_kv(kv)).view(B, M, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k = self.knorm(k)
        if rope is not None:
            cos, sin = rope
            k = apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(qh, k, v).transpose(1, 2).reshape(B, N, D)
        q = q + self.out(o)
        return q + self.ff(self.norm_m(q))


class MMDiTv2(nn.Module):
    def __init__(self, dim, n_layers, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.blocks = nn.ModuleList([JointBlockV2(dim, n_heads) for _ in range(n_layers)])
        self.norm_out = RMSNorm(dim)

    def forward(self, x, c, temb, rope=None):
        for blk in self.blocks:
            x, c = blk(x, c, temb, rope=rope)
        return self.norm_out(x), c
