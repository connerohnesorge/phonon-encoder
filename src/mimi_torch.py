"""PyTorch Mimi encoder that mirrors the xn-ptts Rust `MimiEncoder` exactly.

Tensor names follow the Rust loader (`ptts/src/mimi.rs`, `seanet.rs`, `conv.rs`,
`transformer.rs`, `resample.rs`), so a state dict saved under a `mimi_speaker.` prefix is
what `MimiEnc::load` reads.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncCfg:
    channels: int = 1
    sample_rate: int = 24000
    frame_rate: float = 12.5
    dimension: int = 512
    n_filters: int = 64
    n_residual_layers: int = 1
    ratios: list = field(default_factory=lambda: [8, 6, 5, 4])
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_base: int = 2
    compress: int = 2
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    layer_scale: float = 0.01
    context: int = 250
    max_period: float = 10000.0
    dim_feedforward: int = 2048
    # Output channels of the downsample conv: `dimension` for the speaker encoder (what the
    # Rust loader supports), the codec latent width for the main codec encoder.
    downsample_out: int = 512

    @property
    def hop(self):
        return math.prod(self.ratios)

    @property
    def frame_size(self):
        return round(self.sample_rate / self.frame_rate)


class SConv(nn.Module):
    """`StreamingConv1d` run on a whole signal from a fresh state: causal left padding of
    `(k - 1) * dilation + 1 - stride` frames, zeros or the first frame replicated."""

    def __init__(self, cin, cout, k, stride=1, dilation=1, bias=True, replicate=False):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation, bias=bias)
        self.pad = (k - 1) * dilation + 1 - stride
        self.replicate = replicate

    def forward(self, x):
        if self.pad > 0:
            mode = "replicate" if self.replicate else "constant"
            x = F.pad(x, (self.pad, 0), mode=mode)
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, dim, ks, dilation, compress):
        super().__init__()
        hidden = dim // compress
        self.block = nn.ModuleDict(
            {"1": SConv(dim, hidden, ks, dilation=dilation), "3": SConv(hidden, dim, 1)}
        )

    def forward(self, x):
        v = self.block["1"](F.elu(x))
        v = self.block["3"](F.elu(v))
        return x + v


class SEANetEncoder(nn.Module):
    def __init__(self, c: EncCfg):
        super().__init__()
        layers = {"0": SConv(c.channels, c.n_filters, c.kernel_size)}
        idx, mult = 1, 1
        for ratio in reversed(c.ratios):
            for j in range(c.n_residual_layers):
                layers[str(idx)] = ResBlock(
                    mult * c.n_filters, c.residual_kernel_size, c.dilation_base**j, c.compress
                )
                idx += 1
            layers[str(idx + 1)] = SConv(mult * c.n_filters, mult * c.n_filters * 2, ratio * 2, ratio)
            idx += 2
            mult *= 2
        layers[str(idx + 1)] = SConv(mult * c.n_filters, c.dimension, c.last_kernel_size)
        self.model = nn.ModuleDict(layers)
        self.order = sorted(layers, key=int)

    def forward(self, x):
        for i, k in enumerate(self.order):
            m = self.model[k]
            # an ELU sits before every downsample conv and the final conv (the odd gaps)
            if isinstance(m, SConv) and i > 0:
                x = F.elu(x)
            x = m(x)
        return x


def rope_interleaved(x, max_period):
    """`rope_i` on `[B, T, H, D]`: rotates the pairs (2i, 2i+1)."""
    b, t, h, d = x.shape
    half = d // 2
    inv = 1.0 / (max_period ** (torch.arange(half, device=x.device, dtype=torch.float32) * 2 / d))
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]
    xr = x.float().reshape(b, t, h, half, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1)
    return out.reshape(b, t, h, d).to(x.dtype)


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.heads = heads

    def forward(self, x, mask, max_period):
        b, t, d = x.shape
        hd = d // self.heads
        qkv = self.in_proj(x).reshape(b, t, 3, self.heads, hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = rope_interleaved(q, max_period), rope_interleaved(k, max_period)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class LayerScale(nn.Module):
    def __init__(self, d, init):
        super().__init__()
        self.scale = nn.Parameter(torch.full((d,), init))

    def forward(self, x):
        return x * self.scale


class Layer(nn.Module):
    def __init__(self, c: EncCfg):
        super().__init__()
        self.self_attn = Attn(c.d_model, c.num_heads)
        self.norm1 = nn.LayerNorm(c.d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(c.d_model, eps=1e-5)
        self.linear1 = nn.Linear(c.d_model, c.dim_feedforward, bias=False)
        self.linear2 = nn.Linear(c.dim_feedforward, c.d_model, bias=False)
        self.layer_scale_1 = LayerScale(c.d_model, c.layer_scale)
        self.layer_scale_2 = LayerScale(c.d_model, c.layer_scale)

    def forward(self, x, mask, max_period):
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x), mask, max_period))
        return x + self.layer_scale_2(self.linear2(F.gelu(self.linear1(self.norm2(x)))))


class Transformer(nn.Module):
    def __init__(self, c: EncCfg):
        super().__init__()
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.num_layers)])
        self.c = c

    def forward(self, x):  # [B, T, C]
        t = x.shape[1]
        i = torch.arange(t, device=x.device)
        ok = (i[None, :] <= i[:, None]) & (i[None, :] >= i[:, None] - self.c.context)
        mask = torch.zeros(t, t, device=x.device).masked_fill(~ok, float("-inf"))
        for layer in self.layers:
            x = layer(x, mask, self.c.max_period)
        return x


class MimiEncoder(nn.Module):
    def __init__(self, c: EncCfg):
        super().__init__()
        self.c = c
        self.encoder = SEANetEncoder(c)
        self.encoder_transformer = nn.Module()
        self.encoder_transformer.transformer = Transformer(c)
        stride = round(c.sample_rate / c.hop / c.frame_rate)
        self.downsample = nn.Module()
        self.downsample.conv = SConv(c.dimension, c.downsample_out, 2 * stride, stride, bias=False, replicate=True)

    def pad_input(self, x):
        n = x.shape[-1]
        fs = self.c.frame_size
        extra = (-n) % fs
        return F.pad(x, (0, extra)) if extra else x

    def seanet(self, x):
        """Audio `[B, 1, N]` -> SEANet features `[B, C, N / hop]` (the frozen-able front)."""
        return self.encoder(self.pad_input(x))

    def head(self, h):
        """SEANet features -> latents `[B, C', frames]`."""
        h = self.encoder_transformer.transformer(h.transpose(1, 2)).transpose(1, 2)
        return self.downsample.conv(h)

    def forward(self, x):
        return self.head(self.seanet(x))


def rust_state_dict(module: MimiEncoder, prefix: str):
    """The module's weights under the names the Rust loader reads."""
    out = {}
    for k, v in module.state_dict().items():
        out[prefix + k] = v.detach().float().contiguous()
    return out


def load_moshi(module: MimiEncoder, sd: dict, prefix: str = ""):
    """Loads moshi-named Mimi encoder weights (`kyutai/moshiko` tokenizer file)."""
    own = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        k = k[len(prefix):]
        if not (k.startswith("encoder.") or k.startswith("encoder_transformer.") or k.startswith("downsample.")):
            continue
        k = k.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")
        # moshi wraps every conv once more than the Rust loader does
        k = k.replace(".conv.conv.conv.", ".conv.conv.") if k.startswith("downsample.") else k.replace(".conv.conv.", ".conv.")
        own[k] = v.float()
    missing, unexpected = module.load_state_dict(own, strict=False)
    return missing, unexpected


def load_pocket(module: MimiEncoder, sd: dict, prefix: str = "mimi."):
    """Loads Pocket TTS (`pocket-tts` checkpoint) Mimi encoder weights."""
    own = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        k = k[len(prefix):]
        if not (k.startswith("encoder.") or k.startswith("encoder_transformer.") or k.startswith("downsample.")):
            continue
        own[k] = v.float()
    return module.load_state_dict(own, strict=False)
