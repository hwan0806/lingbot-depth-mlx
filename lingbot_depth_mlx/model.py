"""MLX-native LingBot-Depth v0.5 inference model."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .preprocessing import exact_scale, load_depth, load_rgb, preprocess_encoder_inputs


@lru_cache(maxsize=16)
def cubic_matrix(source: int, target: int, scale: float):
    matrix = np.zeros((target, source), np.float32)
    for output in range(target):
        position = (output + 0.5) / scale - 0.5
        base = math.floor(position)
        for index in range(base - 1, base + 3):
            distance = abs(position - index)
            weight = (1.25 * distance**3 - 2.25 * distance**2 + 1) if distance <= 1 else (-0.75 * distance**3 + 3.75 * distance**2 - 6 * distance + 3) if distance < 2 else 0
            matrix[output, min(max(index, 0), source - 1)] += weight
    return mx.array(matrix)


class PatchEmbed(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1024, 14, stride=14)

    def __call__(self, value):
        value = self.proj(value)
        return value.reshape(value.shape[0], -1, value.shape[-1])


class LayerScale(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.gamma = mx.ones((dims,))

    def __call__(self, value):
        return value * self.gamma


class MLP(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, dims * 4)
        self.fc2 = nn.Linear(dims * 4, dims)

    def __call__(self, value):
        return self.fc2(nn.gelu(self.fc1(value)))


class Attention(nn.Module):
    def __init__(self, dims: int = 1024, heads: int = 16):
        super().__init__()
        self.qkv = nn.Linear(dims, dims * 3)
        self.proj = nn.Linear(dims, dims)
        self.heads = heads
        self.scale = (dims // heads) ** -0.5

    def __call__(self, value):
        batch, length, dims = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, dims // self.heads)
        q, k, v = mx.split(qkv.transpose(2, 0, 3, 1, 4), 3, axis=0)
        attended = mx.fast.scaled_dot_product_attention(q[0], k[0], v[0], scale=self.scale)
        return self.proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, dims))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1024, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale(1024)
        self.norm2 = nn.LayerNorm(1024, eps=1e-6)
        self.mlp = MLP(1024)
        self.ls2 = LayerScale(1024)

    def __call__(self, value):
        value = value + self.ls1(self.attn(self.norm1(value)))
        return value + self.ls2(self.mlp(self.norm2(value)))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = mx.zeros((1, 1, 1024))
        self.pos_embed = mx.zeros((1, 1370, 1024))
        self.mask_token = mx.zeros((1, 1024))
        self.patch_embed = PatchEmbed(3)
        self.depth_patch_embed = PatchEmbed(1)
        self.blocks = [TransformerBlock() for _ in range(24)]
        self.norm = nn.LayerNorm(1024, eps=1e-6)

    def position(self, rows: int, cols: int):
        grid = self.pos_embed[:, 1:].reshape(1, 37, 37, 1024)
        height_scale, width_scale = (rows + 0.1) / 37, (cols + 0.1) / 37
        grid = mx.einsum("oh,bhwc->bowc", cubic_matrix(37, rows, height_scale), grid)
        grid = mx.einsum("pw,bowc->bopc", cubic_matrix(37, cols, width_scale), grid)
        return grid.reshape(1, rows * cols, 1024)

    def __call__(self, image, depth, valid, grid):
        rows, cols = grid
        image_tokens = self.patch_embed(image)
        depth_tokens = self.depth_patch_embed(depth)
        position = self.position(rows, cols)
        image_tokens = image_tokens + position + 1
        depth_tokens = depth_tokens + position + 2
        # Upstream selects tokens from remapped depth, so log(1m) is a masked zero.
        token_valid = (depth != 0) & (depth >= -9.5) & (depth <= 200)
        patch_valid = mx.any(token_valid.reshape(depth.shape[0], rows, 14, cols, 14, 1), axis=(2, 4, 5))
        cls = self.cls_token + self.pos_embed[:, :1]
        tokens = []
        for i in range(image.shape[0]):
            indices = mx.array(np.flatnonzero(np.array(patch_valid[i])))
            tokens.append(mx.concatenate((cls, image_tokens[i : i + 1], mx.take(depth_tokens[i], indices, axis=0)[None]), axis=1))
        for block in self.blocks:
            tokens = [block(value) for value in tokens]
        outputs = [self.norm(value) for value in tokens]
        features = mx.concatenate([value[:, 1 : 1 + rows * cols] for value in outputs], axis=0)
        classes = mx.concatenate([value[:, 0] for value in outputs], axis=0)
        return features.reshape(image.shape[0], rows, cols, 1024), classes


class RGBDEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_mean = mx.zeros((1, 1, 1, 3))
        self.image_std = mx.ones((1, 1, 1, 3))
        self.backbone = Backbone()
        self.output_projections = [nn.Conv2d(1024, 1024, 1)]

    def __call__(self, image, depth, valid, grid):
        features, cls = self.backbone(image, depth, valid, grid)
        return self.output_projections[0](features), cls


class ReplicateConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
        self.edge_padding = kernel_size // 2

    def __call__(self, value):
        pad = self.edge_padding
        return super().__call__(mx.pad(value, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge"))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity(), nn.ReLU(), ReplicateConv2d(channels, channels),
            nn.Identity(), nn.ReLU(), ReplicateConv2d(channels, channels),
        )

    def __call__(self, value):
        return value + self.layers(value)


def resampler(in_channels: int, out_channels: int, kind: str):
    if kind == "conv_transpose":
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2), ReplicateConv2d(out_channels, out_channels))
    if kind == "bilinear":
        return nn.Sequential(nn.Upsample(2, mode="linear", align_corners=False), ReplicateConv2d(in_channels, out_channels))
    raise ValueError(f"unsupported resampler: {kind}")


class ConvStack(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dims = config["dim_res_blocks"]
        inputs = config["dim_in"] if isinstance(config["dim_in"], list) else repeat(config["dim_in"])
        outputs = config["dim_out"] if isinstance(config["dim_out"], list) else repeat(config["dim_out"])
        counts = config["num_res_blocks"] if isinstance(config["num_res_blocks"], list) else repeat(config["num_res_blocks"])
        self.input_blocks = [nn.Conv2d(source, target, 1) if source is not None else nn.Identity() for source, target in zip(inputs, dims)]
        self.resamplers = [resampler(source, target, kind) for source, target, kind in zip(dims[:-1], dims[1:], config["resamplers"])]
        self.res_blocks = [nn.Sequential(*(ResidualConvBlock(channels) for _ in range(count))) for channels, count in zip(dims, counts)]
        self.output_blocks = [nn.Conv2d(source, target, 1) if target is not None else nn.Identity() for source, target in zip(dims, outputs)]

    def __call__(self, features):
        outputs = []
        for index, block in enumerate(self.res_blocks):
            feature = self.input_blocks[index](features[index])
            value = feature if index == 0 else value + feature
            value = block(value)
            outputs.append(self.output_blocks[index](value))
            if index < len(self.resamplers):
                value = self.resamplers[index](value)
        return outputs


@lru_cache(maxsize=80)
def normalized_uv(height: int, width: int, aspect: float):
    span_x = aspect / math.sqrt(1 + aspect**2)
    span_y = 1 / math.sqrt(1 + aspect**2)
    u = mx.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    v = mx.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
    u, v = mx.meshgrid(u, v)
    return mx.stack((u, v), axis=-1)[None]


class LingBotDepth(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.neck = ConvStack(config["neck"])
        self.depth_head = ConvStack(config["depth_head"])
        self.mask_head = ConvStack(config["mask_head"])
        self.config = config

    @classmethod
    def from_pretrained(cls, directory: str | Path):
        directory = Path(directory)
        from .weights import verify_checkpoint

        verify_checkpoint(directory)
        model = cls(json.loads((directory / "config.json").read_text()))
        if (directory / "precision.json").exists():
            precision = json.loads((directory / "precision.json").read_text())
            if precision.get("encoder") != "bfloat16" or precision.get("decoder") != "float32":
                raise ValueError(f"unsupported precision config: {precision}")
            model.use_bfloat16_encoder()
        model.load_weights(str(directory / "weights.safetensors"), strict=True)
        return model

    def __call__(self, image, depth, valid, grid, original_size):
        batch = image.shape[0]
        rows, cols = grid
        aspect = original_size[1] / original_size[0]
        features, cls = self.encoder(image, depth, valid, grid)
        features, cls = features.astype(mx.float32), cls.astype(mx.float32)
        features = features + cls[:, None, None, :]
        pyramid = []
        for level in range(5):
            uv = mx.broadcast_to(normalized_uv(rows * 2**level, cols * 2**level, aspect), (batch, rows * 2**level, cols * 2**level, 2))
            pyramid.append(mx.concatenate((features, uv), axis=-1) if level == 0 else uv)
        features = self.neck(pyramid)
        depth = self.depth_head(features)[-1]
        mask = self.mask_head(features)[-1]
        scale = (exact_scale(depth.shape[1], original_size[0]), exact_scale(depth.shape[2], original_size[1]))
        depth = nn.Upsample(scale, mode="linear", align_corners=False)(depth)[..., 0]
        mask = nn.Upsample(scale, mode="linear", align_corners=False)(mask)[..., 0]
        if self.config["remap_depth_out"] == "exp":
            depth = mx.exp(depth)
        elif self.config["remap_depth_out"] != "linear":
            raise ValueError(f"unsupported depth output remap: {self.config['remap_depth_out']}")
        return {"depth": depth, "mask_probability": mx.sigmoid(mask), "mask": mx.sigmoid(mask) > 0.5}

    def infer(self, rgb: np.ndarray, depth_m: np.ndarray, num_tokens: int = 1200):
        image, depth, valid, grid = preprocess_encoder_inputs(rgb, depth_m, num_tokens)
        dtype = self.encoder.backbone.patch_embed.proj.weight.dtype
        image, depth = image.astype(dtype), depth.astype(dtype)
        return self(image, depth, valid, grid, depth_m.shape)

    def use_bfloat16_encoder(self):
        self.encoder.set_dtype(mx.bfloat16)
        return self

    def infer_files(self, rgb_path: str | Path, depth_path: str | Path, depth_unit: str = "mm", num_tokens: int = 1200):
        return self.infer(load_rgb(rgb_path), load_depth(depth_path, depth_unit), num_tokens)
