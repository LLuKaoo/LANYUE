"""Five-modal LANYUE MAE with shared HiViT and spherical geolocation.

The patch embeddings, encoder, and reconstruction math are adapted from the
original lunar pretraining implementation and HiViT. See LICENSE and NOTICE.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_

from .hivit import PatchEmbed, PatchMerge, BlockWithRPE
from .pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid
from ..data.shards import CE2_ANCHOR_V2_LAYOUTS


def get_2d_sincos_pos_embed_hw(embed_dim, grid_hw, cls_token=False):
    """
    grid_hw: int or (height, width)
    return: [height*width, embed_dim]
    """
    if isinstance(grid_hw, int):
        return get_2d_sincos_pos_embed(embed_dim, grid_hw, cls_token=cls_token)

    grid_h, grid_w = grid_hw
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, len(grid_h), len(grid_w)])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class LanyueMAE(nn.Module):
    """Reconstruct masked, valid pixels for each available CE2-anchor modality.

    Defaults reproduce the five-modal model's base architecture with v2 canvas
    sizes. Smaller dimensions/depths can be supplied for smoke tests. Coordinates
    are [longitude, latitude] in degrees; missing coordinates use a learned vector.
    """

    modality_layouts = CE2_ANCHOR_V2_LAYOUTS
    accepts_nac_only = False

    def __init__(
        self,
        embed_dim=512,
        depths=(2, 2, 20),
        num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=6,
        decoder_num_heads=16,
        modality_decoder_depths=None,
        modality_img_sizes=None,
        loss_weights=None,
        norm_pix_loss=True,
        use_checkpoint=False,
        saliency_masking=True,
        saliency_mask_prob=0.5,
        saliency_mask_strength=0.7,
        saliency_mask_granularity="sample",
        metadata_drop_prob=0.7,
        scale_drop_prob=0.0,
        modality_embed_drop_prob=0.0,
        geo_fourier_num_bands=32,
        geo_fourier_min_wavelength=1.0e-4,
        geo_fourier_max_wavelength=2.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        if len(depths) != 3 or any(int(d) < 1 for d in depths):
            raise ValueError("HiViT requires three positive stage depths.")
        if (
            embed_dim % 16
            or embed_dim % num_heads
            or decoder_embed_dim % 4
            or decoder_embed_dim % decoder_num_heads
        ):
            raise ValueError(
                "Dimensions must support HiViT stages, attention heads, and 2D positions."
            )
        if decoder_depth < 1 or not 0 <= drop_path_rate < 1:
            raise ValueError("Invalid decoder depth or drop-path rate.")
        self.modalities = tuple(self.modality_layouts)
        sizes = {
            name: layout.model_size for name, layout in self.modality_layouts.items()
        }
        if modality_img_sizes is not None:
            if set(modality_img_sizes) != set(self.modalities):
                raise ValueError(
                    "modality_img_sizes must specify all configured modalities."
                )
            sizes = {
                name: self._to_2tuple(size) for name, size in modality_img_sizes.items()
            }
        self.modality_img_sizes = sizes
        self.modality_input_sizes = dict(sizes)
        self.modality_patch_sizes = {
            name: layout.patch_size for name, layout in self.modality_layouts.items()
        }
        self.modality_physical_scales = {
            name: layout.meters_per_pixel
            for name, layout in self.modality_layouts.items()
        }
        self.modality_chans = {name: 1 for name in self.modalities}
        self.num_visible_tokens = {name: None for name in self.modalities}
        defaults = dict(
            ce2_dom=1.0, ce2_dem=0.7, nasa_dom=0.5, nasa_dem=0.5, minirf_sar=1.0
        )
        if "nac" in self.modalities:
            defaults["nac"] = 1.0
        if loss_weights is not None:
            if set(loss_weights) - set(self.modalities):
                raise ValueError("Unknown modality in loss_weights.")
            defaults.update(loss_weights)
        self.loss_weights = {name: float(value) for name, value in defaults.items()}
        if any(not math.isfinite(w) or w <= 0 for w in self.loss_weights.values()):
            raise ValueError("Reconstruction weights must be finite and positive.")
        self.norm_pix_loss = bool(norm_pix_loss)
        self.saliency_masking = bool(saliency_masking)
        self.saliency_mask_prob = float(saliency_mask_prob)
        self.saliency_mask_strength = float(saliency_mask_strength)
        self.saliency_mask_granularity = saliency_mask_granularity
        if saliency_mask_granularity not in {"sample", "patch"}:
            raise ValueError("saliency_mask_granularity must be sample or patch.")
        self.metadata_drop_prob = float(metadata_drop_prob)
        self.scale_drop_prob = float(scale_drop_prob)
        self.modality_embed_drop_prob = float(modality_embed_drop_prob)
        for value in (
            self.saliency_mask_prob,
            self.metadata_drop_prob,
            self.scale_drop_prob,
            self.modality_embed_drop_prob,
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    "Dropout and masking probabilities must lie in [0, 1]."
                )
        if (
            not math.isfinite(self.saliency_mask_strength)
            or self.saliency_mask_strength < 0
        ):
            raise ValueError("saliency_mask_strength must be finite and nonnegative.")
        self.geo_fourier_num_bands = int(geo_fourier_num_bands)
        self.geo_fourier_min_wavelength = float(geo_fourier_min_wavelength)
        self.geo_fourier_max_wavelength = float(geo_fourier_max_wavelength)
        if (
            self.geo_fourier_num_bands < 1
            or not 0 < self.geo_fourier_min_wavelength < self.geo_fourier_max_wavelength
        ):
            raise ValueError(
                "Require positive Fourier bands and increasing positive wavelengths."
            )
        self.use_physical_scale_embed = True
        self.use_spherical_fourier_geo_embed = True
        self.use_checkpoint = bool(use_checkpoint)
        self.ape = True
        self.num_features = embed_dim
        self.num_main_blocks = int(depths[-1])
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        stem_dim = embed_dim // 4
        for name, size in sizes.items():
            patch = self.modality_patch_sizes[name]
            if min(size) < patch or any(side % patch for side in size):
                raise ValueError(
                    f"{name}: canvas must be divisible by patch size {patch}."
                )
        self.patch_embeds = nn.ModuleDict(
            {
                name: PatchEmbed(
                    img_size=sizes[name],
                    patch_size=self.modality_patch_sizes[name],
                    in_chans=1,
                    embed_dim=stem_dim,
                    norm_layer=norm_layer,
                )
                for name in self.modalities
            }
        )
        self.num_patches = {
            name: module.num_patches for name, module in self.patch_embeds.items()
        }
        self.patches_resolution = {
            name: tuple(module.patches_resolution)
            for name, module in self.patch_embeds.items()
        }
        self.absolute_pos_embeds = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.zeros(1, count, embed_dim), requires_grad=False
                )
                for name, count in self.num_patches.items()
            }
        )
        self.modality_embed = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, 1, embed_dim))
                for name in self.modalities
            }
        )
        self.unknown_modality_embed = (
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            if self.modality_embed_drop_prob > 0
            else None
        )
        self.physical_scale_embed = nn.Sequential(
            nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.unknown_scale_embed = (
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            if self.scale_drop_prob > 0
            else None
        )
        wavelengths = torch.logspace(
            math.log10(self.geo_fourier_min_wavelength),
            math.log10(self.geo_fourier_max_wavelength),
            self.geo_fourier_num_bands,
        )
        self.register_buffer("geo_fourier_wavelengths", wavelengths, persistent=True)
        self.spherical_geo_embed = nn.Sequential(
            nn.Linear(6 * self.geo_fourier_num_bands, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.unknown_geo_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList()
        max_grid = max(max(grid) for grid in self.patches_resolution.values())
        dpr = iter(
            torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])).tolist()
        )
        dim = stem_dim
        for index, depth in enumerate(depths):
            main = index == len(depths) - 1
            for _ in range(int(depth) * (1 if main else 2)):
                self.blocks.append(
                    BlockWithRPE(
                        max_grid,
                        dim,
                        num_heads if main else 0,
                        4.0 if main else 3.0,
                        qkv_bias=True,
                        drop_path=next(dpr),
                        rpe=False,
                        norm_layer=norm_layer,
                    )
                )
            if not main:
                self.blocks.append(PatchMerge(dim, norm_layer))
                dim *= 2
        self.norm = norm_layer(embed_dim)
        self.decoder_embeds = nn.ModuleDict(
            {name: nn.Linear(embed_dim, decoder_embed_dim) for name in self.modalities}
        )
        self.mask_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
                for name in self.modalities
            }
        )
        self.decoder_pos_embeds = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.zeros(1, count, decoder_embed_dim), requires_grad=False
                )
                for name, count in self.num_patches.items()
            }
        )
        decoder_depths = {name: decoder_depth for name in self.modalities}
        decoder_depths["minirf_sar"] = 2
        if modality_decoder_depths is not None:
            if set(modality_decoder_depths) - set(self.modalities):
                raise ValueError("Unknown modality in modality_decoder_depths.")
            decoder_depths.update(modality_decoder_depths)
        if any(int(d) < 1 for d in decoder_depths.values()):
            raise ValueError("Decoder depths must be positive.")
        self.decoder_blocks = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        BlockWithRPE(
                            max_grid,
                            decoder_embed_dim,
                            decoder_num_heads,
                            4.0,
                            qkv_bias=True,
                            rpe=False,
                            norm_layer=norm_layer,
                        )
                        for _ in range(int(decoder_depths[name]))
                    ]
                )
                for name in self.modalities
            }
        )
        self.decoder_norms = nn.ModuleDict(
            {name: norm_layer(decoder_embed_dim) for name in self.modalities}
        )
        self.decoder_preds = nn.ModuleDict(
            {
                name: nn.Linear(decoder_embed_dim, self.modality_patch_sizes[name] ** 2)
                for name in self.modalities
            }
        )
        for name in self.modalities:
            pos = get_2d_sincos_pos_embed_hw(embed_dim, self.patches_resolution[name])
            self.absolute_pos_embeds[name].data.copy_(
                torch.from_numpy(pos).float().unsqueeze(0)
            )
            pos = get_2d_sincos_pos_embed_hw(
                decoder_embed_dim, self.patches_resolution[name]
            )
            self.decoder_pos_embeds[name].data.copy_(
                torch.from_numpy(pos).float().unsqueeze(0)
            )
            nn.init.normal_(self.mask_tokens[name], std=0.02)
            nn.init.normal_(self.modality_embed[name], std=0.02)
        self.apply(self._init_weights)
        nn.init.zeros_(self.spherical_geo_embed[-1].weight)
        nn.init.zeros_(self.spherical_geo_embed[-1].bias)
        self.last_losses = {}

    def no_weight_decay(self):
        return {
            name for name, _ in self.named_parameters() if name.startswith("unknown_")
        }

    def saliency_patch_scores(self, imgs, valid_mask, valid_patch_mask, name):
        if not self.saliency_masking:
            return None
        gradient = self._sobel_gradient_magnitude(imgs)
        scores = self.patchify_modality(gradient, name).mean(dim=-1)
        return self._normalize_patch_scores(scores, valid_patch_mask > 0)

    def forward_loss_modality(self, imgs, pred, mask, valid_mask, name):
        target = self.patchify_modality(imgs, name).float()
        valid_pixels = self.patchify_valid_pixels(valid_mask, name).float()
        if self.norm_pix_loss:
            count = valid_pixels.sum(dim=-1, keepdim=True)
            mean = (target * valid_pixels).sum(dim=-1, keepdim=True) / count.clamp_min(1)
            variance = ((target - mean).square() * valid_pixels).sum(
                dim=-1, keepdim=True
            ) / (count - 1).clamp_min(1)
            target = (target - mean) / (variance + 1.0e-6).sqrt()
        return self._masked_patch_mse(
            pred, target, mask, valid_pixels
        )

    def forward(
        self, batch_dict, mask_ratio=0.75, active_modalities=None, return_latents=False
    ):
        if not 0 <= float(mask_ratio) < 1:
            raise ValueError("mask_ratio must lie in [0, 1).")
        active = self.get_active_modalities(batch_dict, active_modalities)
        if "ce2_dom" not in active and not (
            self.accepts_nac_only and active == ["nac"]
        ):
            raise ValueError(
                "Every pretraining sample must include the ce2_dom anchor."
            )
        metadata_keep = self._sample_metadata_keep_mask(batch_dict, active)
        tokenized = {
            name: self.tokenize_modality(
                batch_dict, name, mask_ratio, metadata_keep_mask=metadata_keep
            )
            for name in active
        }
        latents = self.forward_shared_encoder(tokenized, active)
        predictions = {
            name: self.forward_decoder(
                name, latents[name], tokenized[name]["ids_restore"]
            )
            for name in active
        }
        losses = {
            name: self.forward_loss_modality(
                tokenized[name]["padded"],
                predictions[name],
                tokenized[name]["mask"],
                tokenized[name]["valid_mask"],
                name,
            )
            for name in active
        }
        weight_sum = sum(self.loss_weights[name] for name in active)
        total = (
            sum(losses[name] * self.loss_weights[name] for name in active) / weight_sum
        )
        self.last_losses = {name: loss.detach() for name, loss in losses.items()}
        masks = {name: tokenized[name]["mask"] for name in active}
        if return_latents:
            return total, predictions, masks, latents
        return total, predictions, masks

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _to_2tuple(self, size):
        return (size, size) if isinstance(size, int) else tuple(size)

    def _mask_ratio_for(self, mask_ratio, name):
        if isinstance(mask_ratio, dict):
            return float(mask_ratio.get(name, mask_ratio.get("default", 0.75)))
        return float(mask_ratio)

    def _num_keep_for(self, name, mask_ratio):
        num_keep = self.num_visible_tokens.get(name)
        if num_keep is not None:
            return min(int(num_keep), self.num_patches[name])
        return max(
            1,
            int(
                self.num_patches[name] * (1.0 - self._mask_ratio_for(mask_ratio, name))
            ),
        )

    def _condition_embedding(self, known, unknown, batch_size, drop_prob):
        """Replace one whole conditioning field per sample with learned unknown.

        The mask is shared by every token of the same sample/modality.  This is
        field dropout rather than element-wise ``nn.Dropout`` and therefore
        preserves the semantic integrity of each metadata vector.
        """
        if unknown is None or not self.training or drop_prob <= 0.0:
            return known
        keep = torch.rand(batch_size, device=known.device) >= drop_prob
        known = known.expand(batch_size, -1, -1)
        unknown = unknown.expand(batch_size, -1, -1)
        return torch.where(keep[:, None, None], known, unknown)

    def _modality_embed_for(self, name, batch_size):
        return self._condition_embedding(
            self.modality_embed[name],
            self.unknown_modality_embed,
            batch_size,
            self.modality_embed_drop_prob,
        )

    def _physical_scale_embed_for(self, name, batch_size=1):
        if not self.use_physical_scale_embed:
            return 0.0
        meters_per_pixel = max(float(self.modality_physical_scales[name]), 1e-06)
        meters_per_token = max(
            meters_per_pixel * float(self.modality_patch_sizes[name]), 1e-06
        )
        scale = (
            self.modality_embed[name]
            .new_tensor([math.log(meters_per_pixel), math.log(meters_per_token)])
            .view(1, 1, 2)
        )
        return self._condition_embedding(
            self.physical_scale_embed(scale),
            self.unknown_scale_embed,
            batch_size,
            self.scale_drop_prob,
        )

    @staticmethod
    def _center_lonlat_value_for(batch_dict, name):
        value = batch_dict.get(f"{name}_center_lonlat")
        if value is None:
            value = batch_dict.get("center_lonlat")
        return value

    def _sample_metadata_keep_mask(self, batch_dict, active_modalities):
        """Sample one shared, per-sample geolocation dropout mask.

        Sharing the mask across modalities prevents an aligned scene from
        exposing its location through one sensor while hiding it from another.
        """
        if not self.use_spherical_fourier_geo_embed:
            return None
        reference = batch_dict[active_modalities[0]]
        batch_size = reference.shape[0]
        if not self.training or self.metadata_drop_prob <= 0.0:
            return torch.ones(batch_size, dtype=torch.bool, device=reference.device)
        return (
            torch.rand(batch_size, device=reference.device) >= self.metadata_drop_prob
        )

    def _spherical_fourier_features(self, center_lonlat):
        """Encode lunar unit-sphere Cartesian coordinates at multiple scales."""
        value = torch.as_tensor(center_lonlat, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(
                f"center_lonlat must have shape [batch, 2], got {tuple(value.shape)}."
            )
        finite = torch.isfinite(value).all(dim=1)
        finite_lats = value[finite, 1]
        if finite_lats.numel() and torch.any(finite_lats.abs() > 90.0):
            raise ValueError("Finite latitude values must lie in [-90, 90] degrees.")
        safe = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        lon = torch.remainder(safe[:, 0] + 180.0, 360.0) - 180.0
        lat = safe[:, 1]
        lon_rad = torch.deg2rad(lon)
        lat_rad = torch.deg2rad(lat)
        cos_lat = torch.cos(lat_rad)
        xyz = torch.stack(
            (
                cos_lat * torch.cos(lon_rad),
                cos_lat * torch.sin(lon_rad),
                torch.sin(lat_rad),
            ),
            dim=-1,
        )
        wavelengths = self.geo_fourier_wavelengths.to(
            device=xyz.device, dtype=xyz.dtype
        )
        phase = xyz.unsqueeze(-1) * (2.0 * math.pi / wavelengths)
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return (features.flatten(start_dim=1), finite)

    def _spherical_geo_embed_for(self, batch_dict, name, metadata_keep_mask=None):
        if not self.use_spherical_fourier_geo_embed:
            return 0.0
        batch_size = batch_dict[name].shape[0]
        unknown = self.unknown_geo_embed.expand(batch_size, -1, -1)
        value = self._center_lonlat_value_for(batch_dict, name)
        if value is None:
            return unknown
        value = torch.as_tensor(
            value, device=batch_dict[name].device, dtype=torch.float32
        )
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"center_lonlat batch dimension does not match image batch: {value.shape[0]} vs {batch_size}."
            )
        features, finite = self._spherical_fourier_features(value)
        geographic = self.spherical_geo_embed(features).unsqueeze(1)
        known = finite
        if metadata_keep_mask is not None:
            known = known & metadata_keep_mask.to(device=known.device)
        return torch.where(known[:, None, None], geographic, unknown)

    def get_active_modalities(self, batch_dict, active_modalities=None):
        if active_modalities is None:
            active = [name for name in self.modalities if name in batch_dict]
        else:
            requested = list(active_modalities)
            unsupported = [name for name in requested if name not in self.modalities]
            if unsupported:
                raise ValueError(
                    f"Unsupported modalities: {unsupported}. Supported: {self.modalities}"
                )
            missing = [name for name in requested if name not in batch_dict]
            if missing:
                raise KeyError(
                    f"Requested modalities missing from batch_dict: {missing}"
                )
            requested_set = set(requested)
            active = [name for name in self.modalities if name in requested_set]
        if len(active) == 0:
            raise ValueError("batch_dict must contain at least one supported modality.")
        return active

    def pad_modality(self, x, name):
        input_size = self._to_2tuple(self.modality_input_sizes[name])
        padded_size = self._to_2tuple(self.modality_img_sizes[name])
        input_h, input_w = tuple(x.shape[-2:])
        allow_variable_input = self._to_2tuple(input_size) != self._to_2tuple(
            padded_size
        )
        if (input_h, input_w) not in (input_size, padded_size) and (
            not allow_variable_input
        ):
            raise ValueError(
                f"{name}: expected input size {input_size} or padded size {padded_size}, but got {(input_h, input_w)}."
            )
        B, _, H, W = x.shape
        target_h, target_w = padded_size
        if H > target_h or W > target_w:
            raise ValueError(
                f"{name}: input {(H, W)} is larger than padded size {padded_size}."
            )
        valid_mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        pad_h = target_h - H
        pad_w = target_w - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            valid_mask = F.pad(valid_mask, (0, pad_w, 0, pad_h))
        return (x, valid_mask)

    def patchify_modality(self, imgs, name):
        p = self.modality_patch_sizes[name]
        assert imgs.shape[2] % p == 0
        assert imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * p * c))
        return x

    def unpatchify_modality(self, x, name):
        p = self.modality_patch_sizes[name]
        h, w = self.patches_resolution[name]
        c = self.modality_chans[name]
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def patchify_valid_pixels(self, valid_mask, name):
        p = self.modality_patch_sizes[name]
        h = valid_mask.shape[2] // p
        w = valid_mask.shape[3] // p
        x = valid_mask.reshape(valid_mask.shape[0], 1, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(valid_mask.shape[0], h * w, p * p)
        channels = int(self.modality_chans[name])
        return x.repeat_interleave(channels, dim=-1)

    def patchify_valid_mask(self, valid_mask, name):
        return self.patchify_valid_pixels(valid_mask, name).mean(dim=-1)

    def _masked_patch_mse(self, pred, target, mask, valid_pixel_mask):
        pred = pred.float()
        target = target.float()
        mask = mask.float().unsqueeze(-1)
        valid_pixel_mask = valid_pixel_mask.float()
        loss = (pred - target) ** 2
        loss_mask = mask * valid_pixel_mask
        return (loss * loss_mask).sum() / (loss_mask.sum() + 1e-06)

    def _sobel_gradient_magnitude(self, imgs):
        channels = imgs.shape[1]
        kernel_x = (
            imgs.new_tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).view(1, 1, 3, 3)
            / 8.0
        )
        kernel_y = (
            imgs.new_tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3)
            / 8.0
        )
        padded = F.pad(imgs, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, kernel_x.repeat(channels, 1, 1, 1), groups=channels)
        grad_y = F.conv2d(padded, kernel_y.repeat(channels, 1, 1, 1), groups=channels)
        return (grad_x.square() + grad_y.square() + 1e-12).sqrt()

    def _normalize_patch_scores(self, scores, valid_bool):
        scores = scores.masked_fill(~valid_bool, 0.0)
        min_scores = scores.masked_fill(~valid_bool, float("inf")).amin(
            dim=1, keepdim=True
        )
        max_scores = scores.masked_fill(~valid_bool, float("-inf")).amax(
            dim=1, keepdim=True
        )
        min_scores = torch.where(
            torch.isfinite(min_scores), min_scores, torch.zeros_like(min_scores)
        )
        max_scores = torch.where(torch.isfinite(max_scores), max_scores, min_scores)
        denom = (max_scores - min_scores).clamp_min(1e-06)
        return ((scores - min_scores) / denom).masked_fill(~valid_bool, 0.0)

    def masking_id(
        self,
        name,
        batch_size,
        mask_ratio,
        valid_patch_mask=None,
        saliency_patch_scores=None,
    ):
        N, L = (batch_size, self.num_patches[name])
        device = self.modality_embed[name].device
        if valid_patch_mask is None:
            valid_patch_mask = torch.ones(N, L, device=device)
        else:
            valid_patch_mask = valid_patch_mask.to(device=device)
        valid_bool = valid_patch_mask > 0.0
        valid_counts = valid_bool.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError(f"{name}: valid_patch_mask contains an empty sample.")
        requested_keep = self._num_keep_for(name, mask_ratio)
        len_keep = min(requested_keep, int(valid_counts.min().item()))
        noise = torch.rand(N, L, device=device)
        if (
            saliency_patch_scores is not None
            and self.saliency_mask_prob > 0.0
            and (self.saliency_mask_strength > 0.0)
        ):
            saliency_patch_scores = saliency_patch_scores.to(
                device=device, dtype=noise.dtype
            )
            saliency_patch_scores = self._normalize_patch_scores(
                saliency_patch_scores, valid_bool
            )
            gate_length = L if self.saliency_mask_granularity == "patch" else 1
            use_saliency = (
                torch.rand(N, gate_length, device=device) < self.saliency_mask_prob
            ).to(noise.dtype)
            noise = (
                noise
                - use_saliency * self.saliency_mask_strength * saliency_patch_scores
            )
        noise = noise.masked_fill(~valid_bool, float("inf"))
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, L], device=device)
        mask[:, : ids_keep.size(1)] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = torch.where(valid_bool, mask, torch.ones_like(mask))
        return (ids_keep, ids_restore, mask)

    def tokenize_modality(self, batch_dict, name, mask_ratio, metadata_keep_mask=None):
        if f"{name}_valid_mask" in batch_dict:
            padded = batch_dict[name]
            valid_mask = batch_dict[f"{name}_valid_mask"].to(
                device=padded.device, dtype=padded.dtype
            )
            expected_size = self._to_2tuple(self.modality_img_sizes[name])
            if tuple(padded.shape[-2:]) != expected_size:
                raise ValueError(
                    f"{name}: expected pre-padded input size {expected_size}, but got {tuple(padded.shape[-2:])}."
                )
            if tuple(valid_mask.shape[-2:]) != expected_size:
                raise ValueError(
                    f"{name}_valid_mask: expected size {expected_size}, but got {tuple(valid_mask.shape[-2:])}."
                )
        else:
            padded, valid_mask = self.pad_modality(batch_dict[name], name)
        valid_patch_mask = self.patchify_valid_mask(valid_mask, name)
        saliency_scores = self.saliency_patch_scores(
            padded, valid_mask, valid_patch_mask, name
        )
        ids_keep, ids_restore, mask = self.masking_id(
            name,
            padded.size(0),
            mask_ratio,
            valid_patch_mask=valid_patch_mask,
            saliency_patch_scores=saliency_scores,
        )
        tokens = self.patch_embeds[name](padded)
        visible_tokens = torch.gather(
            tokens,
            dim=1,
            index=ids_keep[:, :, None, None, None].expand(-1, -1, *tokens.shape[2:]),
        )
        return {
            "padded": padded,
            "valid_mask": valid_mask,
            "valid_patch_mask": valid_patch_mask,
            "saliency_scores": saliency_scores,
            "visible_tokens": visible_tokens,
            "ids_keep": ids_keep,
            "ids_restore": ids_restore,
            "mask": mask,
            "num_visible": ids_keep.size(1),
            "spherical_geo_embed": self._spherical_geo_embed_for(
                batch_dict, name, metadata_keep_mask=metadata_keep_mask
            ),
        }

    def forward_shared_encoder(self, tokenized, active_modalities):
        xs = []
        keep_lengths = {}
        for name in active_modalities:
            xs.append(tokenized[name]["visible_tokens"])
            keep_lengths[name] = tokenized[name]["num_visible"]
        x = torch.cat(xs, dim=1)
        for blk in self.blocks[: -self.num_main_blocks]:
            x = (
                checkpoint.checkpoint(blk, x, use_reentrant=False)
                if self.use_checkpoint
                else blk(x)
            )
        x = x[..., 0, 0, :]
        if self.ape:
            pos_chunks = []
            batch_size = x.size(0)
            for name in active_modalities:
                absolute_pos_embed = self.absolute_pos_embeds[name]
                pos_embed = torch.gather(
                    absolute_pos_embed.expand(batch_size, -1, -1),
                    dim=1,
                    index=tokenized[name]["ids_keep"][:, :, None].expand(
                        -1, -1, absolute_pos_embed.shape[2]
                    ),
                )
                token_embed = (
                    pos_embed
                    + self._modality_embed_for(name, batch_size)
                    + self._physical_scale_embed_for(name, batch_size)
                    + tokenized[name].get("spherical_geo_embed", 0.0)
                )
                pos_chunks.append(token_embed)
            x = x + torch.cat(pos_chunks, dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks[-self.num_main_blocks :]:
            x = (
                checkpoint.checkpoint(blk, x, None, use_reentrant=False)
                if self.use_checkpoint
                else blk(x, None)
            )
        x = self.norm(x)
        latents = {}
        start = 0
        for name in active_modalities:
            end = start + keep_lengths[name]
            latents[name] = x[:, start:end]
            start = end
        return latents

    def forward_decoder(self, name, x, ids_restore):
        x = self.decoder_embeds[name](x)
        mask_tokens = self.mask_tokens[name].repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = x + self.decoder_pos_embeds[name]
        for blk in self.decoder_blocks[name]:
            x = blk(x)
        x = self.decoder_norms[name](x)
        x = self.decoder_preds[name](x)
        return x

    def masked_distillation_targets(self, batch_dict, mask_ratio=0.75):
        """Encode old modalities with MAE-visible tokens and expose the mask ids."""
        active_modalities = [
            name for name in self.get_active_modalities(batch_dict) if name != "nac"
        ]
        metadata_keep_mask = self._sample_metadata_keep_mask(
            batch_dict, active_modalities
        )
        tokenized = {
            name: self.tokenize_modality(
                batch_dict,
                name,
                mask_ratio,
                metadata_keep_mask=metadata_keep_mask,
            )
            for name in active_modalities
        }
        latents = self.forward_shared_encoder(tokenized, active_modalities)
        return {
            name: {
                "tokens": latents[name],
                "ids_keep": tokenized[name]["ids_keep"],
            }
            for name in active_modalities
        }


def lanyue_base(**kwargs):
    """Construct the five-modal CE2-anchor v2 HiViT-Base model."""
    return LanyueMAE(**kwargs)
