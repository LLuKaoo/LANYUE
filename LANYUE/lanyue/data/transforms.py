"""Aligned five-modal transforms: normalize valid observations, then pad."""
from __future__ import annotations

import random
import numpy as np
import torch

from .shards import CE2_ANCHOR_V2_LAYOUTS, pad_array_and_mask, validate_ce2_anchor_modalities


class FiveModalTransform:
    def __init__(self, normalization=None, dem_clip=(-10000.0, 10000.0),
                 sar_clip=None, random_hflip=False, random_vflip=False):
        self.normalization = normalization or {}
        unknown = set(self.normalization) - set(CE2_ANCHOR_V2_LAYOUTS)
        if unknown:
            raise ValueError(f"Unknown normalization modalities: {unknown}")
        for name, stats in self.normalization.items():
            if set(stats) != {"mean", "std"}:
                raise ValueError(f"{name}: normalization requires mean and std.")
            if not np.isfinite(stats['mean']) or not np.isfinite(stats['std']) or stats['std'] <= 0:
                raise ValueError(f"{name}: normalization must be finite, with std > 0.")
        self.dem_clip = self._clip_bounds(dem_clip)
        self.sar_clip = self._clip_bounds(sar_clip)
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

    @staticmethod
    def _clip_bounds(bounds):
        if bounds is None:
            return None
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Clip bounds must be finite and increasing.")
        return tuple(float(v) for v in bounds)

    def __call__(self, arrays):
        modalities = validate_ce2_anchor_modalities(name for name in CE2_ANCHOR_V2_LAYOUTS if name in arrays)
        output = {}
        hflip = self.random_hflip and random.random() < 0.5
        vflip = self.random_vflip and random.random() < 0.5
        for name in modalities:
            array = np.asarray(arrays[name])
            if array.ndim == 2:
                array = array[None]
            if array.shape != (1, *CE2_ANCHOR_V2_LAYOUTS[name].stored_size):
                raise ValueError(f"{name}: expected one-channel stored size, got {array.shape}.")
            valid = np.asarray(arrays[f'{name}_valid_mask'], dtype=np.bool_)
            if valid.ndim == 2:
                valid = valid[None]
            if valid.shape != array.shape:
                raise ValueError(f"{name}: validity mask shape differs from signal.")
            finite = np.isfinite(array) & (np.abs(array.astype(np.float64)) < 1.0e30)
            if np.any(valid & ~finite):
                raise ValueError(f"{name}: mask declares non-finite/extreme nodata valid; rebuild or repair the dataset.")
            if not valid.any():
                raise ValueError(f"{name}: no valid observations in declared modality.")
            values = array.astype(np.float32, copy=True)
            values[~valid] = 0.0
            if name.endswith('_dom'):
                values /= 255.0
            elif name.endswith('_dem') and self.dem_clip is not None:
                values = np.clip(values, *self.dem_clip) / max(abs(v) for v in self.dem_clip)
            elif name == 'minirf_sar' and self.sar_clip is not None:
                values = np.clip(values, *self.sar_clip)
            stats = self.normalization.get(name)
            if stats is not None:
                values = (values - float(stats['mean'])) / float(stats['std'])
            values[~valid] = 0.0
            # Flip the common physical footprint before modality-specific padding.
            # Flipping padded canvases would move each sensor by a different border.
            if hflip:
                values, valid = np.flip(values, 2), np.flip(valid, 2)
            if vflip:
                values, valid = np.flip(values, 1), np.flip(valid, 1)
            output[name], output[f'{name}_valid_mask'] = pad_array_and_mask(values, valid, name)
        return {name: value.contiguous() for name, value in output.items()}
