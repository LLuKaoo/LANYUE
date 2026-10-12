"""Native 512x512 NAC input and balanced old/NAC distillation batches."""

from __future__ import annotations

import numpy as np
import torch

from .loader import SignatureLoader
from .shards import Ce2AnchorWebDataset, iter_tar_samples

NAC_FORMAT = "nac_native512_v1"


class NacTransform:
    """Apply the source NAC uint8/255 mapping and optional valid-pixel z-score."""

    def __init__(self, normalization=None, random_hflip=False, random_vflip=False):
        self.normalization = normalization or {}
        if self.normalization:
            if set(self.normalization) != {"mean", "std"}:
                raise ValueError("NAC normalization requires mean and std.")
            mean, std = self.normalization["mean"], self.normalization["std"]
            if not np.isfinite([mean, std]).all() or std <= 0:
                raise ValueError("NAC normalization must be finite with std > 0.")
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

    def __call__(self, arrays):
        import random

        values = np.asarray(arrays["nac"])
        mask = np.asarray(arrays["nac_valid_mask"])
        if values.ndim == 2:
            values = values[None]
        if mask.ndim == 2:
            mask = mask[None]
        if values.shape != (1, 512, 512) or mask.shape != values.shape:
            raise ValueError("NAC signal/mask must have shape [1,512,512].")
        if values.dtype != np.uint8:
            raise ValueError(
                "Stored NAC observations must be uint8, encoded from reflectance."
            )
        if not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("NAC masks must be binary and include valid observations.")
        valid = torch.from_numpy(np.array(mask, dtype=np.bool_, copy=True))
        image = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)) / 255.0
        if self.normalization:
            image = (image - self.normalization["mean"]) / self.normalization["std"]
        image[~valid] = 0.0
        if self.random_hflip and random.random() < 0.5:
            image, valid = torch.flip(image, [2]), torch.flip(valid, [2])
        if self.random_vflip and random.random() < 0.5:
            image, valid = torch.flip(image, [1]), torch.flip(valid, [1])
        return {"nac": image.contiguous(), "nac_valid_mask": valid.float().contiguous()}


class NacWebDataset(Ce2AnchorWebDataset):
    """Use the same rank/worker shard assignment as the five-modal reader."""

    def __iter__(self):
        for shard in self._assigned_shards():
            for raw in iter_tar_samples(shard):
                sample_id = raw.pop("sample_id")
                meta = raw.pop("meta")
                if (
                    meta.get("sample_id") != sample_id
                    or meta.get("modalities") != ["nac"]
                    or meta.get("modality_signature") != "nac"
                ):
                    raise ValueError(f"Invalid NAC sample metadata: {sample_id}")
                if set(raw) != {"nac", "nac_valid_mask"}:
                    raise ValueError(f"Invalid NAC array fields: {sample_id}")
                values = (
                    self.transform(raw)
                    if self.transform is not None
                    else NacTransform()(raw)
                )
                center = meta.get("center_lonlat")
                if center is None:
                    center = [float("nan"), float("nan")]
                center = torch.as_tensor(center, dtype=torch.float32)
                if (
                    center.shape != (2,)
                    or (
                        torch.isfinite(center).any()
                        and not torch.isfinite(center).all()
                    )
                    or (torch.isfinite(center).all() and abs(center[1].item()) > 90)
                ):
                    raise ValueError(f"Invalid NAC center_lonlat: {sample_id}")
                yield {
                    **values,
                    "sample_id": sample_id,
                    "center_lonlat": center,
                    "modality_signature": "nac",
                    "batch_type": "nac_mae",
                }


class NacLoader(SignatureLoader):
    dataset_format = NAC_FORMAT
    dataset_class = NacWebDataset
    transform_class = NacTransform

    @staticmethod
    def validate_signature(name):
        if name != "nac":
            raise ValueError("Native NAC datasets must have only the nac signature.")


class DistillationLoader:
    """Consume bounded old:NAC cycles with no depleted-loader restarts."""

    def __init__(
        self, old_loader, nac_loader, old_batches=4, nac_batches=1, max_cycles=None
    ):
        if (
            old_batches < 1
            or nac_batches < 1
            or (max_cycles is not None and max_cycles < 1)
        ):
            raise ValueError(
                "Distillation batch ratios and cycle limits must be positive."
            )
        self.old_loader, self.nac_loader = old_loader, nac_loader
        self.weights = {"old": int(old_batches), "nac": int(nac_batches)}
        self.max_cycles = max_cycles
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.old_loader.set_epoch(epoch)
        self.nac_loader.set_epoch(epoch)
        self.cycles = min(
            len(self.old_loader) // self.weights["old"],
            len(self.nac_loader) // self.weights["nac"],
        )
        if self.max_cycles is not None:
            self.cycles = min(self.cycles, self.max_cycles)
        if not self.cycles:
            raise ValueError(
                "The datasets cannot supply one old/NAC cycle. Reduce batch size, ranks, workers, or the cycle ratio."
            )

    def __len__(self):
        return self.cycles * sum(self.weights.values())

    def __iter__(self):
        iterators = {"old": iter(self.old_loader), "nac": iter(self.nac_loader)}
        scores = {name: 0 for name in self.weights}
        total = sum(self.weights.values())
        for _ in range(len(self)):
            for name, weight in self.weights.items():
                scores[name] += weight
            name = max(scores, key=scores.get)
            scores[name] -= total
            yield next(iterators[name])
