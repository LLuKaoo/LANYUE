"""Streaming reader and fixed-footprint transforms for CE2-anchor v2 data.

The on-disk spatial sizes preserve the requested physical sampling.  Batches
are padded, never resized, to the model canvases below.  Every padded tensor is
paired with a pixel-validity mask so reconstruction losses can ignore padding
and source nodata.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from itertools import zip_longest
from typing import Callable, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - reported when the reader is constructed
    zstd = None


@dataclass(frozen=True)
class ModalityLayout:
    stored_size: tuple[int, int]
    model_size: tuple[int, int]
    patch_size: int
    meters_per_pixel: float


CE2_ANCHOR_V2_LAYOUTS: Dict[str, ModalityLayout] = {
    "ce2_dom": ModalityLayout((512, 512), (512, 512), 16, 7.0),
    "ce2_dem": ModalityLayout((179, 179), (192, 192), 16, 20.0),
    "nasa_dom": ModalityLayout((36, 36), (40, 40), 8, 100.0),
    "nasa_dem": ModalityLayout((61, 61), (64, 64), 8, 59.0),
    "minirf_sar": ModalityLayout((484, 484), (496, 496), 16, 7.4),
}

CANONICAL_MODALITY_ORDER = tuple(CE2_ANCHOR_V2_LAYOUTS)


def _require_zstd() -> None:
    if zstd is None:
        raise ImportError(
            "zstandard is required for CE2-anchor v2 data; install the "
            "dependencies in requirements.txt"
        )


def encode_array_zstd(array: np.ndarray, level: int = 3) -> bytes:
    """Encode an ndarray as a portable NPY frame compressed with Zstandard."""
    _require_zstd()
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return zstd.ZstdCompressor(level=int(level)).compress(stream.getvalue())


def decode_array_zstd(blob: bytes) -> np.ndarray:
    """Decode an array written by :func:`encode_array_zstd`."""
    _require_zstd()
    raw = zstd.ZstdDecompressor().decompress(blob)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def modality_signature(modalities: Iterable[str]) -> str:
    present = set(modalities)
    ordered = [name for name in CANONICAL_MODALITY_ORDER if name in present]
    unknown = sorted(present - set(ordered))
    return "+".join(ordered + unknown)


def validate_ce2_anchor_modalities(modalities: Iterable[str]) -> tuple[str, ...]:
    """Validate and canonicalize one homogeneous CE2-anchor input form."""
    names = tuple(modalities)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate modalities in CE2-anchor sample: {names}")
    unknown = sorted(set(names) - set(CANONICAL_MODALITY_ORDER))
    if unknown:
        raise ValueError(f"Unsupported CE2-anchor modalities: {unknown}")
    if "ce2_dom" not in names:
        raise ValueError(
            "Every CE2-anchor input form must contain the ce2_dom anchor."
        )
    present = set(names)
    return tuple(name for name in CANONICAL_MODALITY_ORDER if name in present)


def visible_tokens_for_mask_ratio(mask_ratio: float) -> Dict[str, int]:
    """Derive per-modality quotas from the current v2 canvases and patches."""
    ratio = float(mask_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {ratio}")
    quotas = {}
    for name, layout in CE2_ANCHOR_V2_LAYOUTS.items():
        height, width = layout.model_size
        patch = layout.patch_size
        num_tokens = (height // patch) * (width // patch)
        quotas[name] = max(1, int(num_tokens * (1.0 - ratio)))
    return quotas


def pad_array_and_mask(
    array: np.ndarray,
    valid_mask: np.ndarray,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a stored array to CHW and pad it to its model canvas.

    Padding is appended on the right and bottom to retain the pixel origin and
    geotransform of the stored chip.  The returned mask contains zero for both
    source nodata and model-only padding.
    """
    if modality not in CE2_ANCHOR_V2_LAYOUTS:
        raise KeyError(f"Unknown CE2-anchor v2 modality: {modality}")
    layout = CE2_ANCHOR_V2_LAYOUTS[modality]
    array = np.asarray(array)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if array.ndim == 2:
        array = array[None]
    if valid_mask.ndim == 2:
        valid_mask = valid_mask[None]
    if tuple(array.shape[-2:]) != layout.stored_size:
        raise ValueError(
            f"{modality}: stored array is {tuple(array.shape[-2:])}, "
            f"expected {layout.stored_size}"
        )
    if valid_mask.shape[-2:] != array.shape[-2:]:
        raise ValueError(
            f"{modality}: mask {valid_mask.shape} does not match array {array.shape}"
        )

    tensor = torch.from_numpy(np.array(array, copy=True))
    mask = torch.from_numpy(np.array(valid_mask, copy=True)).float()
    out_h, out_w = layout.model_size
    pad_h = out_h - tensor.shape[-2]
    pad_w = out_w - tensor.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"{modality}: stored size exceeds model canvas {layout.model_size}"
        )
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
        mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0)
    return tensor.contiguous(), mask.contiguous()


def _member_parts(name: str) -> tuple[str, str] | None:
    base = Path(name).name
    marker = ".meta.json"
    if base.endswith(marker):
        return base[: -len(marker)], "meta"
    marker = ".npy.zst"
    if base.endswith(marker):
        stem = base[: -len(marker)]
        if "." not in stem:
            return None
        sample_id, field = stem.rsplit(".", 1)
        return sample_id, field
    return None


def iter_tar_samples(
    path: str | os.PathLike[str], skip_fields: Iterable[str] = (),
) -> Iterator[dict]:
    """Yield decoded, unpadded records from one v2 TAR shard."""
    skipped = set(skip_fields)
    current_id = None
    record: dict = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = _member_parts(member.name)
            if parts is None:
                continue
            sample_id, field = parts
            if current_id is not None and sample_id != current_id:
                record["sample_id"] = current_id
                yield record
                record = {}
            current_id = sample_id
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot read {member.name} from {path}")
            payload = handle.read()
            if field == "meta":
                record["meta"] = json.loads(payload)
            elif field in skipped:
                continue
            else:
                record[field] = decode_array_zstd(payload)
    if current_id is not None:
        record["sample_id"] = current_id
        yield record


class Ce2AnchorWebDataset(IterableDataset):
    """Dependency-light sequential reader for v2 WebDataset-compatible TARs.

    Shards are divided deterministically across distributed ranks and DataLoader
    workers.  For homogeneous PyTorch collation, pass shards from one modality
    signature at a time (the builder stores signatures in separate folders).
    """

    def __init__(
        self,
        shards: Sequence[str | os.PathLike[str]],
        expected_signature: str | None = None,
        transform: Callable[[dict], dict | None] | None = None,
        num_samples: int | None = None,
        sar_overlay_shards: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        _require_zstd()
        self.shards = tuple(str(Path(path)) for path in shards)
        if not self.shards:
            raise ValueError("Ce2AnchorWebDataset requires at least one shard")
        self.expected_signature = expected_signature
        self.transform = transform
        self.num_samples = None if num_samples is None else int(num_samples)
        self.sar_overlay_shards = {
            str(Path(base)): str(Path(overlay))
            for base, overlay in (sar_overlay_shards or {}).items()
        }
        self.rank, self.world = self._rank()
        self._epoch = multiprocessing.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        if self.num_samples is None:
            raise TypeError(
                "Dataset length is unknown; pass num_samples from dataset.json"
            )
        _, world = self._rank()
        return (self.num_samples + world - 1) // world

    @staticmethod
    def _rank() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _assigned_shards(self) -> Sequence[str]:
        rank, world = self.rank, self.world
        shuffled = list(self.shards)
        random.Random(0xCE2 + int(self._epoch.value)).shuffle(shuffled)
        ranked = shuffled[rank::world]
        worker = get_worker_info()
        if worker is None:
            return ranked
        return ranked[worker.id::worker.num_workers]

    def __iter__(self) -> Iterator[dict]:
        for shard in self._assigned_shards():
            overlay_path = self.sar_overlay_shards.get(str(Path(shard)))
            if overlay_path is None:
                records = iter_tar_samples(shard)
            else:
                base_records = iter_tar_samples(
                    shard,
                    skip_fields={"minirf_sar", "minirf_sar_valid_mask"},
                )
                overlay_records = iter_tar_samples(overlay_path)

                def merged_records():
                    for base, overlay in zip_longest(base_records, overlay_records):
                        if base is None or overlay is None:
                            raise ValueError(
                                f"Base/overlay sample count mismatch: "
                                f"{shard} vs {overlay_path}"
                            )
                        if base.get("sample_id") != overlay.get("sample_id"):
                            raise ValueError(
                                "Base/overlay sample id mismatch: "
                                f"{base.get('sample_id')} vs "
                                f"{overlay.get('sample_id')}"
                            )
                        base["minirf_sar"] = overlay.pop("minirf_sar")
                        base["minirf_sar_valid_mask"] = overlay.pop(
                            "minirf_sar_valid_mask"
                        )
                        overlay.pop("sample_id", None)
                        if overlay:
                            raise ValueError(
                                f"Unexpected SAR overlay fields: {sorted(overlay)}"
                            )
                        yield base

                records = merged_records()
            for raw in records:
                meta = raw.pop("meta", {})
                modalities = validate_ce2_anchor_modalities(
                    meta.get("modalities", ())
                )
                signature = modality_signature(modalities)
                declared_signature = meta.get("modality_signature")
                if declared_signature is not None and declared_signature != signature:
                    raise ValueError(
                        f"{raw['sample_id']}: declared signature="
                        f"{declared_signature}, actual={signature}"
                    )
                if self.expected_signature and signature != self.expected_signature:
                    raise ValueError(
                        f"{raw['sample_id']}: signature={signature}, "
                        f"expected={self.expected_signature}"
                    )
                out = {
                    "sample_id": raw.pop("sample_id"),
                    "scene_id": meta.get("scene_id", meta.get("sample_id")),
                    "parent_scene_id": meta.get("parent_scene_id", meta.get("sample_id")),
                    "batch_type": "ce2_anchor_v2",
                    "modality_signature": signature,
                }
                center = meta.get("center_lonlat")
                if center is not None:
                    out["center_lonlat"] = torch.tensor(center, dtype=torch.float32)
                arrays = {}
                for name in modalities:
                    arrays[name] = raw.pop(name)
                    arrays[f"{name}_valid_mask"] = raw.pop(
                        f"{name}_valid_mask"
                    )
                if self.transform is None:
                    for name in modalities:
                        tensor, valid = pad_array_and_mask(
                            arrays[name], arrays[f"{name}_valid_mask"], name
                        )
                        out[name] = tensor
                        out[f"{name}_valid_mask"] = valid
                else:
                    transformed = self.transform(arrays)
                    if transformed is None:
                        continue
                    out.update(transformed)
                if raw:
                    raise ValueError(
                        f"{out['sample_id']}: unconsumed shard fields {sorted(raw)}"
                    )
                yield out


def discover_signature_shards(
    root: str | os.PathLike[str], split: str = "train"
) -> Mapping[str, list[str]]:
    """Return ``signature -> sorted TAR paths`` from a built v2 dataset."""
    base = Path(root) / "shards" / split
    result = {}
    if not base.is_dir():
        return result
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        shards = sorted(str(path) for path in directory.glob("*.tar"))
        if shards:
            result[directory.name.replace("__", "+")] = shards
    return result
