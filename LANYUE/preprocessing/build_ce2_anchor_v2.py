#!/usr/bin/env python3
"""Build the CE2-anchor v2 pretraining dataset.

All secondary sensors are geospatially warped to the exact footprint of each
512 x 512 CE2 DOM chip.  The stored sizes are 512/179/36/61/484; no model-size
padding is stored on disk.  Output consists of uncompressed TAR shards whose
individual NPY members use Zstandard, plus a line-oriented sample catalogue.

The script accepts either a JSONL source manifest (easy to audit/restart) or a
CE2 master LMDB.  Optional source modalities in a JSONL row have this shape::

    {"sample_id":"patch_...", "center_lonlat":[lon,lat],
     "ce2_dom":{"path":"...tif", "window":[col,row,width,height]},
     "ce2_dem":{"path":"...tif"}, "nasa_dom":{"path":"...tif"}, ...}

For a global WAC/SLDEM source, command-line paths are used when a row does not
provide the modality.  Missing secondary modalities are recorded, not fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lanyue.data.shards import (  # noqa: E402
    CE2_ANCHOR_V2_LAYOUTS,
    encode_array_zstd,
    modality_signature,
)

try:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover - compute environment dependency
    rasterio = None
    Affine = None
    Resampling = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None


STORED_MODALITIES = ("ce2_dom", "ce2_dem", "nasa_dom", "nasa_dem", "minirf_sar")
RESAMPLING = {
    "ce2_dom": "bilinear",
    "ce2_dem": "bilinear",
    "nasa_dom": "bilinear",
    "nasa_dem": "bilinear",
    "minirf_sar": "bilinear",
}
MOON_EQC_CRS = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +R=1737400 +units=m +no_defs"
MOON_RADIUS_M = 1737400.0


def _require_rasterio() -> None:
    if rasterio is None:
        raise ImportError(
            "rasterio and affine are required by the builder"
        ) from _RASTERIO_IMPORT_ERROR


class RasterCache:
    """Small LRU of open datasets; global mosaics stay open across samples."""

    def __init__(self, max_open: int = 32):
        self.max_open = int(max_open)
        self.datasets = OrderedDict()

    def get(self, path):
        key = str(path)
        dataset = self.datasets.pop(key, None)
        if dataset is None:
            dataset = rasterio.open(key)
        self.datasets[key] = dataset
        while len(self.datasets) > self.max_open:
            _, old = self.datasets.popitem(last=False)
            old.close()
        return dataset

    def close(self):
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()


def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(
                    row.get("ce2_dom"), dict
                ):
                    raise ValueError(
                        "Each source row requires a ce2_dom source object."
                    )
                for name in STORED_MODALITIES:
                    specs = row.get(name)
                    if specs is None:
                        continue
                    for spec in specs if isinstance(specs, list) else [specs]:
                        source = Path(spec["path"]).expanduser()
                        if not source.is_absolute():
                            spec["path"] = str(
                                (Path(path).resolve().parent / source).resolve()
                            )
                yield row
            except Exception as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def iter_ce2_lmdb(path: str | os.PathLike[str], start: int = 0) -> Iterator[dict]:
    """Translate the existing CE2 master LMDB records into source rows."""
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lmdb is required for --ce2-master-lmdb") from exc
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=512,
        subdir=Path(path).is_dir(),
    )
    try:
        with env.begin(buffers=False) as txn:
            cursor = txn.cursor()
            if int(start) > 0:
                # Existing CE2 keys are contiguous and one-based
                # (patch_00000001...). set_range avoids O(start) rescans in
                # every independently submitted part.
                cursor.set_range(f"patch_{int(start) + 1:08d}".encode())
            for key, value in cursor:
                if not key.startswith(b"patch_"):
                    continue
                record = pickle.loads(value)
                width = int(record.get("patch_w", 512))
                height = int(record.get("patch_h", 512))
                row = {
                    "sample_id": key.decode("utf-8"),
                    "source_ce2_key": key.decode("utf-8"),
                    "center_lonlat": _json_safe(record.get("center_lonlat")),
                    "patch_boundary_lonlat": _json_safe(
                        record.get("patch_boundary_lonlat")
                        or record.get("corners_lonlat")
                    ),
                    "ce2_dom": {
                        "path": record["tif_path"],
                        "crs": (
                            str(record.get("raw_prj_wkt", ""))
                            .replace("\\x1a", "")
                            .replace("\x1a", "")
                            or None
                        ),
                        "window": [int(record["x"]), int(record["y"]), width, height],
                    },
                }
                yield row
    finally:
        env.close()


def exact_destination_grid(
    ce2_transform: "Affine", stored_size: tuple[int, int]
) -> "Affine":
    """Return a grid with the CE2 chip's exact outer footprint."""
    out_h, out_w = stored_size
    return ce2_transform * Affine.scale(512.0 / out_w, 512.0 / out_h)


def _valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    finite = np.all(np.isfinite(data), axis=0)
    # Some lunar GeoTIFFs use a float32-min sentinel which can be changed by
    # bilinear resampling enough that an exact ``array != src.nodata`` check
    # misses it. Values at this magnitude are never physical lunar data and
    # overflow patch statistics, so reject them independently of metadata.
    if np.issubdtype(data.dtype, np.floating):
        finite &= np.all((data > -1.0e30) & (data < 1.0e30), axis=0)
    if nodata is not None:
        if np.isnan(nodata):
            finite &= np.all(~np.isnan(data), axis=0)
        else:
            finite &= np.all(data != nodata, axis=0)
    return finite


def source_crs_for_warp(src, spec: Mapping):
    """Return a usable lunar CRS, repairing known malformed global headers."""
    if spec.get("crs"):
        return spec["crs"]
    crs = src.crs
    text = "" if crs is None else crs.to_wkt()
    if (
        crs is None
        or "ENGCRS" in text
        or "Geocoding information not available" in text
        or "Moon_Mecator" in text
    ):
        prj_path = Path(src.name).with_suffix(".prj")
        if prj_path.is_file():
            prj = prj_path.read_text(encoding="utf-8", errors="ignore")
            prj = prj.replace("\\x1a", "").replace("\x1a", "").strip()
            if prj:
                return prj
        # The WAC file's GeoTIFF transform is lunar equirectangular, but its
        # CRS tag is an unusable engineering CRS. This is the same lunar
        # radius/eqc interpretation used by the prior crop pipeline.
        return MOON_EQC_CRS
    return crs


def read_ce2_anchor(
    spec: Mapping, cache: RasterCache | None = None
) -> tuple[np.ndarray, np.ndarray, object, "Affine"]:
    _require_rasterio()
    context = (
        nullcontext(cache.get(spec["path"])) if cache else rasterio.open(spec["path"])
    )
    with context as src:
        window = Window(*[int(v) for v in spec["window"]])
        array = src.read(window=window)
        if array.shape[-2:] != (512, 512):
            raise ValueError(
                f"CE2 DOM window returned {array.shape}, expected (*,512,512)"
            )
        valid = _valid_mask(array, src.nodata)
        transform_ = rasterio.windows.transform(window, src.transform)
        crs = source_crs_for_warp(src, spec)
    return array[:1], valid, crs, transform_


def reproject_source(
    spec: Mapping,
    modality: str,
    dst_crs,
    ce2_transform: "Affine",
    cache: RasterCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp one source raster and its validity mask to the CE2 footprint."""
    _require_rasterio()
    out_h, out_w = CE2_ANCHOR_V2_LAYOUTS[modality].stored_size
    dst_transform = exact_destination_grid(ce2_transform, (out_h, out_w))
    context = (
        nullcontext(cache.get(spec["path"])) if cache else rasterio.open(spec["path"])
    )
    with context as src:
        src_crs = source_crs_for_warp(src, spec)
        indexes = spec.get("bands", [1])
        source_window = None
        source_transform = src.transform
        if spec.get("window") is not None:
            source_window = Window(*[int(v) for v in spec["window"]])
            source_transform = rasterio.windows.transform(source_window, src.transform)
        method = getattr(Resampling, spec.get("resampling", RESAMPLING[modality]))
        if source_window is None:
            # A rasterio.band proxy lets GDAL select only source blocks that
            # contribute to this CE2 footprint; the global mosaic is never
            # materialized as a NumPy array.
            destination = np.full((out_h, out_w), np.nan, dtype=np.float32)
            with rasterio.Env(PROJ_IGNORE_CELESTIAL_BODY="YES"):
                reproject(
                    source=rasterio.band(src, indexes[0]),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=np.nan,
                    resampling=method,
                    num_threads=1,
                    init_dest_nodata=True,
                )
            valid = (
                np.isfinite(destination)
                & (destination > -1.0e30)
                & (destination < 1.0e30)
            )
            destination[~valid] = 0.0
            destination = destination[None]
            if modality == "nasa_dom":
                destination = np.clip(np.rint(destination), 0, 255).astype(np.uint8)
            return destination, valid

        source = src.read(indexes=indexes, window=source_window)
        src_valid = _valid_mask(source, src.nodata).astype(np.uint8)
        destination = np.full((1, out_h, out_w), np.nan, dtype=np.float32)
        destination_valid = np.zeros((out_h, out_w), dtype=np.uint8)
        with rasterio.Env(PROJ_IGNORE_CELESTIAL_BODY="YES"):
            reproject(
                source=source[:1],
                destination=destination,
                src_transform=source_transform,
                src_crs=src_crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=method,
                num_threads=1,
                init_dest_nodata=True,
            )
            reproject(
                source=src_valid,
                destination=destination_valid,
                src_transform=source_transform,
                src_crs=src_crs,
                src_nodata=0,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                num_threads=1,
                init_dest_nodata=True,
            )
    valid = (
        (destination_valid > 0)
        & np.isfinite(destination[0])
        & (destination[0] > -1.0e30)
        & (destination[0] < 1.0e30)
    )
    destination[0, ~valid] = 0.0
    if modality in {"nasa_dom"}:
        destination = np.clip(np.rint(destination), 0, 255).astype(np.uint8)
    return destination, valid


def _eqc_window_from_lonlat(src, boundary, center_lon):
    """Raw equirectangular window; return None for dateline-crossing chips."""
    if not boundary:
        return None
    center_lon = float(center_lon)
    unwrapped = []
    for lon, lat in boundary:
        delta = ((float(lon) - center_lon + 180.0) % 360.0) - 180.0
        unwrapped.append((center_lon + delta, float(lat)))
    lons = [point[0] for point in unwrapped]
    if min(lons) < -180.0 or max(lons) > 180.0:
        return None
    xs = [MOON_RADIUS_M * math.radians(point[0]) for point in unwrapped]
    ys = [MOON_RADIUS_M * math.radians(point[1]) for point in unwrapped]
    inverse = ~src.transform
    pixels = [inverse * (x, y) for x, y in zip(xs, ys)]
    col0 = max(0, math.floor(min(point[0] for point in pixels)))
    row0 = max(0, math.floor(min(point[1] for point in pixels)))
    col1 = min(src.width, math.ceil(max(point[0] for point in pixels)))
    row1 = min(src.height, math.ceil(max(point[1] for point in pixels)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def resample_source_window(
    spec: Mapping,
    modality: str,
    boundary,
    center_lonlat,
    cache: RasterCache | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop the old geographic window and resize it without reprojection."""
    out_h, out_w = CE2_ANCHOR_V2_LAYOUTS[modality].stored_size
    context = (
        nullcontext(cache.get(spec["path"])) if cache else rasterio.open(spec["path"])
    )
    with context as src:
        if spec.get("window") is not None:
            window = Window(*[int(v) for v in spec["window"]])
        else:
            window = _eqc_window_from_lonlat(src, boundary, center_lonlat[0])
            if window is None:
                return None
        index = int(spec.get("bands", [1])[0])
        method = getattr(Resampling, spec.get("resampling", RESAMPLING[modality]))
        array = src.read(
            index,
            window=window,
            out_shape=(out_h, out_w),
            resampling=method,
            masked=False,
        ).astype(np.float32, copy=False)
        sensor_mask = (
            src.read_masks(
                index,
                window=window,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            )
            > 0
        )
        valid = sensor_mask & np.isfinite(array) & (array > -1.0e30) & (array < 1.0e30)
        if src.nodata is not None:
            valid &= array != src.nodata
        array[~valid] = 0.0
        array = array[None]
        if modality == "nasa_dom":
            array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
        return array, valid


def center_from_grid(crs, transform_: "Affine") -> list[float]:
    x, y = transform_ * (256.0, 256.0)
    lunar_lonlat = "+proj=longlat +R=1737400 +no_defs"
    with rasterio.Env(PROJ_IGNORE_CELESTIAL_BODY="YES"):
        lon, lat = transform(crs, lunar_lonlat, [x], [y])
    return [float(lon[0]), float(lat[0])]


def tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, fileobj=__import__("io").BytesIO(payload))


@dataclass
class OpenShard:
    archive: tarfile.TarFile
    path: Path
    count: int = 0
    payload_bytes: int = 0


class SignatureShardWriter:
    def __init__(self, root: Path, max_samples: int, max_bytes: int, zstd_level: int):
        self.root = root
        self.max_samples = int(max_samples)
        self.max_bytes = int(max_bytes)
        self.zstd_level = int(zstd_level)
        self.open: dict[str, OpenShard] = {}
        self.next_id = defaultdict(int)
        self.paths: dict[str, list[str]] = defaultdict(list)

    def _new(self, signature: str) -> OpenShard:
        folder = self.root / signature.replace("+", "__")
        folder.mkdir(parents=True, exist_ok=True)
        shard_id = self.next_id[signature]
        self.next_id[signature] += 1
        path = folder / f"part-{shard_id:06d}.tar"
        shard = OpenShard(tarfile.open(path, mode="w:"), path)
        self.paths[signature].append(str(path))
        self.open[signature] = shard
        return shard

    def write(
        self, sample_id: str, meta: dict, arrays: Mapping[str, np.ndarray]
    ) -> str:
        signature = meta["modality_signature"]
        encoded = {}
        for name, array in arrays.items():
            encoded[name] = encode_array_zstd(array, self.zstd_level)
        meta_payload = json.dumps(
            meta, ensure_ascii=False, separators=(",", ":")
        ).encode()
        estimate = len(meta_payload) + sum(map(len, encoded.values()))
        shard = self.open.get(signature)
        if (
            shard is None
            or shard.count >= self.max_samples
            or (shard.count and shard.payload_bytes + estimate > self.max_bytes)
        ):
            if shard is not None:
                shard.archive.close()
            shard = self._new(signature)
        tar_add_bytes(shard.archive, f"{sample_id}.meta.json", meta_payload)
        for name, payload in encoded.items():
            tar_add_bytes(shard.archive, f"{sample_id}.{name}.npy.zst", payload)
        shard.count += 1
        shard.payload_bytes += estimate
        return str(shard.path)

    def close(self) -> None:
        for shard in self.open.values():
            shard.archive.close()
        self.open.clear()


def deterministic_split(sample_id: str, val_percent: float, test_percent: float) -> str:
    bucket = (
        int.from_bytes(hashlib.sha1(sample_id.encode()).digest()[:8], "big") / 2**64
    )
    if bucket < test_percent / 100.0:
        return "test"
    if bucket < (test_percent + val_percent) / 100.0:
        return "val"
    return "train"


def jsonl_catalog_to_parquet(jsonl_path: Path, parquet_path: Path, batch_size=10000):
    """Write a stable, streaming Parquet view of the sample catalogue."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    writer = None
    batch = []

    def flush():
        nonlocal writer, batch
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
        writer.write_table(table)
        batch = []

    with open(jsonl_path, "r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            center = row.get("center_lonlat") or [None, None]
            batch.append(
                {
                    "sample_id": row["sample_id"],
                    "modalities": row["modalities"],
                    "modality_signature": row["modality_signature"],
                    "center_lon": center[0],
                    "center_lat": center[1],
                    "split": row["split"],
                    "shard": row["shard"],
                    "valid_ratios_json": json.dumps(row.get("valid_ratios", {})),
                    "missing_modalities_json": json.dumps(
                        row.get("missing_modalities", {})
                    ),
                }
            )
            if len(batch) >= batch_size:
                flush()
    flush()
    if writer is not None:
        writer.close()
        return True
    return False


def build(args) -> dict:
    _require_rasterio()
    if (
        not 0 <= args.val_percent
        or not 0 <= args.test_percent
        or args.val_percent + args.test_percent >= 100
    ):
        raise ValueError(
            "Split percentages must be nonnegative and sum to less than 100."
        )
    if (
        not 0 < args.min_valid_ratio <= 1
        or args.shard_samples < 1
        or args.shard_gib <= 0
    ):
        raise ValueError("Invalid validity threshold or shard size.")
    if args.start < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("start must be nonnegative and limit must be positive.")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    work = output.with_name(output.name + ".inprogress")
    if work.exists():
        raise FileExistsError(f"Incomplete output already exists: {work}")
    work.mkdir(parents=True)

    rows: Iterable[dict]
    if args.source_manifest:
        rows = iter_jsonl(args.source_manifest)
        source_start_applied = False
    else:
        rows = iter_ce2_lmdb(args.ce2_master_lmdb, start=args.start)
        source_start_applied = True
    globals_ = {
        "nasa_dom": args.wac,
        "nasa_dem": args.sldem,
    }
    join_db = None
    if args.join_sqlite:
        join_db = sqlite3.connect(
            f"file:{Path(args.join_sqlite).resolve()}?mode=ro", uri=True
        )
    counts = Counter()
    signature_counts = Counter()
    shard_counts = Counter()
    list_handles = {}
    writers = {}
    raster_cache = RasterCache(args.raster_cache_size)
    try:
        for split in ("train", "val", "test"):
            writers[split] = SignatureShardWriter(
                work / "shards" / split,
                args.shard_samples,
                int(args.shard_gib * 1024**3),
                args.zstd_level,
            )
            list_handles[split] = open(
                work / f"samples-{split}.jsonl", "w", encoding="utf-8"
            )
        for source_index, row in enumerate(rows):
            if not source_start_applied and source_index < args.start:
                continue
            if args.limit is not None and counts["visited"] >= args.limit:
                break
            counts["visited"] += 1
            absolute_index = source_index + (args.start if source_start_applied else 0)
            sample_id = str(row.get("sample_id", f"patch_{absolute_index:012d}"))
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                raise ValueError(
                    f"sample_id must use letters, numbers, underscores or hyphens: {sample_id!r}"
                )
            if join_db is not None:
                joined = join_db.execute(
                    "SELECT ce2_dem_json,minirf_sar_json FROM joins WHERE ce2_key=?",
                    (str(row.get("source_ce2_key", sample_id)),),
                ).fetchone()
                if joined is not None:
                    if row.get("ce2_dem") is None and joined[0]:
                        row["ce2_dem"] = json.loads(joined[0])
                    if row.get("minirf_sar") is None and joined[1]:
                        row["minirf_sar"] = json.loads(joined[1])
            try:
                dom, dom_valid, dst_crs, ce2_transform = read_ce2_anchor(
                    row["ce2_dom"], cache=raster_cache
                )
            except Exception as exc:
                counts["failed_ce2_dom"] += 1
                if counts["failed_ce2_dom"] <= 20:
                    print(f"[CE2 DOM FAIL] {sample_id}: {exc}", flush=True)
                continue
            if float(dom_valid.mean()) < args.min_valid_ratio:
                counts["failed_ce2_dom"] += 1
                continue
            arrays = {"ce2_dom": dom, "ce2_dom_valid_mask": dom_valid[None]}
            missing = {}
            valid_ratios = {"ce2_dom": float(dom_valid.mean())}
            center_for_mode = row.get("center_lonlat") or center_from_grid(
                dst_crs, ce2_transform
            )
            use_direct_resample = args.hybrid_latitude is not None and abs(
                float(center_for_mode[1])
            ) <= float(args.hybrid_latitude)
            for modality in STORED_MODALITIES[1:]:
                spec = row.get(modality)
                if spec is None and globals_.get(modality):
                    spec = {"path": globals_[modality]}
                if spec is None:
                    missing[modality] = row.get(f"{modality}_match_reason", "no_source")
                    continue
                candidates = spec if isinstance(spec, list) else [spec]
                best = None
                best_ratio = -1.0
                errors = []
                for candidate in candidates:
                    try:
                        result = None
                        if use_direct_resample:
                            result = resample_source_window(
                                candidate,
                                modality,
                                row.get("patch_boundary_lonlat"),
                                center_for_mode,
                                cache=raster_cache,
                            )
                        if result is None:
                            result = reproject_source(
                                candidate,
                                modality,
                                dst_crs,
                                ce2_transform,
                                cache=raster_cache,
                            )
                        array, valid = result
                        ratio = float(valid.mean())
                        if ratio > best_ratio:
                            best, best_ratio = (array, valid), ratio
                    except Exception as exc:
                        errors.append(repr(exc))
                if best is None or best_ratio < args.min_valid_ratio:
                    missing[modality] = (
                        f"valid_ratio={best_ratio:.6f}"
                        if best is not None
                        else f"warp_failed={errors[:2]}"
                    )
                    continue
                arrays[modality] = best[0]
                arrays[f"{modality}_valid_mask"] = best[1][None]
                valid_ratios[modality] = best_ratio

            modalities = [name for name in STORED_MODALITIES if name in arrays]
            signature = modality_signature(modalities)
            split = deterministic_split(sample_id, args.val_percent, args.test_percent)
            center = center_for_mode
            meta = {
                "sample_id": sample_id,
                "source_ce2_key": row.get("source_ce2_key", sample_id),
                "modalities": modalities,
                "modality_signature": signature,
                "center_lonlat": _json_safe(center),
                "patch_boundary_lonlat": _json_safe(row.get("patch_boundary_lonlat")),
                "stored_sizes": {
                    name: list(CE2_ANCHOR_V2_LAYOUTS[name].stored_size)
                    for name in modalities
                },
                "model_sizes": {
                    name: list(CE2_ANCHOR_V2_LAYOUTS[name].model_size)
                    for name in modalities
                },
                "valid_ratios": valid_ratios,
                "missing_modalities": missing,
                "split": split,
                "alignment": (
                    "raw_window_resample"
                    if use_direct_resample
                    else "geospatial_reproject_to_ce2_footprint"
                ),
            }
            shard_path = writers[split].write(sample_id, meta, arrays)
            list_record = {
                "sample_id": sample_id,
                "modalities": modalities,
                "modality_signature": signature,
                "center_lonlat": meta["center_lonlat"],
                "split": split,
                "shard": os.path.relpath(shard_path, work),
                "valid_ratios": valid_ratios,
                "missing_modalities": missing,
            }
            list_handles[split].write(
                json.dumps(list_record, ensure_ascii=False) + "\n"
            )
            shard_counts[list_record["shard"]] += 1
            counts[f"saved_{split}"] += 1
            signature_counts[f"{split}:{signature}"] += 1
            if counts["visited"] % args.log_every == 0:
                print(dict(counts), flush=True)
    except Exception:
        raise
    finally:
        for writer in writers.values():
            writer.close()
        for handle in list_handles.values():
            handle.close()
        if join_db is not None:
            join_db.close()
        raster_cache.close()

    if sum(counts.get(f"saved_{split}", 0) for split in ("train", "val", "test")) == 0:
        raise ValueError(
            "No valid CE2-anchor samples were built; inspect source paths and geometry."
        )
    parquet_splits = []
    if not args.no_parquet:
        for split in ("train", "val", "test"):
            if jsonl_catalog_to_parquet(
                work / f"samples-{split}.jsonl",
                work / f"samples-{split}.parquet",
            ):
                parquet_splits.append(split)
    summary = {
        "format": "ce2_anchor_v2",
        "counts": dict(counts),
        "signature_counts": dict(signature_counts),
        "shard_counts": dict(shard_counts),
        "stored_sizes": {
            name: list(CE2_ANCHOR_V2_LAYOUTS[name].stored_size)
            for name in STORED_MODALITIES
        },
        "model_sizes": {
            name: list(CE2_ANCHOR_V2_LAYOUTS[name].model_size)
            for name in STORED_MODALITIES
        },
        "extreme_nodata_abs_limit": 1.0e30,
        "compression": f"npy+zstd(level={args.zstd_level}) inside uncompressed tar",
        "catalogues": {
            "jsonl": ["train", "val", "test"],
            "parquet": parquet_splits,
        },
    }
    with open(work / "dataset.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(work, output)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-manifest")
    source.add_argument("--ce2-master-lmdb")
    p.add_argument("--output", required=True)
    p.add_argument("--wac", help="Global LROC WAC GeoTIFF")
    p.add_argument("--sldem", help="Global/partial SLDEM GeoTIFF")
    p.add_argument(
        "--join-sqlite", help="Metadata join built by prepare_ce2_anchor_v2_joins.py"
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--min-valid-ratio", type=float, default=0.01)
    p.add_argument(
        "--hybrid-latitude",
        type=float,
        default=60.0,
        help="Directly crop/resize at |lat| <= threshold; reproject outside it.",
    )
    p.add_argument("--val-percent", type=float, default=0.0)
    p.add_argument("--test-percent", type=float, default=0.0)
    p.add_argument("--shard-samples", type=int, default=4096)
    p.add_argument("--shard-gib", type=float, default=2.0)
    p.add_argument("--zstd-level", type=int, default=3)
    p.add_argument("--raster-cache-size", type=int, default=32)
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip the optional streaming Parquet catalogue.",
    )
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument(
        "--confirm-full-build",
        action="store_true",
        help="Required when --limit is omitted, to prevent accidental full rebuilds.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.limit is None and not args.confirm_full_build:
        raise SystemExit("Refusing full build without --confirm-full-build")
    if (
        args.val_percent < 0
        or args.test_percent < 0
        or (args.val_percent + args.test_percent >= 100)
    ):
        raise SystemExit("val/test percentages must be nonnegative and sum to <100")
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
