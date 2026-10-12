"""Build, validate, or measure native NAC TAR/Zstd datasets for v2 distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sqlite3
import tempfile
import zlib
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np

from lanyue.data.nac import NAC_FORMAT, NacTransform, NacWebDataset
from lanyue.data.shards import discover_signature_shards, iter_tar_samples
from .build_ce2_anchor_v2 import SignatureShardWriter


def starts(length, stride):
    if length < 512:
        return []
    values = list(range(0, length - 512 + 1, stride))
    if values[-1] != length - 512:
        values.append(length - 512)
    return values


def group_split(group, seed):
    fraction = (
        int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big")
        / 2**64
    )
    return "train" if fraction < 0.8 else "val" if fraction < 0.9 else "test"


def raster_rows(args):
    import rasterio

    if args.source_manifest:
        manifest = Path(args.source_manifest).resolve()
        with manifest.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    spec = row["nac"]
                    path = Path(spec["path"]).expanduser()
                    if not path.is_absolute():
                        path = manifest.parent / path
                    row.setdefault("group_id", spec["path"])
                    yield row, path, spec["window"]
    else:
        root = Path(args.source_root).resolve()
        paths = sorted(root.glob(args.glob))
        if not paths:
            raise ValueError(f"No NAC rasters matched {args.glob!r} below {root}.")
        for path in paths:
            relative = path.relative_to(root)
            group = relative.parts[0] if len(relative.parts) > 1 else path.stem
            with rasterio.open(path) as source:
                for y in starts(source.height, args.stride):
                    for x in starts(source.width, args.stride):
                        sample_id = (
                            "nac_"
                            + hashlib.sha256(
                                f"{relative}:x{x}:y{y}".encode()
                            ).hexdigest()[:24]
                        )
                        yield {"sample_id": sample_id, "group_id": group}, path, [
                            x,
                            y,
                            512,
                            512,
                        ]


def raster_samples(args):
    import rasterio
    from rasterio.windows import Window

    for row, path, window in raster_rows(args):
        if (
            len(window) != 4
            or list(window[2:]) != [512, 512]
            or any(float(v) != int(v) for v in window)
            or min(window[:2]) < 0
        ):
            raise ValueError(
                "NAC source windows must be nonnegative [column,row,512,512] integer windows."
            )
        with rasterio.open(path) as source:
            masked = source.read(1, window=Window(*window), masked=True)
        values = np.asarray(masked.filled(0))
        if values.shape != (512, 512):
            raise ValueError(f"{path}: native window was not 512x512.")
        valid = (
            ~np.ma.getmaskarray(masked)
            & np.isfinite(values)
            & (np.abs(values.astype(np.float64)) < 1.0e30)
        )
        if float(valid.mean()) < args.min_valid_fraction:
            continue
        domain = row.get("nac", {}).get("input_domain", args.input_domain)
        output = np.zeros((512, 512), dtype=np.uint8)
        if domain == "reflectance":
            output[valid] = np.clip(
                np.rint(values[valid].astype(np.float32) * 255.0), 0, 255
            ).astype(np.uint8)
        elif domain == "uint8":
            if values.dtype != np.uint8:
                raise ValueError("uint8 input domain requires a uint8 raster.")
            output[valid] = values[valid]
        else:
            raise ValueError(f"Unknown NAC input domain: {domain}")
        group = row.get("group_id", str(path))
        split = row.get("split", group_split(group, args.split_seed))
        yield row["sample_id"], split, row.get("center_lonlat"), output[None], valid[
            None
        ]


def lmdb_samples(args):
    """Convert the existing local native512 schema without altering observations."""
    import lmdb

    root = Path(args.lmdb_root)
    metadata = lmdb.open(
        str(root / "meta.lmdb"), readonly=True, lock=False, readahead=False
    )
    chip_env = None
    current_shard = None
    try:
        with metadata.begin() as transaction:
            for key, payload in transaction.cursor():
                if not key.startswith(b"patch_"):
                    continue
                meta = pickle.loads(payload)
                shard = int(meta["shard_id"])
                if shard != current_shard:
                    if chip_env is not None:
                        chip_env.close()
                    chip_env = lmdb.open(
                        str(root / f"chips_{shard:06d}.lmdb"),
                        readonly=True,
                        lock=False,
                        readahead=False,
                    )
                    current_shard = shard
                chip_key = meta.get("chip_key", key.decode())
                if isinstance(chip_key, str):
                    chip_key = chip_key.encode()
                with chip_env.begin() as chip_transaction:
                    raw = chip_transaction.get(chip_key)
                if raw is None:
                    raise ValueError(f"Missing NAC chip: {key.decode()}")
                chip = pickle.loads(raw)
                image = np.load(
                    BytesIO(zlib.decompress(chip["nac_blob"])), allow_pickle=False
                )
                valid = np.load(
                    BytesIO(zlib.decompress(chip["nac_valid_mask_blob"])),
                    allow_pickle=False,
                )
                if image.ndim == 2:
                    image = image[None]
                if valid.ndim == 2:
                    valid = valid[None]
                yield key.decode(), meta["split"], meta.get(
                    "center_lonlat"
                ), image, valid
    finally:
        if chip_env is not None:
            chip_env.close()
        metadata.close()


def build(args):
    if args.stride < 1 or args.stride > 512 or not 0 < args.min_valid_fraction <= 1:
        raise ValueError("Require stride in [1,512] and valid fraction in (0,1].")
    if (
        args.shard_samples < 1
        or args.shard_gib <= 0
        or (args.max_samples is not None and args.max_samples < 1)
    ):
        raise ValueError("Shard sizes and sample limits must be positive.")
    output = Path(args.output).resolve()
    work = output.with_name(output.name + ".inprogress")
    if output.exists() or work.exists():
        raise FileExistsError(f"Output or incomplete build already exists: {output}")
    work.mkdir(parents=True)
    counts, signature_counts, shard_counts = Counter(), Counter(), Counter()
    writers = {
        split: SignatureShardWriter(
            work / "shards" / split,
            args.shard_samples,
            int(args.shard_gib * 1024**3),
            args.zstd_level,
        )
        for split in ("train", "val", "test")
    }
    catalogues = {
        split: (work / f"samples-{split}.jsonl").open("w") for split in writers
    }
    records = lmdb_samples(args) if args.lmdb_root else raster_samples(args)
    identifier_index = sqlite3.connect(work / "build_ids.sqlite")
    identifier_index.execute("CREATE TABLE ids (sample_id TEXT PRIMARY KEY)")
    try:
        for sample_id, split, center, image, valid in records:
            if not re.fullmatch("[A-Za-z0-9_-]+", sample_id) or split not in writers:
                raise ValueError(
                    f"Invalid NAC sample identifier or split: {sample_id}/{split}"
                )
            try:
                identifier_index.execute("INSERT INTO ids VALUES(?)", (sample_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Duplicate NAC sample identifier: {sample_id}"
                ) from exc
            arrays = {"nac": image, "nac_valid_mask": valid}
            NacTransform()(arrays)
            if center is not None:
                center = np.asarray(center, dtype=float)
                if (
                    center.shape != (2,)
                    or not np.isfinite(center).all()
                    or abs(center[1]) > 90
                ):
                    raise ValueError(
                        "NAC coordinates must be finite [longitude,latitude] degrees."
                    )
                center = center.tolist()
            meta = dict(
                sample_id=sample_id,
                modalities=["nac"],
                modality_signature="nac",
                split=split,
                center_lonlat=center,
            )
            path = writers[split].write(sample_id, meta, arrays)
            relative = os.path.relpath(path, work)
            catalogues[split].write(json.dumps({**meta, "shard": relative}) + "\n")
            counts[f"saved_{split}"] += 1
            signature_counts[f"{split}:nac"] += 1
            shard_counts[relative] += 1
            if (
                args.max_samples is not None
                and sum(counts.values()) >= args.max_samples
            ):
                break
    finally:
        records.close()
        for writer in writers.values():
            writer.close()
        for handle in catalogues.values():
            handle.close()
        identifier_index.close()
    (work / "build_ids.sqlite").unlink()
    if not counts:
        raise ValueError("No valid native NAC samples were built.")
    summary = dict(
        format=NAC_FORMAT,
        counts=dict(counts),
        signature_counts=dict(signature_counts),
        shard_counts=dict(shard_counts),
        extreme_nodata_abs_limit=1.0e30,
        stored_sizes={"nac": [512, 512]},
        model_sizes={"nac": [512, 512]},
        compression="npy+zstd inside uncompressed tar",
        spatial_resampling=False,
        reflectance_mapping="uint8=clip(round(reflectance*255),0,255)",
        split_policy="explicit manifest split or deterministic group 80/10/10",
        split_seed=args.split_seed,
    )
    (work / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    work.replace(output)
    return summary


def validate(root):
    root = Path(root).resolve()
    summary = json.loads((root / "dataset.json").read_text())
    if (
        summary.get("format") != NAC_FORMAT
        or summary.get("extreme_nodata_abs_limit") != 1.0e30
    ):
        raise ValueError("Unsupported NAC dataset metadata.")
    counts, signatures = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="nac-validation-") as directory:
        index = sqlite3.connect(Path(directory) / "ids.sqlite")
        index.execute(
            "CREATE TABLE ids (sample_id TEXT PRIMARY KEY, shard TEXT, split TEXT, observed INTEGER DEFAULT 0)"
        )
        for split in ("train", "val", "test"):
            with (root / f"samples-{split}.jsonl").open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (
                        row["split"] != split
                        or row["modalities"] != ["nac"]
                        or row["modality_signature"] != "nac"
                        or Path(row["shard"]).parent != Path("shards") / split / "nac"
                    ):
                        raise ValueError("Invalid NAC catalogue row.")
                    try:
                        index.execute(
                            "INSERT INTO ids(sample_id,shard,split) VALUES(?,?,?)",
                            (row["sample_id"], row["shard"], split),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"Duplicate NAC sample identifier: {row['sample_id']}"
                        ) from exc
        for shard, split in index.execute(
            "SELECT DISTINCT shard,split FROM ids"
        ).fetchall():
            for raw in iter_tar_samples(root / shard):
                sample_id = raw.pop("sample_id")
                meta = raw.pop("meta")
                if index.execute(
                    "SELECT shard,split,observed FROM ids WHERE sample_id=?",
                    (sample_id,),
                ).fetchone() != (shard, split, 0):
                    raise ValueError(f"Unexpected or repeated NAC sample: {sample_id}")
                if (
                    meta.get("sample_id") != sample_id
                    or meta.get("split") != split
                    or meta.get("modalities") != ["nac"]
                    or meta.get("modality_signature") != "nac"
                    or set(raw) != {"nac", "nac_valid_mask"}
                ):
                    raise ValueError("NAC TAR metadata/fields disagree with catalogue.")
                NacTransform()(raw)
                center = meta.get("center_lonlat")
                if center is not None:
                    value = np.asarray(center, dtype=float)
                    if (
                        value.shape != (2,)
                        or not np.isfinite(value).all()
                        or abs(value[1]) > 90
                    ):
                        raise ValueError("Invalid NAC coordinates.")
                index.execute(
                    "UPDATE ids SET observed=1 WHERE sample_id=?", (sample_id,)
                )
                counts[shard] += 1
                signatures[f"{split}:nac"] += 1
        missing = index.execute("SELECT COUNT(*) FROM ids WHERE observed=0").fetchone()[
            0
        ]
        index.close()
        if missing:
            raise ValueError(f"{missing} NAC catalogue samples were absent from TARs.")
    if (
        dict(counts) != summary["shard_counts"]
        or dict(signatures) != summary["signature_counts"]
    ):
        raise ValueError("NAC dataset counts disagree with catalogues and shards.")
    for split in ("train", "val", "test"):
        if signatures.get(f"{split}:nac", 0) != summary["counts"].get(
            f"saved_{split}", 0
        ):
            raise ValueError("NAC split sample counts disagree.")
    actual = {
        str(path.relative_to(root)) for path in (root / "shards").glob("*/*/*.tar")
    }
    if actual != set(counts):
        raise ValueError("NAC dataset has unlisted/missing shards.")
    return {"samples": sum(counts.values()), "shards": len(counts)}


def compute_stats(root):
    paths = discover_signature_shards(root).get("nac", [])
    if not paths:
        raise ValueError("No native NAC training shards.")
    count, mean, m2, samples = 0, 0.0, 0.0, 0
    for sample in NacWebDataset(paths, transform=NacTransform()):
        values = sample["nac"][sample["nac_valid_mask"] > 0].numpy().astype(np.float64)
        size = values.size
        current_mean = values.mean()
        delta = current_mean - mean
        total = count + size
        m2 += ((values - current_mean) ** 2).sum() + delta**2 * count * size / total
        mean += delta * size / total
        count = total
        samples += 1
    std = (m2 / count) ** 0.5
    if std <= 0:
        raise ValueError("NAC observations have zero standard deviation.")
    return {
        "normalization": {"mean": mean, "std": std},
        "samples": samples,
        "valid_pixels": count,
        "domain": "stored uint8 divided by 255",
    }


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-manifest", type=Path)
    source.add_argument("--source-root", type=Path)
    source.add_argument("--lmdb-root", type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--glob", default="ROI_*/lronac/*.tif")
    create.add_argument("--stride", type=int, default=512)
    create.add_argument("--min-valid-fraction", type=float, default=0.6)
    create.add_argument(
        "--input-domain", choices=["reflectance", "uint8"], default="reflectance"
    )
    create.add_argument("--split-seed", default="lronac-native512-v1")
    create.add_argument("--shard-samples", type=int, default=1000)
    create.add_argument("--shard-gib", type=float, default=2.0)
    create.add_argument("--zstd-level", type=int, default=3)
    create.add_argument("--max-samples", type=int)
    for name in ("validate", "stats"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        if name == "stats":
            command.add_argument("--output", required=True, type=Path)
    return parser


def main():
    args = get_parser().parse_args()
    if args.command == "build":
        result = build(args)
    elif args.command == "validate":
        result = validate(args.root)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = compute_stats(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
