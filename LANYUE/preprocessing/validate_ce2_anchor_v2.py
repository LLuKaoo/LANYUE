"""Validate catalogues, shards, validity masks, and sample counts."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from lanyue.data.shards import (
    iter_tar_samples,
    modality_signature,
    validate_ce2_anchor_modalities,
)
from lanyue.data.transforms import FiveModalTransform


def validate(root):
    root = Path(root).resolve()
    summary = json.loads((root / "dataset.json").read_text())
    if (
        summary.get("format") != "ce2_anchor_v2"
        or summary.get("extreme_nodata_abs_limit") != 1.0e30
    ):
        raise ValueError("Unsupported format or missing extreme-nodata protection.")
    declared_shards = summary.get("shard_counts")
    transform = FiveModalTransform()
    observed_signatures, observed_shards = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="lanyue-validation-") as temp:
        database = sqlite3.connect(str(Path(temp) / "samples.sqlite"))
        database.execute(
            "CREATE TABLE samples (sample_id TEXT PRIMARY KEY, shard TEXT, split TEXT, signature TEXT, observed INTEGER DEFAULT 0)"
        )
        for split in ("train", "val", "test"):
            with (root / f"samples-{split}.jsonl").open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["split"] != split:
                        raise ValueError(f"Wrong catalogue split: {row['sample_id']}")
                    modalities = validate_ce2_anchor_modalities(row["modalities"])
                    signature = modality_signature(modalities)
                    if signature != row["modality_signature"]:
                        raise ValueError("Noncanonical catalogue signature.")
                    expected_dir = Path("shards") / split / signature.replace("+", "__")
                    if Path(row["shard"]).parent != expected_dir:
                        raise ValueError(f"Unexpected shard location: {row['shard']}")
                    try:
                        database.execute(
                            "INSERT INTO samples(sample_id,shard,split,signature) VALUES(?,?,?,?)",
                            (row["sample_id"], row["shard"], split, signature),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"Duplicate sample id: {row['sample_id']}"
                        ) from exc
        database.commit()
        for shard, split, signature in database.execute(
            "SELECT DISTINCT shard,split,signature FROM samples"
        ).fetchall():
            path = root / shard
            if not path.is_file():
                raise FileNotFoundError(path)
            overlay_info = summary.get("sar_overlay", {})
            overlay = None
            if overlay_info.get("root") and "minirf_sar" in signature.split("+"):
                overlay = iter_tar_samples(
                    root
                    / overlay_info["root"]
                    / split
                    / signature.replace("+", "__")
                    / path.name
                )
            for raw in iter_tar_samples(
                path,
                skip_fields=(
                    {"minirf_sar", "minirf_sar_valid_mask"}
                    if overlay is not None
                    else ()
                ),
            ):
                sample_id = raw.pop("sample_id")
                if overlay is not None:
                    try:
                        replacement = next(overlay)
                    except StopIteration as exc:
                        raise ValueError("SAR overlay ended early.") from exc
                    if replacement.pop("sample_id") != sample_id:
                        raise ValueError("SAR overlay sample order mismatch.")
                    raw.update(replacement)
                entry = database.execute(
                    "SELECT shard,split,signature,observed FROM samples WHERE sample_id=?",
                    (sample_id,),
                ).fetchone()
                if entry != (shard, split, signature, 0):
                    raise ValueError(f"Unexpected or repeated TAR sample: {sample_id}")
                meta = raw.pop("meta")
                if (
                    meta["sample_id"] != sample_id
                    or meta["modality_signature"] != signature
                    or meta["split"] != split
                ):
                    raise ValueError(f"Metadata disagrees with catalogue: {sample_id}")
                modalities = validate_ce2_anchor_modalities(meta["modalities"])
                if modality_signature(modalities) != signature:
                    raise ValueError(f"Metadata modalities disagree: {sample_id}")
                expected_fields = set(modalities) | {
                    f"{name}_valid_mask" for name in modalities
                }
                if set(raw) != expected_fields:
                    raise ValueError(f"Unexpected array fields: {sample_id}")
                for name in modalities:
                    mask = np.asarray(raw[f"{name}_valid_mask"])
                    if not np.isin(mask, [0, 1]).all():
                        raise ValueError(f"{sample_id}/{name}: mask is not binary.")
                transform(raw)
                center = np.asarray(meta["center_lonlat"], dtype=np.float64)
                if (
                    center.shape != (2,)
                    or not np.isfinite(center).all()
                    or abs(center[1]) > 90
                ):
                    raise ValueError(f"Invalid center_lonlat: {sample_id}")
                database.execute(
                    "UPDATE samples SET observed=1 WHERE sample_id=?", (sample_id,)
                )
                observed_shards[shard] += 1
                observed_signatures[f"{split}:{signature}"] += 1
            if overlay is not None and next(overlay, None) is not None:
                raise ValueError("SAR overlay has extra samples.")
        missing = database.execute(
            "SELECT COUNT(*) FROM samples WHERE observed=0"
        ).fetchone()[0]
        database.close()
        if missing:
            raise ValueError(f"{missing} catalogue samples were absent from TARs.")
    if dict(observed_signatures) != summary["signature_counts"]:
        raise ValueError(
            "dataset.json signature counts disagree with TAR/catalogue counts."
        )
    for split in ("train", "val", "test"):
        count = sum(
            value
            for key, value in observed_signatures.items()
            if key.startswith(split + ":")
        )
        if count != int(summary["counts"].get(f"saved_{split}", 0)):
            raise ValueError(f"dataset.json saved_{split} disagrees with catalogue.")
    if declared_shards is not None and dict(observed_shards) != declared_shards:
        raise ValueError(
            "dataset.json shard counts disagree with TAR/catalogue counts."
        )
    all_shards = {
        str(path.relative_to(root)) for path in (root / "shards").glob("*/*/*.tar")
    }
    if all_shards != set(observed_shards):
        raise ValueError("Unlisted or missing TAR shards.")
    return {
        "samples": sum(observed_shards.values()),
        "shards": len(observed_shards),
        "signature_counts": dict(observed_signatures),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))


if __name__ == "__main__":
    main()
