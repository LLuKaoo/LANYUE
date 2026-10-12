#!/usr/bin/env python3
"""Merge independently built CE2-anchor v2 part roots without copying TARs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def merge(parts_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    work = output.with_name(output.name + ".inprogress")
    if work.exists():
        raise FileExistsError(f"Incomplete merge exists: {work}")
    parts = sorted(
        path for path in parts_root.glob("part-*") if (path / "dataset.json").is_file()
    )
    if not parts:
        raise ValueError(f"No completed part-* datasets below {parts_root}")
    work.mkdir(parents=True)
    counts = Counter()
    signature_counts = Counter()
    shard_counts = Counter()
    catalogue_handles = {
        split: open(work / f"samples-{split}.jsonl", "w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    template = None
    try:
        for part in parts:
            summary = json.loads((part / "dataset.json").read_text())
            if summary.get("sar_overlay"):
                raise ValueError("Materialize SAR overlays before merging parts.")
            if template is not None:
                for key in (
                    "format",
                    "stored_sizes",
                    "model_sizes",
                    "extreme_nodata_abs_limit",
                    "compression",
                ):
                    if summary.get(key) != template.get(key):
                        raise ValueError(
                            f"Incompatible part metadata: {part.name}/{key}"
                        )
            template = template or summary
            counts.update(summary.get("counts", {}))
            signature_counts.update(summary.get("signature_counts", {}))
            part_name = part.name
            for split in ("train", "val", "test"):
                source_list = part / f"samples-{split}.jsonl"
                if not source_list.is_file():
                    continue
                with open(source_list, "r", encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        source_tar = part / row["shard"]
                        signature_dir = (
                            work
                            / "shards"
                            / split
                            / row["modality_signature"].replace("+", "__")
                        )
                        signature_dir.mkdir(parents=True, exist_ok=True)
                        target_tar = signature_dir / f"{part_name}-{source_tar.name}"
                        if not target_tar.exists():
                            os.symlink(source_tar.resolve(), target_tar)
                        row["shard"] = os.path.relpath(target_tar, work)
                        shard_counts[row["shard"]] += 1
                        catalogue_handles[split].write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
    finally:
        for handle in catalogue_handles.values():
            handle.close()
    merged = dict(template)
    merged["counts"] = dict(counts)
    merged["signature_counts"] = dict(signature_counts)
    merged["shard_counts"] = dict(shard_counts)
    merged["parts"] = [str(path.resolve()) for path in parts]
    merged["merge_mode"] = "symlink_no_tar_copy"
    merged["catalogues"] = {"jsonl": ["train", "val", "test"], "parquet": []}
    (work / "dataset.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(work, output)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.parts_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
