"""Measure valid-pixel normalization statistics in the training input domain."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lanyue.data.shards import Ce2AnchorWebDataset, discover_signature_shards
from lanyue.data.transforms import FiveModalTransform


def compute(root, max_samples=None, dem_clip=(-10000.0, 10000.0), sar_clip=None):
    summary = json.loads((Path(root) / "dataset.json").read_text())
    transform = FiveModalTransform(dem_clip=dem_clip, sar_clip=sar_clip)
    moments = defaultdict(lambda: [0, 0.0, 0.0])
    samples = 0
    for signature, paths in discover_signature_shards(root).items():
        overlay = summary.get("sar_overlay", {})
        overlay_paths = None
        if overlay.get("root") and "minirf_sar" in signature.split("+"):
            overlay_paths = {
                path: str(
                    Path(root)
                    / overlay["root"]
                    / "train"
                    / signature.replace("+", "__")
                    / Path(path).name
                )
                for path in paths
            }
        dataset = Ce2AnchorWebDataset(
            paths,
            expected_signature=signature,
            transform=transform,
            sar_overlay_shards=overlay_paths,
        )
        for sample in dataset:
            for name in signature.split("+"):
                values = (
                    sample[name][sample[f"{name}_valid_mask"] > 0]
                    .numpy()
                    .astype(np.float64)
                )
                count, mean, m2 = moments[name]
                current_count = values.size
                current_mean = values.mean()
                delta = current_mean - mean
                total = count + current_count
                m2 += (
                    (values - current_mean) ** 2
                ).sum() + delta**2 * count * current_count / total
                mean += delta * current_count / total
                moments[name] = [total, mean, m2]
            samples += 1
            if max_samples is not None and samples >= max_samples:
                break
        if max_samples is not None and samples >= max_samples:
            break
    if not samples:
        raise ValueError("No training samples found.")
    normalization = {}
    for name, (count, mean, m2) in moments.items():
        std = (m2 / count) ** 0.5
        if std <= 0:
            raise ValueError(
                f"{name}: constant observations have zero standard deviation."
            )
        normalization[name] = {"mean": mean, "std": std}
    return {
        "normalization": normalization,
        "samples": samples,
        "valid_pixels": {name: moment[0] for name, moment in moments.items()},
        "dem_clip": dem_clip,
        "sar_clip": sar_clip,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dem-clip", nargs=2, type=float, default=(-10000.0, 10000.0))
    parser.add_argument("--sar-clip", nargs=2, type=float)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compute(args.root, args.max_samples, args.dem_clip, args.sar_clip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
