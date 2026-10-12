"""Homogeneous signature batches with equal, finite epoch lengths across ranks."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .shards import (
    Ce2AnchorWebDataset,
    discover_signature_shards,
    validate_ce2_anchor_modalities,
)
from .transforms import FiveModalTransform


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def signature_batch_quota(shards, counts, epoch, world_size, num_workers, batch_size):
    """Minimum full-batch count across ranks, including per-worker drop_last."""
    shuffled = list(shards)
    random.Random(0xCE2 + epoch).shuffle(shuffled)
    workers = max(num_workers, 1)
    rank_batches = []
    for rank in range(world_size):
        assigned = shuffled[rank::world_size]
        rank_batches.append(
            sum(
                sum(counts[path] for path in assigned[w::workers]) // batch_size
                for w in range(workers)
            )
        )
    return min(rank_batches)


class SignatureLoader:
    """Visit each signature for its quota without restarting exhausted readers.

    A quota is calculated from actual shard counts for every epoch. Shards are
    shuffled identically on every rank, and each rank uses the minimum number
    of full batches available to any rank. Remainders rotate with shard order.
    """

    dataset_format = "ce2_anchor_v2"
    dataset_class = Ce2AnchorWebDataset
    transform_class = FiveModalTransform

    @staticmethod
    def validate_signature(name):
        if "+".join(validate_ce2_anchor_modalities(name.split("+"))) != name:
            raise ValueError(f"Noncanonical signature directory: {name}")

    def __init__(
        self,
        root,
        batch_size,
        num_workers=0,
        split="train",
        signature=None,
        transform=None,
        pin_memory=True,
        prefetch_factor=2,
        seed=0,
    ):
        if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
            raise ValueError("Invalid batch size, worker count, or prefetch factor.")
        self.root = Path(root)
        summary = json.loads((self.root / "dataset.json").read_text())
        if summary.get("format") != self.dataset_format:
            raise ValueError(f"Expected dataset format {self.dataset_format}.")
        if float(summary.get("extreme_nodata_abs_limit", 0)) != 1.0e30:
            raise ValueError(
                "Dataset lacks extreme-nodata protection; rebuild or repair before training."
            )
        self.shards = discover_signature_shards(root, split)
        for name in self.shards:
            self.validate_signature(name)
        if signature:
            if signature not in self.shards:
                raise ValueError(f"Requested signature {signature!r} is absent.")
            self.shards = {signature: self.shards[signature]}
        if not self.shards:
            raise ValueError(f"No TAR shards for split {split!r} below {root}.")
        self.counts = {
            str(self.root / path): int(count)
            for path, count in summary.get("shard_counts", {}).items()
        }
        if any(
            path not in self.counts for paths in self.shards.values() for path in paths
        ):
            # Compatibility with v2 datasets created before shard_counts existed.
            self.counts = Counter()
            with (self.root / f"samples-{split}.jsonl").open() as handle:
                for line in handle:
                    if line.strip():
                        self.counts[str(self.root / json.loads(line)["shard"])] += 1
        if any(
            self.counts.get(path, 0) < 1
            for paths in self.shards.values()
            for path in paths
        ):
            raise ValueError("Every discovered shard needs a positive catalogue count.")
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loaders = {}
        for name, paths in self.shards.items():
            overlay = summary.get("sar_overlay", {})
            overlay_paths = None
            if overlay.get("root") and "minirf_sar" in name.split("+"):
                overlay_paths = {
                    path: str(
                        self.root
                        / overlay["root"]
                        / split
                        / name.replace("+", "__")
                        / Path(path).name
                    )
                    for path in paths
                }
                if any(not Path(path).is_file() for path in overlay_paths.values()):
                    raise FileNotFoundError("SAR overlay shards are missing.")
            dataset = self.dataset_class(
                paths,
                expected_signature=name,
                transform=transform or self.transform_class(),
                sar_overlay_shards=overlay_paths,
            )
            kwargs = dict(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=pin_memory,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed),
                persistent_workers=False,
            )
            if num_workers:
                kwargs["prefetch_factor"] = prefetch_factor
            self.loaders[name] = DataLoader(**kwargs)
        self.seed = int(seed)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.quotas = {
            name: signature_batch_quota(
                paths,
                self.counts,
                self.epoch,
                self.world_size,
                self.num_workers,
                self.batch_size,
            )
            for name, paths in self.shards.items()
        }
        empty = [name for name, quota in self.quotas.items() if quota == 0]
        if empty:
            raise ValueError(
                f"Signatures cannot supply a full batch on every rank: {empty}. "
                "Reduce batch size/workers/world size or build more shards."
            )
        for loader in self.loaders.values():
            loader.dataset.set_epoch(self.epoch)
            loader.generator.manual_seed(self.seed + self.epoch)

    def __len__(self):
        return sum(self.quotas.values())

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        scores = {name: 0 for name in self.quotas}
        remaining = dict(self.quotas)
        total = len(self)
        for _ in range(total):
            for name, weight in self.quotas.items():
                scores[name] += weight
            name = max(
                (name for name in remaining if remaining[name]),
                key=lambda key: scores[key],
            )
            scores[name] -= total
            remaining[name] -= 1
            try:
                yield next(iterators[name])
            except StopIteration as exc:
                raise RuntimeError(
                    f"{name}: actual samples disagree with catalogue counts. Run dataset validation."
                ) from exc
