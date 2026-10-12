"""Expand a v2 five-modal checkpoint with native NAC retention distillation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data.loader import SignatureLoader
from .data.nac import DistillationLoader, NacLoader, NacTransform
from .data.transforms import FiveModalTransform
from .distillation import (
    NACDistillation,
    build_teacher_student,
    checkpoint_digest,
    distillation_parameter_groups,
)
from .train import train_one_epoch

CHECKPOINT_FORMAT = "lanyue_nac_distillation_v1"


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--data-root", required=True, type=Path, help="CE2-anchor v2 retention data."
    )
    parser.add_argument(
        "--nac-root", required=True, type=Path, help="Native NAC TAR/Zstd dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a NAC JSON configuration (distillation, nac_data, training).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lanyue_nac"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accum-iter", type=int)
    return parser


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"distillation", "nac_data", "training"}
    if not required <= set(config) or set(config) - required - {"teacher_config"}:
        raise ValueError(
            "NAC config requires distillation, nac_data, and training sections."
        )
    # A saved run config can be reused; the supplied checkpoint remains authoritative.
    config.pop("teacher_config", None)
    return config


def save_checkpoint(path, student, optimizer, scaler, epoch, config, teacher_sha256):
    payload = dict(
        format=CHECKPOINT_FORMAT,
        model=student.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        epoch=epoch,
        config=config,
        teacher_sha256=teacher_sha256,
    )
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, student, optimizer, scaler, config, teacher_sha256):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("teacher_sha256") != teacher_sha256
    ):
        raise ValueError(
            "Resume requires a NAC distillation checkpoint with the identical frozen teacher."
        )
    saved = checkpoint["config"]
    for key in ("teacher_config", "distillation", "nac_data"):
        if saved[key] != config[key]:
            raise ValueError(f"NAC resume configuration differs: {key}")
    for key in ("batch_size", "accum_iter", "mask_ratio", "seed"):
        if saved["training"][key] != config["training"][key]:
            raise ValueError(f"NAC resume training setting differs: {key}")
    student.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"])


def main(args):
    config = load_config(args.config)
    settings, options = config["training"], config["distillation"]
    for key in ("batch_size", "epochs", "num_workers", "accum_iter"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if (
        any(
            settings[key] < 1
            for key in ("batch_size", "epochs", "accum_iter", "save_freq", "print_freq")
        )
        or settings["num_workers"] < 0
    ):
        raise ValueError("Training counts must be positive; workers may be zero.")
    if (
        not 0 <= settings["warmup_epochs"] < settings["epochs"]
        or not 0 <= settings["mask_ratio"] < 1
    ):
        raise ValueError("Invalid warmup duration or mask ratio.")
    if (
        not math.isfinite(settings["lr"])
        or settings["lr"] <= 0
        or not 0 <= settings["min_lr"] <= settings["lr"]
        or settings["weight_decay"] < 0
    ):
        raise ValueError("Invalid NAC learning rate or weight decay.")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = (
        torch.device("cuda", local_rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; use --device cpu with a small teacher for checks."
            )
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed = int(settings["seed"]) + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        teacher, student, teacher_config, report = build_teacher_student(
            args.teacher_checkpoint, options["nac_meters_per_pixel"]
        )
        if settings["mask_ratio"] != teacher_config["training"]["mask_ratio"]:
            raise ValueError(
                "Retention masking ratio must match the teacher pretraining configuration."
            )
        config["teacher_config"] = teacher_config
        digest = checkpoint_digest(args.teacher_checkpoint)
        teacher.to(device)
        student.to(device)
        old_loader = SignatureLoader(
            args.data_root.resolve(),
            settings["batch_size"],
            num_workers=settings["num_workers"],
            transform=FiveModalTransform(**teacher_config["data"]),
            pin_memory=device.type == "cuda",
            prefetch_factor=settings["prefetch_factor"],
            seed=seed,
        )
        nac_loader = NacLoader(
            args.nac_root.resolve(),
            settings["batch_size"],
            num_workers=settings["num_workers"],
            transform=NacTransform(**config["nac_data"]),
            pin_memory=device.type == "cuda",
            prefetch_factor=settings["prefetch_factor"],
            seed=seed,
        )
        loader = DistillationLoader(
            old_loader,
            nac_loader,
            options["old_batches_per_cycle"],
            options["nac_batches_per_cycle"],
            options["max_cycles"],
        )
        optimizer = torch.optim.AdamW(
            distillation_parameter_groups(
                student, settings["weight_decay"], options["shared_lr_multiplier"]
            ),
            lr=settings["lr"],
            betas=(0.9, 0.95),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=settings["amp"] and device.type == "cuda"
        )
        start_epoch = (
            load_checkpoint(args.resume, student, optimizer, scaler, config, digest)
            if args.resume
            else 0
        )
        if start_epoch >= settings["epochs"]:
            raise ValueError(
                "Checkpoint already completed the requested number of epochs."
            )
        objective = NACDistillation(
            student, teacher, options["token_weight"], options["pool_weight"]
        ).to(device)
        if world > 1:
            objective = DistributedDataParallel(
                objective,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "config.json").write_text(
                json.dumps(config, indent=2) + "\n"
            )
            (args.output_dir / "initialization.json").write_text(
                json.dumps({**report, "teacher_sha256": digest}, indent=2) + "\n"
            )
            print(
                f'NAC distillation: {report}; old:NAC={options["old_batches_per_cycle"]}:{options["nac_batches_per_cycle"]}; steps/rank={len(loader)}',
                flush=True,
            )
        started = time.time()
        for epoch in range(start_epoch, settings["epochs"]):
            torch.manual_seed(seed + epoch * 1000)
            np.random.seed(seed + epoch * 1000)
            random.seed(seed + epoch * 1000)
            loader.set_epoch(epoch)
            stats = train_one_epoch(
                objective,
                loader,
                optimizer,
                scaler,
                device,
                epoch,
                settings,
                settings["lr"],
            )
            if rank == 0:
                stats["epoch"] = epoch + 1
                with (args.output_dir / "log.jsonl").open("a") as handle:
                    handle.write(json.dumps(stats) + "\n")
                if (epoch + 1) % settings["save_freq"] == 0 or epoch + 1 == settings[
                    "epochs"
                ]:
                    save_checkpoint(
                        args.output_dir / f"checkpoint-{epoch+1}.pth",
                        student,
                        optimizer,
                        scaler,
                        epoch + 1,
                        config,
                        digest,
                    )
        if rank == 0:
            print(
                f"NAC distillation complete in {time.time()-started:.1f}s.", flush=True
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
