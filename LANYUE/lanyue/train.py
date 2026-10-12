"""Train the five-modal CE2-anchor v2 model on one device or with torchrun."""

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
from .data.transforms import FiveModalTransform
from .models import lanyue_base


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a pretraining JSON configuration (model, data, training).",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lanyue_base"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accum-iter", type=int)
    parser.add_argument("--signature", help="Optional canonical modality signature.")
    return parser


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"model", "data", "training"}:
        raise ValueError(
            "Config must contain exactly model, data, and training sections."
        )
    return config


def parameter_groups(model, weight_decay):
    skip = model.no_weight_decay()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = (
                no_decay
                if parameter.ndim == 1 or name.endswith(".bias") or name in skip
                else decay
            )
            target.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def learning_rate(progress, lr, min_lr, epochs, warmup_epochs):
    if warmup_epochs and progress < warmup_epochs:
        return lr * progress / warmup_epochs
    fraction = (progress - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return min_lr + (lr - min_lr) * 0.5 * (
        1 + math.cos(math.pi * min(max(fraction, 0), 1))
    )


def save_checkpoint(path, model, optimizer, scaler, completed_epoch, config):
    payload = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        epoch=completed_epoch,
        config=config,
        format="lanyue_ce2_anchor_v2",
    )
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, scaler, config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "lanyue_ce2_anchor_v2":
        raise ValueError("Resume expects a LANYUE v2 training checkpoint.")
    saved = checkpoint["config"]
    if saved["model"] != config["model"] or saved["data"] != config["data"]:
        raise ValueError("Model/data configuration differs from the resume checkpoint.")
    for key in ("batch_size", "accum_iter", "mask_ratio", "seed"):
        if saved["training"][key] != config["training"][key]:
            raise ValueError(f"Resume training setting differs: {key}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    # epoch is the number of fully completed epochs, already the next index.
    return int(checkpoint["epoch"])


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, settings, base_lr):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    task_totals = {}
    steps = len(loader)
    accumulation = settings["accum_iter"]
    amp = bool(settings["amp"] and device.type == "cuda")
    for step, batch in enumerate(loader):
        if step % accumulation == 0:
            lr = learning_rate(
                epoch + step / steps,
                base_lr,
                settings["min_lr"],
                settings["epochs"],
                settings["warmup_epochs"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
        samples = {
            name: (
                value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            )
            for name, value in batch.items()
        }
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, _, _ = model(samples, mask_ratio=settings["mask_ratio"])
        invalid = torch.tensor(int(not math.isfinite(loss.item())), device=device)
        if dist.is_initialized():
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        if invalid.item():
            raise FloatingPointError(
                f"Non-finite loss in epoch {epoch + 1}, step {step}."
            )
        total_loss += loss.item()
        module = model.module if hasattr(model, "module") else model
        task = getattr(module, "last_task", None)
        if task is not None:
            sums = task_totals.setdefault(task, [0.0, 0])
            sums[0] += loss.item()
            sums[1] += 1
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, steps - group_start)
        scaler.scale(loss / group_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            scaler.unscale_(optimizer)
            if settings["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["clip_grad"]
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if (step % settings["print_freq"] == 0 or step + 1 == steps) and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            print(
                f"Epoch {epoch + 1} [{step + 1}/{steps}] loss={loss.item():.6f} lr={lr:.3e}",
                flush=True,
            )
    totals = torch.tensor([total_loss, steps], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    stats = {"loss": (totals[0] / totals[1]).item(), "lr": lr, "steps_per_rank": steps}
    for task, values in sorted(task_totals.items()):
        reduced = torch.tensor(values, dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(reduced)
        stats[f"{task}_loss"] = (reduced[0] / reduced[1]).item()
    return stats


def main(args):
    config = load_config(args.config)
    settings = config["training"]
    for key in ("batch_size", "epochs", "num_workers", "accum_iter"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if (
        any(
            settings[k] < 1
            for k in ("batch_size", "epochs", "accum_iter", "save_freq", "print_freq")
        )
        or settings["num_workers"] < 0
    ):
        raise ValueError("Training counts must be positive; num_workers may be zero.")
    if not 0 <= settings["warmup_epochs"] < settings["epochs"]:
        raise ValueError(
            "Require 0 <= warmup_epochs < epochs; set warmup_epochs=0 for a one-epoch check."
        )
    if not 0 <= settings["mask_ratio"] < 1:
        raise ValueError("mask_ratio must lie in [0, 1).")
    if settings["blr"] <= 0 or settings["min_lr"] < 0 or settings["weight_decay"] < 0:
        raise ValueError("Invalid learning rate or weight decay.")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = (
        torch.device(args.device, local_rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use --device cpu with a small-model configuration."
            )
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed = int(settings["seed"]) + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        transform = FiveModalTransform(**config["data"])
        loader = SignatureLoader(
            args.data_root.resolve(),
            settings["batch_size"],
            num_workers=settings["num_workers"],
            signature=args.signature,
            transform=transform,
            pin_memory=device.type == "cuda",
            prefetch_factor=settings["prefetch_factor"],
            seed=seed,
        )
        model = lanyue_base(**config["model"]).to(device)
        effective_batch = settings["batch_size"] * settings["accum_iter"] * world
        base_lr = settings["blr"] * effective_batch / 256
        optimizer = torch.optim.AdamW(
            parameter_groups(model, settings["weight_decay"]),
            lr=base_lr,
            betas=(0.9, 0.95),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=settings["amp"] and device.type == "cuda"
        )
        start_epoch = (
            load_checkpoint(args.resume, model, optimizer, scaler, config)
            if args.resume
            else 0
        )
        if start_epoch >= settings["epochs"]:
            raise ValueError(
                f"Checkpoint has already completed {start_epoch} epochs; increase epochs to continue."
            )
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "config.json").write_text(
                json.dumps(config, indent=2) + "\n"
            )
            print(
                f"LANYUE: {sum(p.numel() for p in model.parameters()):,} parameters; effective batch={effective_batch}",
                flush=True,
            )
        raw_model = model
        if world > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        started = time.time()
        for epoch in range(start_epoch, settings["epochs"]):
            # Epoch-local seeds make transforms and masking reproducible when resuming.
            torch.manual_seed(seed + epoch * 1000)
            np.random.seed(seed + epoch * 1000)
            random.seed(seed + epoch * 1000)
            loader.set_epoch(epoch)
            stats = train_one_epoch(
                model, loader, optimizer, scaler, device, epoch, settings, base_lr
            )
            if rank == 0:
                stats["epoch"] = epoch + 1
                with (args.output_dir / "log.jsonl").open("a") as handle:
                    handle.write(json.dumps(stats) + "\n")
                if (epoch + 1) % settings["save_freq"] == 0 or epoch + 1 == settings[
                    "epochs"
                ]:
                    save_checkpoint(
                        args.output_dir / f"checkpoint-{epoch + 1}.pth",
                        raw_model,
                        optimizer,
                        scaler,
                        epoch + 1,
                        config,
                    )
        if rank == 0:
            print(f"Training complete in {time.time() - started:.1f}s.", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
