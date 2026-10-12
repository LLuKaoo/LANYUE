"""Frozen-teacher retention loss, NAC initialization, and trainable roles."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .models import lanyue_base, lanyue_nac

BRANCH_ROOTS = (
    "patch_embeds",
    "absolute_pos_embeds",
    "modality_embed",
    "decoder_embeds",
    "mask_tokens",
    "decoder_pos_embeds",
    "decoder_blocks",
    "decoder_norms",
    "decoder_preds",
)


def is_nac_branch(name):
    return any(
        name == f"{root}.nac" or name.startswith(f"{root}.nac.")
        for root in BRANCH_ROOTS
    )


def is_shared_encoder(name):
    return (
        name.startswith(("blocks.", "norm.", "spherical_geo_embed."))
        or name == "unknown_geo_embed"
    )


def configure_distillation(student):
    # Match the source stage: NAC position tensors become trainable as well.
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(is_shared_encoder(name) or is_nac_branch(name))
    return {
        name: sum(
            p.numel()
            for key, p in student.named_parameters()
            if p.requires_grad and predicate(key)
        )
        for name, predicate in (("shared", is_shared_encoder), ("nac", is_nac_branch))
    }


def distillation_parameter_groups(student, weight_decay, shared_lr_multiplier):
    if not math.isfinite(shared_lr_multiplier) or shared_lr_multiplier <= 0:
        raise ValueError("shared_lr_multiplier must be finite and positive.")
    groups = {}
    skip = student.no_weight_decay()
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = shared_lr_multiplier if is_shared_encoder(name) else 1.0
        decay = (
            0.0
            if parameter.ndim == 1 or name.endswith(".bias") or name in skip
            else weight_decay
        )
        group = groups.setdefault(
            (multiplier, decay),
            {"params": [], "weight_decay": decay, "lr_scale": multiplier},
        )
        group["params"].append(parameter)
    return list(groups.values())


def initialize_nac_from_ce2(student):
    state = student.state_dict()
    remapped = {}
    for key, target in state.items():
        if not is_nac_branch(key):
            continue
        source = key.replace(".nac", ".ce2_dom", 1)
        if source not in state or state[source].shape != target.shape:
            raise ValueError(
                f"Cannot initialize the entire NAC branch from CE2 DOM: {key}"
            )
        remapped[key] = state[source].clone()
    if not remapped:
        raise ValueError("Student has no NAC branch to initialize.")
    student.load_state_dict(remapped, strict=False)
    return tuple(remapped)


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_teacher_student(path, nac_meters_per_pixel=1.0):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "lanyue_ce2_anchor_v2":
        raise ValueError(
            "NAC distillation requires a LANYUE CE2-anchor v2 five-modal checkpoint."
        )
    config = checkpoint["config"]
    teacher = lanyue_base(**config["model"])
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.requires_grad_(False).eval()
    student = lanyue_nac(nac_meters_per_pixel=nac_meters_per_pixel, **config["model"])
    incompatible = student.load_state_dict(checkpoint["model"], strict=False)
    if (
        incompatible.unexpected_keys
        or not incompatible.missing_keys
        or any(not is_nac_branch(key) for key in incompatible.missing_keys)
    ):
        raise ValueError(
            f"Checkpoint does not exactly initialize the five old branches: {incompatible}"
        )
    initialized = initialize_nac_from_ce2(student)
    roles = configure_distillation(student)
    return (
        teacher,
        student,
        config,
        {"nac_tensors_initialized": len(initialized), "trainable_parameters": roles},
    )


class NACDistillation(nn.Module):
    """Use old sensor batches for retention and NAC-only batches for MAE.

    The teacher chooses visible indices once. The student reuses those indices,
    so token-level cosine comparisons always compare corresponding patches.
    """

    def __init__(self, student, teacher, token_weight=1.0, pool_weight=0.25):
        super().__init__()
        if (
            any(not math.isfinite(w) or w < 0 for w in (token_weight, pool_weight))
            or token_weight + pool_weight <= 0
        ):
            raise ValueError(
                "Distillation weights must be finite, nonnegative, and not both zero."
            )
        self.student = student
        self.teacher = teacher.requires_grad_(False).eval()
        self.token_weight = float(token_weight)
        self.pool_weight = float(pool_weight)
        self.last_task = None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, samples, mask_ratio=0.75):
        if "nac" in samples:
            if any(name in samples for name in self.teacher.modalities):
                raise ValueError(
                    "A NAC batch must be homogeneous and contain NAC only."
                )
            self.last_task = "nac_mae"
            return self.student(samples, mask_ratio=mask_ratio)
        if "ce2_dom" not in samples:
            raise ValueError("Retention batches require the CE2 DOM anchor.")
        self.last_task = "distill"
        with torch.no_grad():
            targets = self.teacher.masked_distillation_targets(
                samples, mask_ratio=mask_ratio
            )
        return self.student(
            samples,
            distill_targets=targets,
            distill_token_weight=self.token_weight,
            distill_pool_weight=self.pool_weight,
        )
