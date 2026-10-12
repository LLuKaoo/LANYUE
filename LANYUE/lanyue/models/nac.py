"""NAC student extension and masked retention distillation from the source model."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .mae import LanyueMAE
from ..data.shards import CE2_ANCHOR_V2_LAYOUTS, ModalityLayout


class LanyueNAC(LanyueMAE):
    """Extend the v2 model with a native-resolution NAC reconstruction branch."""

    accepts_nac_only = True

    def __init__(self, nac_meters_per_pixel=1.0, **kwargs):
        if not 0 < float(nac_meters_per_pixel) < float("inf"):
            raise ValueError("NAC pixel scale must be finite and positive.")
        self.modality_layouts = {
            **CE2_ANCHOR_V2_LAYOUTS,
            "nac": ModalityLayout(
                (512, 512), (512, 512), 16, float(nac_meters_per_pixel)
            ),
        }
        if kwargs.get("modality_img_sizes") is not None:
            sizes = dict(kwargs["modality_img_sizes"])
            sizes.setdefault("nac", sizes["ce2_dom"])
            kwargs["modality_img_sizes"] = sizes
        decoder_depths = dict(kwargs.get("modality_decoder_depths") or {})
        decoder_depths.setdefault(
            "nac", decoder_depths.get("ce2_dom", kwargs.get("decoder_depth", 6))
        )
        kwargs["modality_decoder_depths"] = decoder_depths
        super().__init__(**kwargs)
        self.last_aux_losses = {}

    def forward(
        self,
        batch_dict,
        mask_ratio=0.75,
        distill_targets=None,
        distill_token_weight=1.0,
        distill_pool_weight=0.25,
        **kwargs,
    ):
        if distill_targets is not None:
            return self.forward_distillation_loss(
                batch_dict, distill_targets, distill_token_weight, distill_pool_weight
            )
        return super().forward(batch_dict, mask_ratio=mask_ratio, **kwargs)

    def _prepare_dense_modality(self, inputs, name):
        if f"{name}_valid_mask" in inputs:
            padded = inputs[name]
            valid_mask = inputs[f"{name}_valid_mask"].to(
                device=padded.device, dtype=padded.dtype
            )
            expected_size = self._to_2tuple(self.modality_img_sizes[name])
            if tuple(padded.shape[-2:]) != expected_size:
                raise ValueError(
                    f"{name}: configured padded size {expected_size}, got "
                    f"{tuple(padded.shape[-2:])}."
                )
        else:
            padded, valid_mask = self.pad_modality(inputs[name], name)
        return padded, valid_mask

    def forward_distillation_loss(
        self, batch_dict, teacher_outputs, token_weight=1.0, pool_weight=0.25
    ):
        """Keep old-modality representations close to a frozen A5 teacher."""
        modalities = [
            name
            for name in self.modalities
            if name != "nac" and name in batch_dict and name in teacher_outputs
        ]
        if not modalities:
            raise ValueError("Distillation batch contains no shared old modalities.")

        metadata_keep_mask = self._sample_metadata_keep_mask(batch_dict, modalities)
        tokenized = {}
        for name in modalities:
            padded, valid_mask = self._prepare_dense_modality(batch_dict, name)
            ids_keep = teacher_outputs[name]["ids_keep"].detach()
            tokens = self.patch_embeds[name](padded)
            visible_tokens = torch.gather(
                tokens,
                dim=1,
                index=ids_keep[:, :, None, None, None].expand(
                    -1, -1, *tokens.shape[2:]
                ),
            )
            tokenized[name] = {
                "padded": padded,
                "valid_mask": valid_mask,
                "visible_tokens": visible_tokens,
                "ids_keep": ids_keep,
                "num_visible": ids_keep.size(1),
                "spherical_geo_embed": self._spherical_geo_embed_for(
                    batch_dict,
                    name,
                    metadata_keep_mask=metadata_keep_mask,
                ),
            }
        student_latents = self.forward_shared_encoder(tokenized, modalities)

        token_losses = []
        pool_losses = []
        for name in modalities:
            student_tokens = student_latents[name].float()
            teacher_tokens = teacher_outputs[name]["tokens"].detach().float()
            if student_tokens.shape != teacher_tokens.shape:
                raise ValueError(
                    f"Teacher/student token mismatch for {name}: "
                    f"{tuple(teacher_tokens.shape)} vs {tuple(student_tokens.shape)}"
                )

            student_unit = F.normalize(student_tokens, dim=-1)
            teacher_unit = F.normalize(teacher_tokens, dim=-1)
            cosine = 1.0 - (student_unit * teacher_unit).sum(dim=-1)
            token_losses.append(cosine.mean())

            student_pool = student_tokens.mean(dim=1)
            teacher_pool = teacher_tokens.mean(dim=1)
            pool_losses.append(
                (
                    1.0
                    - (
                        F.normalize(student_pool, dim=-1)
                        * F.normalize(teacher_pool, dim=-1)
                    ).sum(dim=-1)
                ).mean()
            )

        token_loss = torch.stack(token_losses).mean()
        pool_loss = torch.stack(pool_losses).mean()
        total_loss = float(token_weight) * token_loss + float(pool_weight) * pool_loss
        self.last_losses = {"distill": total_loss}
        self.last_aux_losses = {
            "distill_token": token_loss.detach(),
            "distill_pool": pool_loss.detach(),
        }
        return total_loss, {}, {}


def lanyue_nac(**kwargs):
    """Construct a v2 student with the NAC extension."""
    return LanyueNAC(**kwargs)
