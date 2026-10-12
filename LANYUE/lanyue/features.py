"""Dense downstream encoding with interpolated positions and strict core loading."""
import torch
from torch.nn import functional as F


DECODER_PREFIXES = ("decoder_", "mask_tokens.")
LEGACY_UNUSED_PREFIXES = ("patch_embed.", "fc_norm.", "head.", "projection_heads.")
LEGACY_UNUSED_KEYS = {"absolute_pos_embed", "relative_position_index"}


def load_encoder_weights(model, state_dict):
    """Require every encoder tensor; never skip shape mismatches.

    Pass checkpoint['model'] explicitly. Decoders and obsolete source-only heads
    are ignored. This is an encoder import, not training-state resume.
    """
    current = model.state_dict()
    required = {k for k in current if not k.startswith(DECODER_PREFIXES)}
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    missing = sorted(required - cleaned.keys())
    mismatched = [key for key in required & cleaned.keys()
                  if current[key].shape != cleaned[key].shape]
    unexpected = sorted(key for key in cleaned if key not in current
                        and not key.startswith(DECODER_PREFIXES + LEGACY_UNUSED_PREFIXES)
                        and key not in LEGACY_UNUSED_KEYS
                        and not (key == "unknown_scale_embed" and model.scale_drop_prob == 0))
    if missing or mismatched or unexpected:
        raise ValueError(f"Encoder checkpoint incompatible: missing={missing}, "
                         f"shape_mismatch={mismatched}, unexpected={unexpected}")
    model.load_state_dict({key: cleaned[key] for key in required}, strict=False)
    return {"loaded_encoder_tensors": len(required),
            "ignored_tensors": sorted(set(cleaned) - required)}


def encode_features(model, inputs):
    """Return all dense tokens for any available subset of pretrained modalities.

    Inputs must already be normalized and padded/resized for the downstream
    protocol. Spatial dimensions must be multiples of the modality patch size.
    Unlike pretraining, no CE2 anchor is required. Use eval() for deterministic
    conditioning. This follows the downstream dense path, including all tokens;
    it does not implement validity-masked attention or resize images.
    """
    active = model.get_active_modalities(inputs)
    metadata_keep = model._sample_metadata_keep_mask(inputs, active)
    tokenized = {}
    positions = []
    lengths = {}
    for name in active:
        image = inputs[name]
        patch = model.modality_patch_sizes[name]
        if image.ndim != 4 or image.shape[1] != 1 or any(n % patch for n in image.shape[-2:]):
            raise ValueError(f"{name}: require B,1,H,W with spatial multiples of {patch}")
        tokenized[name] = model.patch_embeds[name](image)
        lengths[name] = tokenized[name].shape[1]
        grid = (image.shape[-2] // patch, image.shape[-1] // patch)
        source_grid = model.patches_resolution[name]
        pos = model.absolute_pos_embeds[name]
        if grid != source_grid:
            pos = pos.reshape(1, *source_grid, model.num_features).permute(0, 3, 1, 2)
            pos = F.interpolate(pos.float(), size=grid, mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, lengths[name], model.num_features)
        positions.append(pos.to(device=image.device, dtype=tokenized[name].dtype) + model._modality_embed_for(name, image.shape[0])
                         + model._physical_scale_embed_for(name, image.shape[0])
                         + model._spherical_geo_embed_for(inputs, name, metadata_keep))
    x = torch.cat([tokenized[name] for name in active], dim=1)
    for block in model.blocks[:-model.num_main_blocks]:
        x = block(x)
    x = model.pos_drop(x[..., 0, 0, :] + torch.cat(positions, dim=1))
    for block in model.blocks[-model.num_main_blocks:]:
        x = block(x, None)
    x = model.norm(x)
    result = {}
    start = 0
    for name in active:
        result[name] = x[:, start:start + lengths[name]]
        start += lengths[name]
    return result
