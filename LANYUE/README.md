# LANYUE-FM

Lunar masked pretraining with a shared HiViT encoder, physical-scale embeddings,
spherical geolocation and structural saliency guidance.

## Model overview

- Five CE2-anchored sources: CE-2 DOM/DEM, LROC WAC, SLDEM and Mini-RF.
- Native-resolution NAC extension through frozen-teacher distillation.
- Dense features for any supported modality subset, including WAC/DEM-only
  and SAR-only inputs.

**Pretrained weights:** public download links are pending.

## Installation

Run from `LANYUE-FM/`, with a PyTorch build compatible with your CUDA version:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-preprocessing.txt
```

## Pretraining

Prepare a dataset using [`LANYUE-Pretrain`](../LANYUE-Pretrain/README.md).
Copy its measured normalization statistics into `data.normalization` in the config.
Training configurations are supplied externally through `--config`; this directory
does not bundle a `configs/` folder. A pretraining JSON must contain `model`,
`data` and `training` sections with the required run settings.

```bash
python -m lanyue.train \
  --data-root /data/ce2_anchor_v2 \
  --config /data/configs/pretrain.json \
  --output-dir outputs/lanyue_base

# Multiple GPUs
DATA_ROOT=/data/ce2_anchor_v2 CONFIG=/data/configs/pretrain.json \
NUM_GPUS=8 bash scripts/pretrain.sh
```

Historical paper checkpoints use different
canvases and decoder depths; use the matching configuration when loading them.

## NAC distillation

```bash
python -m lanyue.distill \
  --teacher-checkpoint /models/lanyue_v2/checkpoint-100.pth \
  --data-root /data/ce2_anchor_v2 \
  --nac-root /data/nac_native512 \
  --config /data/configs/nac_distill.json \
  --output-dir outputs/lanyue_nac
```

The NAC JSON must contain `distillation`, `nac_data` and `training` sections.
The teacher checkpoint supplies the old-modality model and preprocessing settings.
Use native 512 × 512 NAC data, prepared through the
[NAC data tools](../LANYUE-Pretrain/data_collection/README.md#native-nac).

## Using the model

For a five-modal v2 checkpoint written by `lanyue.train`:

```python
import torch
from lanyue.models import lanyue_base
from lanyue.features import encode_features

checkpoint = torch.load("/models/lanyue_v2.pth", map_location="cpu", weights_only=True)
model = lanyue_base(**checkpoint["config"]["model"])
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

# Illustrative inputs; use the checkpoint/task preprocessing for real images.
inputs = {"nasa_dom": torch.randn(1, 1, 224, 224),
          "nasa_dem": torch.randn(1, 1, 224, 224)}
with torch.no_grad():
    tokens = encode_features(model, inputs)
    descriptor = torch.cat([x.mean(1) for x in tokens.values()], dim=-1)
```

Inputs use `[B,1,H,W]` tensors, with spatial sizes divisible by the modality
patch size. Coordinates, when available, use `center_lonlat` in longitude,
latitude order and degrees. Inference from a v2 checkpoint uses its saved model
configuration and does not require a separate training JSON. NAC checkpoints
must be instantiated with `lanyue_nac` using their saved teacher model settings.

## License

[MIT](LICENSE). Attribution is retained in [NOTICE](NOTICE).
