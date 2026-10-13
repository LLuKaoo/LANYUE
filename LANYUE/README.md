# LANYUE

Lunar masked pretraining with a shared HiViT encoder, physical-scale embeddings,
spherical geolocation and structural saliency guidance.

## Model overview

![LANYUE model architecture](assets/lanyue-architecture.png)

- Five CE2-anchored sources: CE-2 DOM/DEM, LROC WAC, SLDEM and Mini-RF.
- Native-resolution NAC extension through frozen-teacher distillation.
- Dense features for any supported modality subset, including WAC/DEM-only
  and SAR-only inputs.

**Pretrained weights:** [Hugging Face](https://huggingface.co/LLukaoo/LANYUE) · [Download KD3](https://huggingface.co/LLukaoo/LANYUE/resolve/main/lanyue-kd3.pth?download=true).

The released checkpoint supports all six modalities, including NAC, and
corresponds to distillation epoch 3. Its matching model configuration and
normalization statistics are included in the checkpoint.

## Installation

Run from `LANYUE/`, with a PyTorch build compatible with your CUDA version:

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

Load the released six-modality KD3 checkpoint from Hugging Face:

```python
import torch
from huggingface_hub import hf_hub_download
from lanyue.models import lanyue_nac
from lanyue.features import encode_features

path = hf_hub_download("LLukaoo/LANYUE", "lanyue-kd3.pth")
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model = lanyue_nac(**checkpoint["config"]["model"])
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

# Illustrative normalized inputs; apply the appropriate image preprocessing.
inputs = {"nasa_dom": torch.randn(1, 1, 48, 88),
          "nasa_dem": torch.randn(1, 1, 80, 144)}
with torch.no_grad():
    tokens = encode_features(model, inputs)
    descriptor = torch.cat([x.mean(1) for x in tokens.values()], dim=-1)
```

Inputs use `[B,1,H,W]` tensors with spatial sizes divisible by the modality
patch size. Coordinates, when available, use `center_lonlat` in longitude,
latitude order and degrees. Any supported modality subset can be supplied,
including NAC alone. Use the checkpoint's saved model configuration and the
appropriate normalization; a separate training JSON is not required for inference.

## License

[MIT](LICENSE). Attribution is retained in [NOTICE](NOTICE).
