# LANYUE-FM

基于共享HiViT编码器的月球掩码预训练，结合物理尺度、球面地理位置编码
和结构显著性引导，并支持NAC蒸馏扩展。

[English](README.md)

## 安装

在本目录运行，选择与本机CUDA兼容的PyTorch：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-preprocessing.txt
```

## 预训练

使用[数据工具](../LANYUE-Pretrain/README_zh-CN.md)准备数据，并把统计结果写入
配置的 `data.normalization`。
训练配置通过 `--config`从外部提供；当前目录不包含 `configs/`。
预训练JSON需包含 `model`、`data`、`training`三部分及所需运行参数。

```bash
python -m lanyue.train \
  --data-root /data/ce2_anchor_v2 \
  --config /data/configs/pretrain.json \
  --output-dir outputs/lanyue_base

DATA_ROOT=/data/ce2_anchor_v2 CONFIG=/data/configs/pretrain.json \
NUM_GPUS=8 bash scripts/pretrain.sh
```

历史论文权重的输入画布和decoder深度不同，加载时需匹配对应模型参数。

## NAC蒸馏

```bash
python -m lanyue.distill \
  --teacher-checkpoint /models/lanyue_v2/checkpoint-100.pth \
  --data-root /data/ce2_anchor_v2 \
  --nac-root /data/nac_native512 \
  --config /data/configs/nac_distill.json \
  --output-dir outputs/lanyue_nac
```

NAC JSON需包含 `distillation`、`nac_data`、`training`三部分，旧模态模型和
预处理参数来自教师checkpoint。NAC数据构建见[数据工具](../LANYUE-Pretrain/data_collection/README.md#native-nac)。

## 模型使用

`lanyue.features.encode_features`返回各模态密集token，支持任意已训练模态
子集。输入为预处理后的 `[B,1,H,W]`，空间尺寸需被对应patch大小整除；
地理坐标通过 `center_lonlat`提供，顺序为经度、纬度，单位为度。

预训练权重下载链接待补充。加载与特征提取示例见[English README](README.md)。
v2 checkpoint保存了模型配置，推理不需要独立的训练JSON；NAC权重则需按
保存的教师模型参数实例化 `lanyue_nac`。

## 许可证

[MIT](LICENSE)，来源声明见[NOTICE](NOTICE)。
