# ResNet: Paper-Accurate Implementation with Modern Tools

This repository provides a modular and reproducible implementation of **ResNet** and **PlainNet** as described in [He et al. (2015)](https://arxiv.org/abs/1512.03385). It is designed to support ImageNet-scale training using modern PyTorch infrastructure: **Lightning**, **Hydra**, **WebDataset**, and **W&B**.

---

## Highlights

- From-scratch **ResNet** and **PlainNet** implementations (BasicBlock and Bottleneck)
- Trains **ResNet-50 on ImageNet** using [WebDataset](https://github.com/webdataset/webdataset) format
- Modular configuration using **Hydra**
- Scalable training with **PyTorch Lightning** (DDP, AMP, checkpointing)
- Integrated with **Weights & Biases** for logging and tracking
- Export script to extract raw `state_dict` weights from Lightning checkpoint
- Pre-commit integration: Black, Flake8, isort, nbQA

---

## Training Environment

ImageNet training was performed on the following GCP instance:

- **Instance Type**: `a2-highgpu-2g`
- **GPUs**: 2 × NVIDIA A100 (40GB each)
- **CPU**: 24 vCPUs (12 physical cores with hyperthreading)
- **RAM**: 170 GB system memory

This configuration is sufficient for full-batch ImageNet training using WebDataset and mixed precision.

---

## How to Use

### 1. Define a Run

Runs are configured via Hydra. For example:

```yaml
# configs/experiment/train_imagenet.yaml
# @package _global_
experiment:
  name: train_imagenet
  tags: ["resnet", "imagenet", "resnet50"]
````

### 2. Train the Model

```bash
python train.py experiment=train_imagenet
```

Hydra composes the full configuration (model, dataset, optimizer, scheduler, logger, trainer) automatically.

### 3. Export Trained Weights

Convert a Lightning checkpoint into raw `state_dict` weights for deployment or fine-tuning:

```bash
python export.py \
  --ckpt-path path/to/checkpoint.ckpt \
  --output-path resnet50-imagenet.pt \
  experiment=train_imagenet
```

The script reconstructs the model using Hydra config and saves the weights via `torch.save`.

---

## 📥 Downloading ImageNet (WebDataset Format)

This project uses ImageNet from:

* [`timm/imagenet-1k-wds`](https://huggingface.co/datasets/timm/imagenet-1k-wds)

**⚠️ Note**: This dataset is **150+ GB**. Ensure you have sufficient disk space before downloading.

To download and organize the WebDataset shards:

```bash
python src/utils/setup_imagenet_wds.py
```

This creates the following directory structure:

```
datasets/wds_imagenet1k/
├── train/
│   ├── imagenet1k-train-0000.tar
│   └── ...
└── validation/
    ├── imagenet1k-validation-00.tar
    └── ...
```

Matches `configs/dataset/imagenet_wds.yaml`.

---

## Project Structure

```
configs/           # Hydra configs (datasets, models, training, logger, etc.)
src/models/        # ResNet and PlainNet implementations (blocks, stems)
src/data/          # LightningDataModules for CIFAR and ImageNet
src/utils/         # WebDataset utilities, export script, synthetic data
train.py           # Entry point for Hydra-based training
export.py          # Converts Lightning checkpoint to raw model weights
```

---

## 🛠️ Dependencies

Main dependencies:

* `torch`, `torchvision`
* `lightning>=2.0` (PyTorch Lightning)
* `hydra-core`, `omegaconf`
* `wandb`, `optuna`, `webdataset`
* `pre-commit`, `black`, `flake8`, `isort`, `nbQA`

Install all requirements:

```bash
pip install -r requirements.txt
```

---

## 📚 References

* [He et al., 2015 (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)
* [timm/imagenet-1k-wds](https://huggingface.co/datasets/timm/imagenet-1k-wds)
* [TorchVision ResNet](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)
