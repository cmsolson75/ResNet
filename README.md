# ResNet: Paper-Accurate Implementation with Modern Tools

This project implements ResNet (and PlainNet) as described in [He et al. (2015)](https://arxiv.org/abs/1512.03385). It uses **PyTorch Lightning**, **Hydra**, and **W\&B** to train models on **ImageNet (via WebDataset)** and replicate original CIFAR-10/CIFAR-100 experiments, including **intermediate activation analysis**.

---

## 🎯 Objectives

* Implement ResNet and PlainNet from scratch
* Train ResNet-50 on ImageNet using WebDataset format
* Reproduce CIFAR-10 ResNet vs. PlainNet results
* Perform internal analysis of layer-wise activations
* Conduct hyperparameter sweeps on CIFAR-10 and CIFAR-100

---

## ⚙️ Framework Stack

| Tool              | Role                                   |
| ----------------- | -------------------------------------- |
| PyTorch Lightning | Structured training and checkpointing  |
| Hydra             | Experiment configuration management    |
| W\&B              | Logging and hyperparameter sweeps      |
| WebDataset        | Efficient large-scale ImageNet loading |
| Optuna            | Hyperparameter Sweeps                  |

---

## 🧪 Experiment Plan

### 1. **ImageNet Training**

* **Model**: ResNet-50
* **Data**: `timm/imagenet-1k-wds` (WebDataset)
  - https://huggingface.co/datasets/timm/imagenet-1k-wds
* **Training Setup**:

  * SGD + momentum
  * Cosine or multi-step LR
  * Mixed-precision
* **Output**: `resnet50-imagenet.pt`

---

### 2. **CIFAR-10/PlainNet Replication**

* **Goal**: Reproduce ResNet vs. PlainNet results
* **Models**:

  * `Plain-{20,32,44,56,110}`
  * `ResNet-{20,32,44,56,110}`
* **Training Setup**:

  * Dataset: CIFAR-10
  * Epochs: 164
  * LR: 0.1, decayed at 82 and 123
  * Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
* **Metrics**:

  * Train/val accuracy
  * Degradation trend with increased depth

---

### 3. **Intermediate Activation Analysis**

* **Objective**: Replicate the activation magnitude study from the paper (Fig. 6)
* **Method**:
  * Hook into each residual block
  * Compute L2 norm of output activations
  * Aggregate across layers and depth
* **Analysis**:
  * Compare ResNet vs. PlainNet
  * Validate gradient preservation across layers
  * Evaluate depth-wise signal degradation

---

### 4. **CIFAR-10/100 Hyperparameter Sweeps**

* **Purpose**: Evaluate impact model components.
* **Method**: Hydra multirun + Optuna
* **Targets**:
  * Get optimal ResNet for CIFAR-10 and CIFAR-100

---

## 📚 References

* [He et al., 2015 (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)
* [TorchVision ResNet](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)
* [TinyGrad ResNet](https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py)

---
