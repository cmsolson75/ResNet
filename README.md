# ResNet Implementation


Implementation References
- [TinyGrad](https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py)
- [Torch](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html)
- [Paper](https://arxiv.org/abs/1512.03385)

Next Steps
- Evaluate ResNet on Cifar10 and Cifar100
- Implement extendable codebase.



## 🧪 Experiment Plan

This project explores residual networks (ResNets) under various training regimes, including full ImageNet training, fine-tuning, and CIFAR-10 replication studies. The goal is to analyze model behavior, generalization, and training dynamics across transfer learning and architecture variants.

---

### **Phase 1: ImageNet Pretraining**

**Objective**: Train a base model on ILSVRC-2012 for use in downstream transfer learning.

* **Models**:
  * `ResNet-50`
* **Setup**:
  * Standard ResNet training on ImageNet (90 epochs)
  * SGD + momentum, weight decay, multi-step LR schedule
  * Output: `resnet50_imagenet.pt`

---

### **Phase 2: CIFAR-10 Fine-Tuning**

**Objective**: Evaluate transfer learning effectiveness from ImageNet → CIFAR-10 under multiple fine-tuning strategies.

* **Base Model**: `ResNet-50` pretrained on ImageNet

#### 🔁 Fine-Tuning Variants

| Strategy     | Description                                   |
| ------------ | --------------------------------------------- |
| Head only    | Freeze backbone, train new classifier head    |
| Head + Stem  | Freeze middle layers, fine-tune low/high ends |
| Full FT      | Fine-tune entire model                        |
| FastAI-style | Discriminative LR, gradual unfreezing         |

#### 🧪 Techniques Explored

* Warm-up schedules
* Learning rate finders
* Layer freezing/unfreezing control
* BatchNorm handling during FT

---

### **Phase 3: CIFAR-10 From Scratch**

**Objective**: Replicate and validate results from He et al. (2015) for ResNet vs. PlainNet.

* **Datasets**: CIFAR-10 (32×32)
* **Models**:

  * `ResNet-{20, 32, 44, 56, 110}`
  * `Plain-{20, 32, 44, 56, 110}`
* **Training Setup**:

  * SGD + momentum, weight decay
  * LR: 0.1, decay at epochs 82 and 123
  * Epochs: 164
  * Data Augmentation: random crop, horizontal flip
* **Metrics**:

  * Training vs. validation error
  * Degradation trends
  * Layer-wise response analysis (L2 norm of activations)

---

### **Phase 4: Internal Analysis**

**Objective**: Analyze internal activations and learning behavior.

* Hook-based capture of intermediate activations
* Compute per-layer L2 norms (as in ResNet paper)
* Compare depth-wise behavior in ResNet vs. PlainNet

