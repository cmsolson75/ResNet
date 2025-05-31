import torch
from torch import nn
from resnet.models.blocks import BasicBlock
from resnet.models.resnet import ResNet
from resnet.models.stems import CIFARStem

NOTES = """
Scratch File
For testing the basic implementation
Will add
- Hydra
- Lightning Training Wrapper
- W&B for logging
"""


from torchvision.transforms import transforms
import torchvision


def create_cifar10_dataloader_from_config(
    train: bool,
    download: bool = True,
    batch_size: int = 128,
) -> torch.utils.data.DataLoader:
    batch_size = batch_size
    root_dir = "./cifar10"

    num_workers = 7

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        shuffle = True
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        shuffle = False

    dataset = torchvision.datasets.CIFAR10(
        root=root_dir, train=train, download=download, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


train_loader = create_cifar10_dataloader_from_config(True, True, 256)
loss_fn = nn.CrossEntropyLoss()

# dummy_inp = torch.randn(256, 3, 32, 32)
device = "cuda"
model = ResNet(
    stem=CIFARStem(out_channels=16),
    block=BasicBlock,
    block_layers=[3, 3, 4, 4],
    stage_channels=[16, 32, 64, 128],
    num_classes=10,
    use_residual=True,
)
model = model.to(device)


optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=len(train_loader), T_mult=1, eta_min=1e-5
)
for epoch in range(50):
    loss_current = 0
    i = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_current += loss.item()
        i += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {loss_current / i:.4f}")


from torch.utils.data import DataLoader


def evaluate(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    print("Testing...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    model.train()
    print(f"Test Accuracy: {100 * correct // total}%")


evaluate(model, create_cifar10_dataloader_from_config(False, True, 256), device)
