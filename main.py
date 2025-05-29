import torch
from torch import nn

NOTES = """
Scratch File
For testing the basic implementation
Will add
- Hydra
- Lightning Training Wrapper
- W&B for logging
"""

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return self.act(x)


# Building -> Will make more extendable
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, stride, padding=1)
    def forward(self, x):
        return x
    
class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_channels)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

dummy_inp = torch.randn(256, 3, 224, 224)
model = ResNet18(3, 1000)
model = model.to("cuda")
out = model(dummy_inp.to("cuda"))

print(out.shape)
print(NOTES)