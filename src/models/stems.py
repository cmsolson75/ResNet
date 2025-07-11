import torch.nn as nn


class CIFARStem(nn.Module):
    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.out_channels = out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.stem(x)


class ImageNetStem(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.out_channels = out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.stem(x)
