import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, use_residual: bool = True
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            bias=False,
            padding=1,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if use_residual and (stride != 1 or in_planes != planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out += self.shortcut(x)
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, use_residual: bool = True
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # Technically ResNet 1.5
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Identity()
        if use_residual and (stride != 1 or in_planes != planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_residual:
            out += self.shortcut(x)

        return self.relu(out)
