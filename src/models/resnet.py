import torch
import torch.nn as nn
from typing import List, Union, Type
from src.models.blocks import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(
        self,
        stem: nn.Module,
        block: Type[Union[BasicBlock, Bottleneck]],
        block_layers: List[int],
        stage_channels: List[int],
        num_classes: int,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.stem = stem
        self.in_planes = stem.out_channels

        assert len(block_layers) == len(
            stage_channels
        ), "block_layers and stage_channels must align"

        self.stages = nn.ModuleList()
        for idx, (num_blocks, out_planes) in enumerate(
            zip(block_layers, stage_channels)
        ):
            stride = 1 if idx == 0 else 2
            stage = self._make_layer(
                block, out_planes, num_blocks, stride=stride, use_residual=use_residual
            )
            self.stages.append(stage)

        final_planes = stage_channels[-1] * getattr(block, "expansion", 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_planes, num_classes)

    def _make_layer(
        self,
        block: Union[BasicBlock, Bottleneck],
        out_planes: int,
        blocks: int,
        stride: int,
        use_residual: bool,
    ) -> nn.Module:
        layers = []
        layers.append(block(self.in_planes, out_planes, stride, use_residual))
        self.in_planes = out_planes * getattr(block, "expansion", 1)
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, out_planes, 1, use_residual))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
