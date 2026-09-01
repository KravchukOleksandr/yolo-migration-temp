from __future__ import annotations

import torch
from torch import nn


class GRN(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        response = torch.norm(inputs, p=2, dim=(2, 3), keepdim=True)
        normalized = response / (response.mean(dim=1, keepdim=True) + self.eps)
        return inputs + self.gamma * inputs * normalized + self.beta


class ConvBNAct(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                inputs,
                outputs,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outputs),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class MicroNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.activation = nn.GELU()
        self.grn = GRN(hidden)
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.depthwise(inputs)
        output = self.normalization(output)
        output = self.expand(output)
        output = self.activation(output)
        output = self.grn(output)
        return inputs + self.project(output)


class HelmetMicroNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 16, stride=2)
        self.stage1 = MicroNeXtBlock(16)
        self.downsample = ConvBNAct(16, 24, stride=2)
        self.stage2 = nn.Sequential(*(MicroNeXtBlock(24) for _ in range(3)))
        self.projection = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        output = self.stage1(output)
        output = self.downsample(output)
        output = self.stage2(output)
        output = self.projection(output)
        return self.head(output).mean(dim=(2, 3)).squeeze(1)


def trainable_parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)
