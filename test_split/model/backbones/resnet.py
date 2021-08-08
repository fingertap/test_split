import torch
import torch.nn as nn
from torchvision import models

from test_split.model import MODULES


@MODULES.register()
class ResNet(nn.Module):
    def __init__(self, depth: int, pretrained: bool = True):
        assert depth in [18, 34, 50, 101, 152]
        nn.Module.__init__(self)
        resnet = getattr(models, f'resnet{depth}')(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layers = nn.ModuleList([
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        ])

    def forward(self, data: torch.Tensor) -> tuple:
        data = self.stem(data)
        result = []
        for layer in self.layers:
            data = layer(data)
            result.append(data)
        return tuple(result)
