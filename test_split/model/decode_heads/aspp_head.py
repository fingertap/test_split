import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from test_split.model import MODULES
from test_split.model.bricks import ConvModule


@MODULES.register()
class ASPPModule(nn.ModuleList):
    def __init__(self, dilations: Tuple[int], in_channels: int, channels):
        nn.ModuleList.__init__(self)
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        for dilation in self.dilations:
            self.append(ConvModule(
                self.in_channels, self.channels,
                1 if dilation == 1 else 3,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation
            ))

    def forward(self, x):
        return [module(x) for module in self]


class ASPPHead(nn.Module):
    def __init__(self,
                 dilations=(1, 6, 12, 18),
                 in_channels=None,
                 channels=None,
                 num_classes=None,
                 in_index=3,
                 align_corners=False,
                 dropout_ratio=0.1):
        nn.Module.__init__(self)
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.align_corners = align_corners
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(self.in_channels, self.channels, 1)
        )
        self.aspp_modules = ASPPModule(
            dilations, self.in_channels, self.channels
        )
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels, self.channels, 3, padding=1
        )
        self.cls_conv = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs[self.in_index]
        aspp_out = [F.F.interpolate(
            self.image_pool(x),
            size=x.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )]
        aspp_out.extend(self.aspp_modules(x))
        aspp_out = torch.cat(aspp_out, dim=1)
        output = self.bottleneck(aspp_out)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.cls_conv(output)
        return output
