import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResnetBackboned(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        resnet = models.resnet50(pretrained=True)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layers = nn.ModuleList([
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ])

    def forward(self, image):
        x = self.stem(image)
        result = []
        for layer in self.layers:
            x = layer(x)
            result.append(x)
        return result


class FPN(nn.Module):
    def __init__(self, input_channels: list, out_channels: int):
        nn.Module.__init__(self)
        self.lconv = nn.ModuleList([
            nn.Conv2d(in_channel, out_channels, 1)
            for in_channel in input_channels
        ])
        self.fpnconv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in input_channels
        ])

    def forward(self, feature: list) -> list:
        laterals = [self.lconv[i](f) for i, f in enumerate(feature)]
        out = [self.fpnconv[i](f) for i, f in enumerate(laterals)]
        for i in range(len(feature) -1 , 0, -1):
            out[i-1] += F.interpolate(
                out[i],
                out[i-1].size()[-2:],
                mode='bilinear',
                align_corners=True
            )
        return out

class BucketModel(ResnetBackboned):
    def __init__(self):
        ResnetBackboned.__init__(self)
        self.fpn = FPN(
            input_channels=[256, 512, 1024, 2048],
            out_channels=256
        )
        self.compress = nn.Conv2d(256, 1, 1)

    def forward(self, image):
        feature = super().forward(image)
        feature = self.compress(self.fpn(feature)[0])
        row_feature = feature.mean(dim=-2)
        col_feature = feature.mean(dim=-1)
        row_res = F.interpolate(
            row_feature,
            size=image.size(-1),
            mode='linear',
            align_corners=True
        )
        col_res = F.interpolate(
            col_feature,
            size=image.size(-2),
            mode='linear',
            align_corners=True
        )
        return row_res.view(image.size(0), -1), col_res.view(image.size(0), -1)
