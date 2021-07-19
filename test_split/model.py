import torch
import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, n_buckets: int):
        nn.Module.__init__(self)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.row_pred = nn.Linear(
            resnet.fc.in_features + 800, n_buckets * 2
        )
        self.col_pred = nn.Linear(
            resnet.fc.in_features + 800, n_buckets * 2
        )

    def forward(self, image):
        B = image.size(0)
        feature = self.backbone(image).view(B, -1)
        row_pixels = image.mean(dim=[1, 2])
        col_pixels = image.mean(dim=[1, 3])
        row_pred = self.row_pred(torch.cat([feature, row_pixels], dim=-1))
        col_pred = self.col_pred(torch.cat([feature, col_pixels], dim=-1))
        return torch.stack([row_pred, col_pred], dim=1)
