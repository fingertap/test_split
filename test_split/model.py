import torch
import torch.nn as nn
import torchvision.models as models


class ResnetBackboned(nn.Module):
    def __init__(self):
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
        self.d_model = resnet.fc.in_features

    def forward(self, image):
        return self.backbone(image).view(image.size(0), -1)


class RegressionModel(ResnetBackboned):
    def __init__(self):
        ResnetBackboned.__init__(self)
        self.row_pred = nn.Linear(self.d_model * 2, 1, bias=True)
        self.col_pred = nn.Linear(self.d_model * 2, 1, bias=True)
        self.feature_filter = nn.Linear(self.d_model, self.d_model, bias=True)
        self.row_filter = nn.Linear(800, self.d_model, bias=True)
        self.col_filter = nn.Linear(800, self.d_model, bias=True)

    def forward(self, image):
        feature = self.feature_filter(super().forward(image))
        row_pixels = self.row_filter(image.mean(dim=[1, 3]))
        col_pixels = self.col_filter(image.mean(dim=[1, 2]))
        row_pred = self.row_pred(torch.cat([feature, row_pixels], dim=-1))
        col_pred = self.col_pred(torch.cat([feature, col_pixels], dim=-1))

        return torch.sigmoid(row_pred), torch.sigmoid(col_pred)


class BucketModel(ResnetBackboned):
    def __init__(self, n_buckets: int):
        ResnetBackboned.__init__(self)
        self.row_pred = nn.Linear(self.d_model + 800, n_buckets, bias=True)
        self.col_pred = nn.Linear(self.d_model + 800, n_buckets, bias=True)

    def forward(self, image):
        feature = super().forward(image)
        row_pixels = image.mean(dim=[1, 3])
        col_pixels = image.mean(dim=[1, 2])
        row_pred = self.row_pred(torch.cat([feature, row_pixels], dim=-1))
        col_pred = self.col_pred(torch.cat([feature, col_pixels], dim=-1))
        return row_pred, col_pred
