import torch
import torch.nn as nn


class BucketLoss(nn.Module):
    def __init__(self, n_buckets: int):
        nn.Module.__init__(self)
        self.nb = n_buckets
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        # pred: [B, 1000], float
        # label: [B], float \in [0, 1]
        pos = (label * self.nb).long()
        return self.ce(pred, pos)


class RegressionLoss(nn.Module):
    def forward(self, pred, label):
        return torch.mean((pred.view(label.size()) - label) ** 2)
