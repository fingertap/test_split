import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2., alpha: float = 1.):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, pred, label):
        pred = torch.sigmoid(pred)
        target = F.one_hot(label, num_classes=800).float().cumsum(dim=-1)
        # eps = 1e-8
        # loss = - (
        #     target * torch.log(pred + eps) * (1. - pred ** self.gamma)
        #     + (1. - target) * torch.log(1. - pred + eps) * pred ** self.gamma
        # )
        # loss[target < 0.5] *= self.alpha
        # return loss.mean()
        return self.bce(pred, target)



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
