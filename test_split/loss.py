import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self, n_buckets: int):
        nn.Module.__init__(self)
        self.nb = n_buckets
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        # pred: [B, 1000], float
        # label: [B], float \in [0, 1]
        pos = (label * self.nb).long()
        return self.ce(pred, pos)

