import torch.nn as nn


class BaseDecodeHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 dropout: float = None,
                 align_corners: bool = False,
                 **kwargs
                 ):
        self.in_channels = in_channels
        self.channels = channels
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout2d(dropout, inplace=True)
        else:
            self.dropout = None
        self.conv_cls = nn.Conv2d(channels, num_classes, 1)
        self.align_corners = align_corners

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_cls(feat)
        return output
