import torch.nn as nn
from tablerec.model import MODULES


@MODULES.register()
class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 padding_mode='zero',
                 activation='ReLU',
                 norm_type='SyncBatchNorm'):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode
        )
        if activation is not None:
            self.activation = getattr(nn, activation)()
        else:
            self.activation = None
        if norm_type is not None:
            self.bn = getattr(nn, norm_type)(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
        