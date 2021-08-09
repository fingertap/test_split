import torch.nn as nn
from tablerec.model.registry import MODULES


@MODULES.register()
class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 padding_mode='zeros',
                 activation='ReLU',
                 norm_type='BatchNorm2d'):
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
        if self.activation is not None:
            x = self.activation(x)
        if self.bn is not None:
            training_status = self.bn.training
            if x.size(0) * x.size(2) * x.size(3) == 1:
                self.bn.training = False
            x = self.bn(x)
            if x.size(0) * x.size(2) * x.size(3) == 1:
                self.bn.training = training_status
        return x
        