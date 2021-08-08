import torch
import torch.nn as nn
from tablerec.model import MODULES
from tablerec.model.bricks import ConvModule


@MODULES.register()
class FCNHead(nn.Module):
    def __init__(self,
                 in_channels=None,
                 channels=None,
                 num_classes=None,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 in_index=3):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        nn.Module.__init__(self)
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        self.concat_input = concat_input
        if num_convs == 0:
            assert in_channels == channels
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.cls_conv = nn.Conv2d(self.channels, num_classes, kernel_size=1)

        conv_padding = (kernel_size // 2) * dilation  # 1:1 convolution
        convs = []
        convs.append(ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=dilation
        ))
        for _ in range(num_convs - 1):
            convs.append(ConvModule(
                self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation
            ))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.concat_conv = ConvModule(
                self.in_channels + self.out_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )

    def forward(self, inputs):
        x = inputs[self.in_index]
        output = self.convs(x)
        if self.concat_input:
            output = self.concat_conv(
                torch.cat([x, output], dim=1)
            )
        output = self.cls_conv(output)
        return output
