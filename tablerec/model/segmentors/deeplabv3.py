import torch
import torch.nn as nn
import torch.nn.functional as F
from tablerec.model.registry import MODULES


@MODULES.register()
class DeepLabV3(nn.Module):
    def __init__(self,
                 backbone_params: dict,
                 decode_head_params: dict,
                 auxilary_head_params: dict,
                 align_corners: bool = False):
        nn.Module.__init__(self)
        self.align_corners = align_corners
        self.backbone = MODULES.build('ResNet', backbone_params)
        self.decode_head = MODULES.build('ASPPHead', decode_head_params)
        self.auxilary_head = MODULES.build('FCNHead', auxilary_head_params)

    def forward(self, images):
        feature = self.backbone(images)
        dh_out = self.decode_head(feature)
        if self.training:
            ah_out = self.decode_head(feature)
            return dh_out, ah_out
        output = F.interpolate(
            torch.sigmoid(dh_out),
            images.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return output
