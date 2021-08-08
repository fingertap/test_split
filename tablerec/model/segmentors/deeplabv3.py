import torch
import torch.nn as nn
import torch.nn.functional as F
from tablerec.model import MODULES


@MODULES.register()
class DeepLabV3(nn.Module):
    def __init__(self,
                 backbone: str, backbone_params: dict,
                 decode_head: str, decode_head_params: dict,
                 auxilary_head: str, auxilary_head_params: dict,
                 align_corners: bool = False):
        nn.Module.__init__(self)
        self.align_corners = align_corners
        self.backbone = MODULES.build(backbone, backbone_params)
        self.decode_head = MODULES.build(decode_head, decode_head_params)
        self.auxilary_head = MODULES.build(auxilary_head, auxilary_head_params)

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
