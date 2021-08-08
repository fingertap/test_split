import torch.nn as nn
from test_split.model import MODULES
from test_split.model.decode_heads import ASPPHead, FCNHead


@MODULES.register()
class DeepLabV3(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = ASPPHead(
            in_channels=2048,
            channels=512,
            dilations=(1, 12, 24, 36),
        )
