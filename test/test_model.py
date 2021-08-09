import torch
from tablerec.model import MODULES


backbone_params = {'depth': 50}
decode_head_params = {
    'in_channels': 2048,
    'in_index': 3,
    'channels': 512,
    'dilations': (1, 12, 24, 36),
    'dropout_ratio': 0.1,
    'num_classes': 4,
    'align_corners': False
}
aux_head_params = {
    'in_channels': 1024,
    'in_index': 2,
    'channels': 256,
    'num_convs': 1,
    'concat_input': False,
    'dropout_ratio': 0.1,
    'num_classes': 19,
    'align_corners': False,
}

model = MODULES.build('DeepLabV3', params={
    'backbone_params': backbone_params,
    'decode_head_params': decode_head_params,
    'auxilary_head_params': aux_head_params,
    'align_corners': False
})

data = torch.rand(1, 3, 1024, 1024)
res = model(data)