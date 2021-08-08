import torch
from test_split.model.backbone.resnet import ResNet


model = ResNet(50, pretrained=False)
data = torch.rand(1, 3, 224, 224)
res = model(data)
assert len(res) == 4
assert res[0].size() == (1, 256, 56, 56)
assert res[1].size() == (1, 512, 28, 28)
assert res[2].size() == (1, 1024, 14, 14)
assert res[3].size() == (1, 2048, 7, 7)

print('测试通过')