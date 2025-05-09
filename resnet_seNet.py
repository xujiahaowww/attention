import torch
from torchvision.models.resnet import _resnet, BasicBlock,ResNet18_Weights
from seNet import SENet


class SE_BasicBlock(BasicBlock):
    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        # 调用父类的初始化函数
        super(SE_BasicBlock, self).__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        # 加入注意力
        self.se = SENet(inplanes)

    def forward(self, x):
        identity = x

        # 在这里加入SE
        x = self.se(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet18senet(*, weights=None, progress=True, **kwargs):
    weights = ResNet18_Weights.verify(weights)
    return _resnet(SE_BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

# 测试数据
if __name__ == "__main__":
    import torch

    x = torch.randn(1, 3, 224, 224)
    model = resnet18senet()
    y = model(x)
    print(y.shape)
