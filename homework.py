import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision.models._utils import _ovewrite_named_param

from torchvision.models.resnet import BasicBlock, ResNet


# 定义个seNet
class Net(nn.Module):
    def __init__(self, input_channels, r=16):
        super(Net, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fn1 = nn.Sequential(
            nn.Linear(input_channels, input_channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // r, input_channels)
        )

    def forward(self, x):
        intifi = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fn1(x).view(x.size(0), -1, 1, 1)

        return intifi * x


# 定义个类继承BasicBlock并执行seNet
class Re_SeNet(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, ):
        # 调用父类的初始化函数
        super(Re_SeNet, self).__init__(
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
        self.se = Net(inplanes)

    def forward(self, x):
        identity = x
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


class ModifyNet(ResNet):
    def __init__(self,
                 block,
                 layers,
                 num_classes,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation=None,
                 norm_layer=None, ):
        super(ModifyNet, self).__init__(
            block,
            layers,
            num_classes = 1000,
            zero_init_residual=False,
            groups=1,
            width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,

       )
        self.modefiy(self.layer4)

    def modefiy(self, layer4):
        seNet = Net(input_channels=512)
        return nn.Sequential(
            layer4,
            seNet
        )


def _resnet(
        block,
        layers,
        weights,
        progress: bool,
        **kwargs,
):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ModifyNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnet18senet(*, weights=None, progress=True, **kwargs):
    weights = ResNet18_Weights.verify(weights)
    return _resnet(Re_SeNet, [2, 2, 2, 2], weights, progress, **kwargs)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = resnet18senet()
    y = model(x)
    print(y.shape)
