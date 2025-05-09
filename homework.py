from typing import Union, Type
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision.models._api import register_model
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck


# 定义个seNet
class SeNet(nn.Module):
    def __init__(self, input_channels, r=16):
        super(SeNet, self).__init__()
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


class ModifyNet(ResNet):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ModifyNet, self).__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer
        )
        self.seNet = SeNet(input_channels=512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.seNet(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers,
        weights,
        progress,
        **kwargs,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ModifyNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18_renet(*, weights=None, progress: bool = True, **kwargs) -> ResNet:
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = resnet18_renet()
    y = model(x)
    print(y.shape)
