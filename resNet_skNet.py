import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet50_Weights, _resnet

from skNet import skNet


class basicBlock(Bottleneck):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super().__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        width = int(planes * (base_width / 64.0)) * groups
        if stride == 1:
            self.conv2 = skNet(width)
        else:
            self.conv2 = nn.Sequential(
                skNet(width),
                nn.Conv2d(width, width, kernel_size=1, stride=2, bias=False),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet50(*, weights=None, progress: bool = True, **kwargs):
    weights = ResNet50_Weights.verify(weights)
    return _resnet(basicBlock, [3, 4, 6, 3], weights, progress, **kwargs)

if __name__ == '__main__':
    model = resnet50()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
