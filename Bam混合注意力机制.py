import torch
from torch import nn


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.BatchNorm1d(in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.mlp(y).unsqueeze(-1).unsqueeze(-1)
        return y


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, in_planes, radio=8):
        super(SpatialAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_planes, in_planes // radio, 1, bias=False)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_planes // radio, in_planes // radio, 3, padding=4, stride=1, dilation=4, bias=False),
            nn.Conv2d(in_planes // radio, in_planes // radio, 3, padding=4, stride=1, dilation=4, bias=False),
            nn.BatchNorm2d(in_planes // radio)
        )
        self.conv = nn.Conv2d(in_planes // radio, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.conv3x3(y)
        y = self.conv(y)
        y = self.bn(y)
        return y


# BAM实现
class BAM(nn.Module):
    def __init__(self, in_planes, radio=16):
        super(BAM, self).__init__()
        self.ca = ChannelAttention(in_planes, radio)
        self.sa = SpatialAttention(in_planes, radio)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        res = x
        ca = self.ca(x)
        sa = self.sa(x)
        out = self.sig(ca + sa)
        output = out * x + res
        return output


if __name__ == '__main__':
    x = torch.randn(2, 30, 256, 256)
    model = BAM(30)
    y = model(x)
    print(y.shape)
