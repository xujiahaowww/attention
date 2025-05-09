import torch
import torch.nn as nn


class skNet(nn.Module):
    def __init__(self, in_channels):
        l = 4
        r = max(in_channels // l, 32)
        super(skNet, self).__init__()

        self.cov3x3 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.cov5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, groups=in_channels, dilation=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 第一层全连接
        self.all_liner_1 = nn.Linear(in_channels, r)

        # 对3*3和5*5分别进行还原通道
        self.all_liner_2_3 = nn.Linear(r, in_channels)
        self.all_liner_2_5 = nn.Linear(r, in_channels)

    def forward(self, x):
        value3x3 = self.cov3x3(x)
        value5x5 = self.cov5x5(x)
        value_all = value3x3 + value5x5

        avgpool = self.avgpool(value_all)

        Z = avgpool.view(avgpool.size(0), -1)
        Z = self.all_liner_1(Z)
        z_3 = self.all_liner_2_3(Z)
        z_5 = self.all_liner_2_5(Z)
        #       将两个全连接进行堆叠
        A = torch.stack((z_3, z_5), dim=0)

        A = torch.softmax(A, dim=0)

        A_3 = A[0].unsqueeze(-1).unsqueeze(-1)
        A_5 = A[1].unsqueeze(-1).unsqueeze(-1)

        value3x3_a = value3x3 * A_3
        value5x5_a = value5x5 * A_5

        value_all = value3x3_a + value5x5_a

        return value_all


if __name__ == '__main__':
    test = torch.randn(1, 32, 255, 255)
    net = skNet(32)
    out = net(test)
    print(out.shape)
