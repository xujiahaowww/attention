import torch.nn as nn
import torch


class TD_attention(nn.Module):
    def __init__(self, in_channels):
        r = 4
        super(TD_attention, self).__init__()
        self.AvgP = nn.AdaptiveAvgPool2d(1)
        self.MaxP = nn.AdaptiveMaxPool2d(1)

        self.fn1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.AvgP(x)
        b = self.MaxP(x)

        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        print(a.shape)
        a = self.fn1(a)
        b = self.fn1(b)

        all = self.sigmoid(a + b)

        all = all.unsqueeze(-1).unsqueeze(-1)
        out = all * x
        return out


class KJ_attention(nn.Module):
    def __init__(self):
        super(KJ_attention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(pool)
        output = out * x
        return output


if __name__ == '__main__':
    b = torch.randn(2, 3, 45, 45)
    test1 = TD_attention(in_channels=3)
    test2 = KJ_attention()
    value = test1(b)
    value = test2(value)
    print(value)

