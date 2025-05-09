import torch
import torch.nn as nn


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        print(max_pool.shape, avg_pool.shape)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(pool)
        output = out * x
        return output


if __name__ == '__main__':
    test = torch.randn(1, 3, 255, 255)
    net = SpatialAttentionModule()
    out = net(test)
    print(out.shape)
