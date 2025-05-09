from typing import Tuple, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image


def data_get() -> tuple[DataLoader[Any], Any]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.RandomRotation(10),  # 随机旋转 ±10 度
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
    ])
    path = 'E:\华清ai课程\code\卷积神经网络\day03\\flowers\\flowers'
    print(path)
    data_train = datasets.ImageFolder(path, transform=transform)
    data_train_loader = DataLoader(data_train, batch_size=50, shuffle=True)
    print(data_train.classes)
    print(data_train.class_to_idx)
    return data_train_loader, data_train.classes


# 定义个自己的类
class Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super(Net, self).__init__()
        self.features_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, groups=in_channels),
            nn.AdaptiveAvgPool2d(100),
            nn.ReLU()
        )
        self.features_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, groups=32),
            nn.AdaptiveAvgPool2d(50),
            nn.ReLU(),
        )
        self.Fn1 = nn.Sequential(
            nn.BatchNorm1d(64 * 50 * 50),
            nn.Linear(64 * 50 * 50, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.Fn2 = nn.Sequential(
            nn.Linear(128, out_channels),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.features_1(x)
        x = self.features_2(x)
        x = x.view(x.size(0), -1)
        x = self.Fn1(x)
        x = self.Fn2(x)
        return x


def train(model, data_train_loader) -> Any:
    pass


if __name__ == '__main__':
    data_train_loader, classes = data_get()
    model = Net(3)
    train()
