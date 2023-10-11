import torch
from torch import nn


class Weight(nn.Module):
    def __init__(self, channel):
        super(Weight, self).__init__()
        # 定义一个可训练的权重参数
        self.weight = nn.Parameter(torch.randn(channel, 1))  # 初始化权重参数

    def forward(self, x):
        # 将输入tensor与可训练权重相乘
        result = x * self.weight
        return result
