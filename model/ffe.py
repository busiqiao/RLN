import torch
from torch import nn


class FFE(nn.Module):
    def __init__(self, num_class=6):
        super().__init__()
        self.cov1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=1)
        self.cov2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=4, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cov3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, stride=1)
        self.cov4 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=1)
        self.activate = nn.LeakyReLU() if num_class == 6 else nn.ELU()

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.pool(x)
        x = self.cov3(x)
        x = self.cov4(x)
        x = self.activate(x)

        return x


if __name__ == '__main__':
    modal = FFE()
    data = torch.randn((3, 1, 32))  # [b, c, t]
    modal(data)
