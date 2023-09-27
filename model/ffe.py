from torch import nn


class FFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1)
        self.cov2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=4, stride=2)
        self.cov3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, stride=1)
        self.cov4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, stride=1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)
        x = self.cov4(x)
        x = self.activate(x)
        return x
