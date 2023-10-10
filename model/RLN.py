import math

import torch
import numpy as np
from torch import nn
from model import ffe
from model import tfe


class RLN(nn.Module):
    def __init__(self, num_class=6, channel=124, batch_size=64):
        super().__init__()
        self.channel = channel
        self.batch_size = batch_size
        self.tfe = tfe.TFE(input_size=32, hidden_size=16, num_heads=1, dropout=0.5)
        self.ffe = ffe.FFE()
        self.lstm = tfe.LSTM(input_size=16, hidden_size=64, dropout=0.5)
        self.out = nn.Sequential(
            nn.Linear(in_features=64, out_features=600),
            nn.LeakyReLU(),
            nn.Linear(in_features=600, out_features=num_class),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = np.zeros((self.batch_size, 1, 16), dtype=np.float32)
        x1 = torch.from_numpy(x1).cuda()
        x2 = x1

        one_dim_tensors = torch.chunk(x, self.channel, dim=1)

        for i in range(self.channel):
            tmp = self.tfe(one_dim_tensors[i])
            x1 = torch.cat((x1, tmp), dim=1)
        x1 = torch.cat((x1[:, :0, :], x1[:, 1:, :]), dim=1)

        for i in range(math.ceil(self.channel / 16)):
            tmp = self.ffe(one_dim_tensors[i])
            x2 = torch.cat((x2, tmp), dim=1)
        x2 = torch.cat((x2[:, :0, :], x2[:, 1:self.channel+1, :]), dim=1)

        x = torch.cat((x1, x2), dim=1)

        # x = torch.transpose(tmp_x, dim0=1, dim1=2)
        x, (h_0, c_0) = self.lstm(x)
        x = self.out(x[:, -1, :])
        return x
