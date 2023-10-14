import torch
import numpy as np
from torch import nn
from model import ffe
from model import tfe
from model import weight


class RLN(nn.Module):
    def __init__(self, num_class=6, channel=124, batch_size=64):
        super().__init__()
        self.channel = channel
        self.batch_size = batch_size
        self.weight = weight.Weight(channel=channel)
        self.tfe = tfe.TFE(input_size=1, hidden_size=16, num_heads=8, dropout=0.5)
        self.ffe = ffe.FFE(num_class=num_class)
        self.lstm = tfe.LSTM(input_size=16, hidden_size=64, dropout=0.5)
        self.out = nn.Sequential(
            nn.Linear(in_features=248, out_features=600),
            nn.LeakyReLU() if num_class == 6 else nn.ELU(),
            nn.Linear(in_features=600, out_features=num_class),
            nn.LeakyReLU() if num_class == 6 else nn.ELU()
        )

    def forward(self, x):  # [b, c, t]
        x1 = np.zeros((self.batch_size, 16, 1), dtype=np.float32)
        x1 = torch.from_numpy(x1).cuda()
        x2 = x1

        x = self.weight(x)

        one_dim_tensors = torch.chunk(x, self.channel, dim=1)  # [b, 1, t]

        for i in range(self.channel):
            tmp = torch.transpose(one_dim_tensors[i], dim0=1, dim1=2)
            tmp = self.tfe(tmp)
            tmp = torch.transpose(tmp, dim0=1, dim1=2)
            x1 = torch.cat((x1, tmp), dim=2)

            tmp = self.ffe(one_dim_tensors[i])
            x2 = torch.cat((x2, tmp), dim=2)

        x1 = torch.cat((x1[:, :, :0], x1[:, :, 1:]), dim=2)
        x2 = torch.cat((x2[:, :, :0], x2[:, :, 1:]), dim=2)

        x = torch.cat((x1, x2), dim=2)

        x = torch.transpose(x, dim0=1, dim1=2)
        x, (h_0, c_0) = self.lstm(x)
        x = torch.transpose(x, dim0=1, dim1=2)
        x = self.out(x[:, -1, :])
        return x
