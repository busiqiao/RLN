import torch
from torch import nn
import ffe
import tfe


class RLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tfe = tfe.TFE(input_size=1, hidden_size=16, num_heads=1, dropout=0.5)
        self.ffe = ffe.FFE()
        self.lstm = tfe.LSTM(input_size=16, hidden_size=64, dropout=0.5)
        self.out = nn.Sequential(
            nn.Linear(in_features=64, out_features=600),
            nn.LeakyReLU(),
            nn.Linear(in_features=600, out_features=6),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.tfe(x)
        x2 = self.ffe(x)
        x = torch.cat((x1, x2), dim=2)
        x = self.lstm(x)
        x = self.out(x)
        return x
