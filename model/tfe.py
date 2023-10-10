from einops import rearrange
from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bias=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=dropout, bias=bias)

    def forward(self, x):
        x = self.lstm(x)
        return x


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MHA, self).__init__()
        self.self_attn = nn.MultiheadAttention(batch_first=True, embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # [b, c, t]
        # Multi-head self-attention
        attention_output, _ = self.self_attn(x[0], x[0], x[0])
        return attention_output


class TFE(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout=0.5):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.mha = MHA(embed_dim=16, num_heads=num_heads)

    def forward(self, x):
        x = self.lstm(x)
        x = self.mha(x)
        return x
