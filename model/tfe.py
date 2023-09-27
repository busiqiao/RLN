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
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.h = num_heads
        self.d = channels // num_heads
        # scale factor
        self.scale = self.d ** -0.5

        self.conv_qkv = nn.Conv2d(in_channels=channels, out_channels=3 * channels, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [b, c, p, t]
        qkv = self.conv_qkv(x)  # [b, c, p, t] -> [b, 3*c, p, t]
        q, k, v = rearrange(qkv, 'b (qkv h d) p t -> qkv b h d p t', qkv=3, h=self.h, d=self.d)
        q = rearrange(q, 'b h d p t -> b h p (d t)')
        k = rearrange(k, 'b h d p t -> b h (d t) p')
        v = rearrange(v, 'b h d p t -> b h p (d t)')

        dots = torch.matmul(q, k) * self.scale  # [b, h, p, p]
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)  # [b, h, p, (dt)]
        out = rearrange(out, 'b h p (d t) -> b (h d) p t', h=self.h, d=self.d)
        return out


class TFE(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout=0.5):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.mha = MHA(channels=1, num_heads=num_heads)

    def forward(self, x):
        x = self.lstm(x)
        x = self.mha(x)
        return x
