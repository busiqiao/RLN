from einops import rearrange
from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bias=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, num_layers=2, hidden_size=hidden_size, bias=bias,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, (h_n, _) = self.lstm(x)
        # x1 = x[:, -1, :]
        h_n = self.dropout(h_n[-1, :, :]).unsqueeze(0)
        x = self.dropout(x)
        return x, h_n


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MHA, self).__init__()
        self.self_attn = nn.MultiheadAttention(batch_first=True, embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # [b, c, t]
        # Multi-head self-attention
        attention_output, _ = self.self_attn(x, x, x)
        return attention_output


class TFE(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout=0.5):
        super().__init__()
        self.lstm_with_dropout = LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.mha = MHA(embed_dim=16, num_heads=num_heads)

    def forward(self, x):
        _, h_n = self.lstm_with_dropout(x)
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n[:, -1, :].unsqueeze(1)
        # h_n = h_n.permute(0, 2, 1)
        x = self.mha(h_n)
        return x


if __name__ == '__main__':
    model = TFE(input_size=1, hidden_size=16, num_heads=1, dropout=0.5)
    data = torch.randn((3, 32, 1))  # B,L,C
    model(data)
