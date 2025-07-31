
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

class ChannelSEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)

    def forward(self, x):
        b, c, t = x.shape
        y = x.mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(2)
        return x * y

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.bn(out)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=4, ff_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.ff2 = FeedForward(dim, ff_expansion)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.ff1(self.norm(x)))
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x = x + self.dropout(x_conv.transpose(1, 2))
        x = x + self.dropout(self.ff2(self.norm(x)))
        return x

class EEGClassifier(nn.Module):
    def __init__(self, in_channels=128, seq_len=250, num_classes=5, embed_dim=128):
        super().__init__()
        self.conv = MultiScaleConvBlock(in_channels, embed_dim // 3)
        self.se = ChannelSEAttention(embed_dim)
        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.encoder = nn.Sequential(
            ConformerBlock(embed_dim),
            ConformerBlock(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.proj(x).transpose(1, 2)
        b, t, d = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed[:, :t+1]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
