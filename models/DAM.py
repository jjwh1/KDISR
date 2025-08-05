import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PixelAttention(nn.Module):
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pixel_attention(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // 16, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        ca_weights = self.global_avg_pool(x).view(b, c)
        ca_weights = self.relu(self.fc1(ca_weights))
        ca_weights = self.sigmoid(self.fc2(ca_weights)).view(b, c, 1, 1)
        return x * ca_weights

class DAM(nn.Module):
    def __init__(self, input_channels, teacher_channels):
        super(DAM, self).__init__()
        self.pixel_attention = PixelAttention(input_channels)
        self.feature_projection = nn.Conv2d(input_channels, teacher_channels, kernel_size=1)
        self.conv = nn.Conv2d(input_channels + teacher_channels, teacher_channels, kernel_size=1)
        self.channel_attention = ChannelAttention(teacher_channels)

    def forward(self, x):
        pixel_attended = self.pixel_attention(x)
        projected = self.feature_projection(x)
        fused = torch.cat([pixel_attended, projected], dim=1)
        fused = self.conv(fused)
        output = self.channel_attention(fused)

        return output

