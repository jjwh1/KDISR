import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiKernelConvBlock, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=self.get_same_padding(k)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for k in kernel_sizes
        ])

    def get_same_padding(self, kernel_size):
        return (kernel_size - 1) // 2

    def forward(self, x):
        return [branch(x) for branch in self.branches]


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
        b, c, _, _ = x.size()
        ca_weights = self.global_avg_pool(x).view(b, c)
        ca_weights = self.relu(self.fc1(ca_weights))
        ca_weights = self.sigmoid(self.fc2(ca_weights)).view(b, c, 1, 1)
        return x * ca_weights


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.kernel_sizes = [3, 5, 7]
        num_branches = len(self.kernel_sizes)

        self.multi_kernel_conv = MultiKernelConvBlock(in_channels, in_channels, self.kernel_sizes)
        self.pixel_attention = nn.ModuleList([PixelAttention(in_channels) for _ in range(num_branches)])
        self.channel_attention = ChannelAttention(in_channels * num_branches)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = self.multi_kernel_conv(x)
        features = [self.pixel_attention[i](features[i]) for i in range(len(features))]
        x = torch.cat(features, dim=1)
        x = self.channel_attention(x)
        return self.final_conv(x)



class Generator_T(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(Generator_T, self).__init__()

        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = AttentionBlock(64, 128)
        self.enc3 = AttentionBlock(128, 256)
        self.enc4 = AttentionBlock(256, 512)
        self.enc5 = AttentionBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.final_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))

        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final_layer(dec1)), enc3, enc5, dec3, dec1