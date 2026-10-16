# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.CNNBaseModel import CNNBaseModel
from src.models.CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock
# Author : Hassan Kajila
'''
Improved YourSegNet network with:
- Residual blocks
- Dense blocks
- Bottleneck blocks
- Dilated (atrous) convolutions
- Asymmetric skip connections
'''


class YourSegNet(CNNBaseModel):
    """
    Asymmetric architecture inspired by DeepLabV3 with:
    - Deep encoder with atrous convolutions and residual blocks
    - Lightweight decoder with progressive upsampling and dense blocks
    - Improved asymmetric skip connections
    - Bottleneck blocks to reduce parameters
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds YourSegNet model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__(num_classes, init_weights)

        # Deep encoder with residual blocks
        in_channels = 1
        self.encoder1 = self._encoder_block(in_channels, 64)
        self.encoder2 = self._encoder_block(64, 128)
        self.encoder3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # Stride 2 for downsampling
            ResidualBlock(256, 256)
        )

        # Atrous-Bottleneck blocks to capture multi-scale context
        self.encoder4 = nn.Sequential(
            self._atrous_bottleneck_block(256, 512, dilation=2),
            ResidualBlock(512, 512)
        )
        self.encoder5 = nn.Sequential(
            self._atrous_bottleneck_block(512, 1024, dilation=4),
            ResidualBlock(1024, 1024)
        )

        # Lightweight decoder with dense blocks for better feature fusion
        self.decoder4 = self._decoder_block(1024, 512)
        self.decoder3 = self._decoder_block(512, 256)
        self.decoder2 = self._decoder_block(256, 128)
        self.decoder1 = self._decoder_block(128, 64)

        # Fusion layers for skip connections
        self.fusion3 = self._fusion_block(256 + 256, out_channels=256)  # Output: 256 channels for decoder2
        self.fusion2 = self._fusion_block(128 + 128, out_channels=128)  # Output: 128 channels for decoder1
        self.fusion1 = self._fusion_block(64 + 64, out_channels=64)  # Output: 64 channels for the final layer

        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoder with improved skip connections
        d4 = self.decoder4(e5)

        # Concatenation and fusion
        d3_concat = torch.cat([self.decoder3(d4), e3], dim=1)
        d3 = self.fusion3(d3_concat)

        d2_concat = torch.cat([self.decoder2(d3), e2], dim=1)
        d2 = self.fusion2(d2_concat)

        d1_concat = torch.cat([self.decoder1(d2), e1], dim=1)
        d1 = self.fusion1(d1_concat)

        # Final upsample to restore original size
        output = self.final(d1)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

    @staticmethod
    def _encoder_block(in_channels, out_channels):
        """
        Encoder block with standard convolutions and ResidualBlock
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels),  # Add a residual block
            nn.MaxPool2d(2)
        )

    @staticmethod
    def _atrous_bottleneck_block(in_channels, out_channels, dilation):
        """
        Block combining atrous convolutions and bottleneck for efficiency
        """
        mid_channels = out_channels // 4  # Bottleneck reduction factor
        return nn.Sequential(
            # Bottleneck down
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),

            # Atrous convolution
            nn.Conv2d(mid_channels, mid_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),

            # Bottleneck up
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    @staticmethod
    def _decoder_block(in_channels, out_channels):
        """
        Decoder block with upsampling and fewer convolutions
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    @staticmethod
    def _fusion_block(in_channels, out_channels):
        """
        Fusion block using DenseBlock to improve feature combination

        Args:
            in_channels: number of input channels (sum of concatenated channels)
            out_channels: number of output channels (must match the next block input)
        """
        mid_channels = in_channels // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),  # Initial channel reduction
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            DenseBlock(mid_channels),  # Adds mid_channels by concatenating -> mid_channels*2
            nn.Conv2d(mid_channels * 2, out_channels, 1),  # Adjust to desired output channels
            nn.ReLU()
        )

