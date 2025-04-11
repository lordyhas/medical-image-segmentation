# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import BottleneckBlock, DenseBlock, ResidualBlock
import torchvision
import torch.nn.functional as F

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourUNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''


class YourUNet(CNNBaseModel):
    """
     Class that implements a brand new UNet segmentation network
    """

    def __init__(self, num_classes=4, init_weights=True):
        super().__init__(num_classes, init_weights)

        input_channels = 1  # images en niveaux de gris

        # ENCODER
        self.encoder_block1 = nn.Sequential(
            self._conv_block(input_channels, 64),
            self._contraction_block(64, 128),
            ResidualBlock(128, 128),
            DenseBlock(128)
        )

        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            BottleneckBlock(256)
        )

        self.encoder_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512, stride=1),
            nn.BatchNorm2d(512),
            DenseBlock(512)
        )

        # MIDDLE
        self.middle_upconv = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        # DECODER
        self.decoder_block1 = nn.Sequential(
            self._expansion_block(512, 256),
            ResidualBlock(256, 256),
            nn.BatchNorm2d(256)
        )

        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            ResidualBlock(256, 128),
            DenseBlock(128),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.ReLU()
        )

        self.final_output_block = self._final_block(128, 64, num_classes)

    def forward(self, x):
        """
        Forward pass of the YourUNet model
        """
        # Encode
        enc1 = self.encoder_block1(x)
        enc2 = self.encoder_block2(enc1)
        enc3 = self.encoder_block3(enc2)

        # Middle
        mid = self.middle_upconv(enc3)

        # Decode
        dec_input1 = torch.cat((mid, enc2), dim=1)
        dec1 = self.decoder_block1(dec_input1)

        dec_input2 = torch.cat((dec1, enc1), dim=1)
        dec2 = self.decoder_block2(dec_input2)

        output = self.final_output_block(dec2)
        return output

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=3):
        """
        Bloc de convolution standard avec MaxPool
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    @staticmethod
    def _contraction_block(in_channels, out_channels, kernel_size=3):
        """
        Bloc de contraction avec double MaxPool
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    @staticmethod
    def _expansion_block(in_channels, out_channels, kernel_size=3):
        """
        Bloc d'expansion avec upsampling par convolution transposée
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    @staticmethod
    def _final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Bloc final de sortie du modèle YourUNet
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )



'''
Fin de votre code.
'''