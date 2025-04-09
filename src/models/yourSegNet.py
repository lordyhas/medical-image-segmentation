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
import torch.nn.functional as F
from src.models.CNNBaseModel import CNNBaseModel

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourSegNet.  Un réseau très différent du UNet.
Soyez originaux et surtout... amusez-vous!

'''


class YourSegNet(CNNBaseModel):
    """
    Architecture non-symétrique inspirée de DeepLabV3 avec :
    - Encoder profond avec atrous convolutions
    - Decoder léger avec upsampling progressif
    - Skip connections asymétriques
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds YourSegNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__(num_classes, init_weights)

        # Encoder profond
        in_channels = 1
        self.encoder1 = self._encoder_block(in_channels, 64)
        self.encoder2 = self._encoder_block(64, 128)
        self.encoder3 = self._encoder_block(128, 256)
        self.encoder4 = self._atrous_block(256, 512, dilation=2)
        self.encoder5 = self._atrous_block(512, 1024, dilation=4)

        # Decoder léger
        self.decoder4 = self._decoder_block(1024, 512)
        self.decoder3 = self._decoder_block(512, 256)
        self.decoder2 = self._decoder_block(256, 128)
        self.decoder1 = self._decoder_block(128, 64)

        # Couche finale
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoder avec skip connections asymétriques
        d4 = self.decoder4(e5)
        d3 = self.decoder3(d4) + e3  # Skip connection asymétrique
        d2 = self.decoder2(d3) + e2  # Skip connection asymétrique
        d1 = self.decoder1(d2) + e1  # Skip connection asymétrique

        # Upsample final pour obtenir la taille d'origine
        output = self.final(d1)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

    @staticmethod
    def _encoder_block(in_channels, out_channels):
        """
        Bloc encodeur avec convolutions standard
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    @staticmethod
    def _atrous_block(in_channels, out_channels, dilation):
        """
        Bloc avec atrous convolutions pour réception field plus grand
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    @staticmethod
    def _decoder_block(in_channels, out_channels):
        """
        Bloc décodeur avec upsampling
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )


'''
Fin de votre code.
'''