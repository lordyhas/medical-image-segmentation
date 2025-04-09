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

Ajouter du code ici pour faire fonctionner le réseau YourUNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''


class YourUNet(CNNBaseModel):
    """
    Version améliorée du UNet avec :
    - Convolutions séparables
    - Couches résiduelles
    - Blocs bottleneck
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds yourUNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__(num_classes, init_weights)

        # Encoder
        in_channels = 1
        self.conv_encoder1 = self._conv_block(in_channels=in_channels, out_channels=64)
        self.conv_encoder2 = self._contraction_block(64, 128)
        self.conv_encoder3 = self._contraction_block(128, 256)
        self.conv_encoder4 = self._contraction_block(256, 512)
        self.conv_encoder5 = self._contraction_block(512, 1024)

        # Bottleneck
        self.bottleneck = self._bottleneck_block(1024, 1024)

        # Middle
        self.conv_mid_decoder = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                                   kernel_size=3, stride=2, padding=1, output_padding=1)

        # Decoder
        self.conv_decoder4 = self._expansion_block(1024, 512)  # 512 + 512 = 1024
        self.conv_decoder3 = self._expansion_block(768, 256)  # 256 + 256 = 512
        self.conv_decoder2 = self._expansion_block(384, 128)  # 128 + 128 = 256
        self.final_layer = self._final_block(192, 64, num_classes)  # 64 + 64 = 128

        # Couche supplémentaire pour assurer les bonnes dimensions -
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encoder1(x)
        encode_block2 = self.conv_encoder2(encode_block1)
        encode_block3 = self.conv_encoder3(encode_block2)
        encode_block4 = self.conv_encoder4(encode_block3)
        encode_block5 = self.conv_encoder5(encode_block4)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block5)

        # Middle
        decode_middle = self.conv_mid_decoder(bottleneck)

        # Decode with residual connections
        decode_block4 = torch.cat((decode_middle, encode_block4), 1)
        cat_layer3 = self.conv_decoder4(decode_block4)

        decode_block3 = torch.cat((cat_layer3, encode_block3), 1)
        cat_layer2 = self.conv_decoder3(decode_block3)

        decode_block2 = torch.cat((cat_layer2, encode_block2), 1)
        cat_layer1 = self.conv_decoder2(decode_block2)

        decode_block1 = torch.cat((cat_layer1, encode_block1), 1)
        final_layer = self.final_layer(decode_block1)

        # Nous retournons directement le résultat sans softmax
        # Le softmax sera appliqué dans la fonction de perte
        return final_layer

    @staticmethod
    def _separable_conv(in_channels, out_channels, kernel_size=3):
        """
        Implémente une convolution séparable (depthwise + pointwise)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    @staticmethod
    def _residual_block(in_channels, out_channels):
        """
        Bloc résiduel avec convolutions séparables
        """
        return nn.Sequential(
            YourUNet._separable_conv(in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            YourUNet._separable_conv(out_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=3):
        """
        Bloc de convolution avec connexions résiduelles
        """
        return nn.Sequential(
            YourUNet._residual_block(in_channels, out_channels),
            YourUNet._residual_block(out_channels, out_channels)
        )

    @staticmethod
    def _contraction_block(in_channels, out_channels, kernel_size=3):
        """
        Bloc de contraction avec convolutions séparables
        """
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            YourUNet._residual_block(in_channels, out_channels),
            YourUNet._residual_block(out_channels, out_channels)
        )

    @staticmethod
    def _bottleneck_block(in_channels, out_channels):
        """
        Bloc bottleneck pour réduire la dimensionnalité
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels // 4),
            YourUNet._separable_conv(out_channels // 4, out_channels // 4),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels // 4),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    @staticmethod
    def _expansion_block(in_channels, out_channels, kernel_size=3):
        """
        Bloc d'expansion avec convolutions séparables
        """
        return nn.Sequential(
            YourUNet._residual_block(in_channels, out_channels),
            YourUNet._residual_block(out_channels, out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    @staticmethod
    def _final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Bloc final avec convolutions séparables
        """
        return nn.Sequential(
            YourUNet._residual_block(in_channels, mid_channels),
            YourUNet._residual_block(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1)  # Sortie finale avec le nombre de classes
        )


'''
Fin de votre code.
'''