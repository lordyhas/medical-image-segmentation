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
from src.models.CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock

'''
Réseau YourSegNet amélioré avec:
- Blocs résiduels
- Blocs denses
- Blocs bottleneck
- Convolutions dilatées (atrous)
- Skip connections asymétriques
'''


class YourSegNet(CNNBaseModel):
    """
    Architecture non-symétrique inspirée de DeepLabV3 avec :
    - Encoder profond avec atrous convolutions et blocs résiduels
    - Decoder léger avec upsampling progressif et blocs denses
    - Skip connections asymétriques améliorées
    - Blocs bottleneck pour réduire les paramètres
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

        # Encoder profond avec blocs résiduels
        in_channels = 1
        self.encoder1 = self._encoder_block(in_channels, 64)
        self.encoder2 = self._encoder_block(64, 128)
        self.encoder3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # Stride 2 pour le downsampling
            ResidualBlock(256, 256)
        )

        # Blocs Atrous-Bottleneck pour capturer les contextes multi-échelles
        self.encoder4 = nn.Sequential(
            self._atrous_bottleneck_block(256, 512, dilation=2),
            ResidualBlock(512, 512)
        )
        self.encoder5 = nn.Sequential(
            self._atrous_bottleneck_block(512, 1024, dilation=4),
            ResidualBlock(1024, 1024)
        )

        # Decoder léger avec blocs denses pour une meilleure fusion des caractéristiques
        self.decoder4 = self._decoder_block(1024, 512)
        self.decoder3 = self._decoder_block(512, 256)
        self.decoder2 = self._decoder_block(256, 128)
        self.decoder1 = self._decoder_block(128, 64)

        # Couches de fusion pour les skip connections
        # Important: assurez-vous que la sortie a le même nombre de canaux que l'entrée du decoder suivant
        self.fusion3 = self._fusion_block(256 + 256, out_channels=256)  # Sortie: 256 canaux pour decoder2
        self.fusion2 = self._fusion_block(128 + 128, out_channels=128)  # Sortie: 128 canaux pour decoder1
        self.fusion1 = self._fusion_block(64 + 64, out_channels=64)  # Sortie: 64 canaux pour la couche finale

        # Couche finale
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

        # Decoder avec skip connections améliorées
        d4 = self.decoder4(e5)

        # Concaténation et fusion
        d3_concat = torch.cat([self.decoder3(d4), e3], dim=1)
        d3 = self.fusion3(d3_concat)

        d2_concat = torch.cat([self.decoder2(d3), e2], dim=1)
        d2 = self.fusion2(d2_concat)

        d1_concat = torch.cat([self.decoder1(d2), e1], dim=1)
        d1 = self.fusion1(d1_concat)

        # Upsample final pour obtenir la taille d'origine
        output = self.final(d1)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

    @staticmethod
    def _encoder_block(in_channels, out_channels):
        """
        Bloc encodeur avec convolutions standard et ResidualBlock
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels),  # Ajout d'un bloc résiduel
            nn.MaxPool2d(2)
        )

    @staticmethod
    def _atrous_bottleneck_block(in_channels, out_channels, dilation):
        """
        Bloc combinant atrous convolutions et bottleneck pour efficacité
        """
        mid_channels = out_channels // 4  # Facteur de réduction bottleneck
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
        Bloc décodeur avec upsampling et less convolutions
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    @staticmethod
    def _fusion_block(in_channels, out_channels):
        """
        Bloc de fusion utilisant DenseBlock pour améliorer la combinaison des features

        Args:
            in_channels: nombre de canaux d'entrée (sum des canaux concaténés)
            out_channels: nombre de canaux de sortie (doit correspondre à l'entrée du bloc suivant)
        """
        mid_channels = in_channels // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),  # Réduction initiale des canaux
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            DenseBlock(mid_channels),  # Ajoute mid_channels en concaténant -> mid_channels*2
            nn.Conv2d(mid_channels * 2, out_channels, 1),  # Ajustement au nombre de canaux de sortie souhaité
            nn.ReLU()
        )


'''
Fin de votre code.
'''