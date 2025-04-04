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
from src.models.CNNBaseModel import CNNBaseModel

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourSegNet.  Un réseau très différent du UNet.
Soyez originaux et surtout... amusez-vous!

'''

# source: https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/_deeplab.py
class YourSegNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
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
        # Encoder
        in_channels = 1
        self.conv_encoder1 = self._encoder_block(in_channels=in_channels, out_channels=64)
        self.conv_encoder2 = self._encoder_block(64, 128)
        self.conv_encoder3 = self._encoder_block(128, 256)
        self.conv_encoder4 = self._encoder_block(256, 512)
        self.conv_encoder5 = self._encoder_block(512, 1024)
        # Bottleneck
        self.bottleneck = self._bottleneck_block(1024, 2048)
        # Middle
        self.conv_mid_decoder = nn.ConvTranspose2d(in_channels=2048, out_channels=1024,
                                                   kernel_size=3, stride=2, padding=1, output_padding=1)
        # Decoder
        self.conv_decoder4 = self._decoder_block(2048, 512)
        self.conv_decoder3 = self._decoder_block(1024, 256)
        self.conv_decoder2 = self._decoder_block(512, 128)
        self.final_layer = self._final_block(256, 128, num_classes)
        # Final layer
        self.final_layer = nn.Conv2d(128, num_classes, kernel_size=1)
        self.final_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the model
        :param x:
        :return:
        """

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

        # Decode
        decode_block4 = self.conv_decoder4(decode_middle)
        decode_block3 = self.conv_decoder3(decode_block4)
        decode_block2 = self.conv_decoder2(decode_block3)

        return self.final_layer(decode_block2)

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
    def _decoder_block(in_channels, out_channels):
        """
        Bloc décodeur avec convolutions standard
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        )

'''
Fin de votre code.
'''