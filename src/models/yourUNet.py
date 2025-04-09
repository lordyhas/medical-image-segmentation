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
def center_crop(enc_feat, target_feat):
    _, _, H, W = target_feat.size()
    return torchvision.transforms.CenterCrop([H, W])(enc_feat)


class YourUNet(CNNBaseModel):
    """
     Class that implements a brand new UNet segmentation network
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
        # encoder
        in_channels = 1  # gray image
        self.conv_encoder1 = self._conv_block(in_channels=in_channels, out_channels=64)
        self.conv_encoder2 = self._contraction_block(64, 128)
        self.conv_encoder3 = self._contraction_block(128, 256)
        self.conv_encoder4 = self._contraction_block(256, 512)
        self.conv_encoder5 = self._contraction_block(512, 1024)

        # Middle
        self.conv_mid_decoder = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        

        # Decode
        self.conv_decoder4 = self._expansion_block(1024, 256)
        self.conv_decoder3 = self._expansion_block(512, 128)
        self.conv_decoder2 = self._expansion_block(256, 64)
        self.final_layer = self._final_block(128, 64, num_classes)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode
        encode_block1 = self.conv_encoder1(x)
        encode_block2 = self.conv_encoder2(encode_block1)
        encode_block3 = self.conv_encoder3(encode_block2)
        encode_block4 = self.conv_encoder4(encode_block3)
        encode_block5 = self.conv_encoder5(encode_block4)

        # middle
        decode_middle = self.conv_mid_decoder(encode_block5)
        
        # Decode
        decode_block4 = torch.cat((decode_middle, encode_block4), 1)
        cat_layer3 = self.conv_decoder4(decode_block4)
        
        encode_block3 = center_crop(encode_block3, cat_layer3)
        decode_block3 = torch.cat((cat_layer3, encode_block3), dim=1)
        cat_layer2 = self.conv_decoder3(decode_block3)
        encode_block2 = center_crop(encode_block2, cat_layer2)
        decode_block2 = torch.cat((cat_layer2, encode_block2), 1)
        cat_layer1 = self.conv_decoder2(decode_block2)
        encode_block1 = center_crop(encode_block1, cat_layer1)
        decode_block1 = torch.cat((cat_layer1, encode_block1), 1)
        final_layer = self.final_layer(decode_block1)
        final_layer = F.interpolate(final_layer, size=(256, 256), mode='bilinear', align_corners=False)
        return final_layer

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    @staticmethod
    def _contraction_block(in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part
        """
        
        block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        return block

    @staticmethod
    def _expansion_block(in_channels, out_channels, kernel_size=3):
        """
        Building block of the expansion part
        """
        block = nn.Sequential(
            DenseBlock(in_channels),
            nn.BatchNorm2d(2*in_channels),
            BottleneckBlock(2*in_channels),
            nn.BatchNorm2d(2*in_channels),
            nn.ConvTranspose2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=1,
                               output_padding=1)
        )
        
        return block

    @staticmethod
    def _final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Final block of the UNet model
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block
'''
Fin de votre code.
'''