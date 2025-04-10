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
        self.encoder1 = nn.Sequential(
            self._conv_block(in_channels=in_channels, out_channels=64),
            self._contraction_block(64, 128),
            ResidualBlock(128, 128),
            DenseBlock(128)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            BottleneckBlock(256),
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512, stride=1),
            nn.BatchNorm2d(512),
            DenseBlock(512),
        )

        # Middle
        self.mid_layer4 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2,
                                                   padding=1, output_padding=1)

        # Decode
        self.decoder5 = nn.Sequential(
            self._expansion_block(512, 256),
            ResidualBlock(256, 256, stride=1),
            nn.BatchNorm2d(256)
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            ResidualBlock(256, 128),
            DenseBlock(128),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0, output_padding=0),
            nn.ReLU(),
        )
        self.final_layer7 = self._final_block(128, 64, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode
        encode_block1 = self.encoder1(x)
        encode_block2 = self.encoder2(encode_block1)
        encode_block3 = self.encoder3(encode_block2)

        # middle
        middle_block4 = self.mid_layer4(encode_block3)

        # Decode
        decode_block5 = torch.cat((middle_block4, encode_block2), 1)
        cat_layer1 = self.decoder5(decode_block5)
        decode_block6 = torch.cat((cat_layer1, encode_block1), 1)
        decode_block7 = self.decoder6(decode_block6)
        final_layer = self.final_layer7(decode_block7)
        return final_layer

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=3):
        """
        Building block of the conv part with MaxPool
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
        )
        return block

    @staticmethod
    def _contraction_block(in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part with different padding
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
        )
        return block

    @staticmethod
    def _expansion_block(in_channels, out_channels, kernel_size=3):
        """
        Building block of the expansion part with upsampling
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

 
    @staticmethod
    def _final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Final block of the yourUNet model with 
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
            nn.BatchNorm2d(out_channels),
        )
        return block


'''
Fin de votre code.
'''