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

Ajouter du code ici pour faire fonctionner le réseau YourUNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''


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

'''
Fin de votre code.
'''