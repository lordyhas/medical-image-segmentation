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
from CNNBaseModel import CNNBaseModel

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourSegNet.  Un réseau très différent du UNet.
Soyez originaux et surtout... amusez-vous!

'''


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

'''
Fin de votre code.
'''