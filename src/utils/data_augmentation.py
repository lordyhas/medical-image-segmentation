#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date: 2023
License: Opensource, free to use
Other: Augmentation de données avancée pour les images IRM et leurs masques de segmentation
"""

import torch
import random
import numpy as np
import torchvision.transforms as T


class ACDCJointTransform:
    """
    Classe qui applique les mêmes transformations aléatoires à une image et son masque.
    Utilise les transformations standard de torchvision, mais garantit la cohérence.

    Args:
        rotation (bool): Activer la rotation aléatoire
        elastic (bool): Activer la déformation élastique
        rotation_degrees (float): Angle maximal de rotation (+ et -)
        elastic_alpha (float): Amplitude de la déformation élastique
        elastic_sigma (float): Lissage de la déformation élastique.
    """

    def __init__(self, rotation=False, elastic=False, rotation_degrees=60,
                 elastic_alpha=50.0, elastic_sigma=5.0):
        self.rotation = rotation
        self.elastic = elastic
        self.rotation_degrees = rotation_degrees
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    def __call__(self, img):
        """
        Applique les transformations aléatoires à l'image ou au masque.
        Gère automatiquement la conversion entre numpy et torch.Tensor.

        Args:
            img: Image ou masque à transformer (numpy array ou torch.Tensor)
        Returns:
            Image ou masque transformé dans le même format que l'entrée
        """
        # Conversion en tensor si nécessaire
        was_tensor = isinstance(img, torch.Tensor)
        if not was_tensor:
            img = torch.from_numpy(img.copy()).float()

        # Sauvegarde de la forme d'origine
        original_shape = img.shape

        # Préparation du tenseur pour torchvision (nécessite C,H,W)
        if len(original_shape) == 2:  # Image 2D (H,W)
            img = img.unsqueeze(0)
        elif len(original_shape) == 3 and original_shape[2] <= 4:  # H,W,C
            img = img.permute(2, 0, 1)

        # Fixer le seed pour les transformations
        seed = random.randint(0, 2 ** 32 - 1)

        # Appliquer la rotation si demandée
        if self.rotation:
            torch.manual_seed(seed)
            random.seed(seed)
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = T.functional.rotate(img, angle, T.InterpolationMode.BILINEAR)

        # Appliquer la déformation élastique si demandée
        if self.elastic and hasattr(T, 'ElasticTransform'):  # Disponible dans torchvision >= 0.9.0
            torch.manual_seed(seed)
            random.seed(seed)
            elastic_transform = T.ElasticTransform(
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma
            )
            img = elastic_transform(img)

        # Retour au format d'origine si nécessaire
        if len(original_shape) == 2:
            img = img.squeeze(0)
        elif len(original_shape) == 3 and original_shape[2] <= 4:
            img = img.permute(1, 2, 0)

        # Retourner le résultat dans le format d'origine
        if not was_tensor:
            img = img.numpy()

        return img