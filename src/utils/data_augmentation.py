#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date: 2023
License: Opensource, free to use
Other: Advanced data augmentation for cardiac MRI images and their segmentation masks
"""

import torch
import random
import numpy as np
import torchvision.transforms as T


class ACDCJointTransform:
    """
    Class that applies the same random transformations to an image and its mask.
    It uses standard torchvision transforms while ensuring consistency.

    Args:
        rotation (bool): Enable random rotation
        elastic (bool): Enable elastic deformation
        rotation_degrees (float): Maximum rotation angle (+/-)
        elastic_alpha (float): Amplitude of the elastic deformation
        elastic_sigma (float): Smoothing of the elastic deformation
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
        Applies random transformations to the image or mask.
        Automatically handles conversion between numpy arrays and torch.Tensor.

        Args:
            img: Image or mask to transform (numpy array or torch.Tensor)
        Returns:
            Transformed image or mask in the same format as the input
        """
        # Convert to tensor if needed
        was_tensor = isinstance(img, torch.Tensor)
        if not was_tensor:
            img = torch.from_numpy(img.copy()).float()

        # Save the original shape
        original_shape = img.shape

        # Prepare the tensor for torchvision (requires C,H,W)
        if len(original_shape) == 2:  # 2D image (H,W)
            img = img.unsqueeze(0)
        elif len(original_shape) == 3 and original_shape[2] <= 4:  # H,W,C
            img = img.permute(2, 0, 1)

        # Fix the seed for the transformations
        seed = random.randint(0, 2 ** 32 - 1)

        # Apply rotation if requested
        if self.rotation:
            torch.manual_seed(seed)
            random.seed(seed)
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = T.functional.rotate(img, angle, T.InterpolationMode.BILINEAR)

        # Apply elastic deformation if requested
        if self.elastic and hasattr(T, 'ElasticTransform'):  # Available in torchvision >= 0.9.0
            torch.manual_seed(seed)
            random.seed(seed)
            elastic_transform = T.ElasticTransform(
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma
            )
            img = elastic_transform(img)

        # Return to the original format if needed
        if len(original_shape) == 2:
            img = img.squeeze(0)
        elif len(original_shape) == 3 and original_shape[2] <= 4:
            img = img.permute(1, 2, 0)

        # Return the result in the original format
        if not was_tensor:
            img = img.numpy()

        return img