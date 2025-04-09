import torch
import numpy as np
import scipy.ndimage

class ElasticTransform:
        def __init__(self, alpha=500, sigma=20):
            self.alpha = alpha
            self.sigma = sigma
            
        def __call__(self, img):
            if isinstance(img, torch.Tensor):
                is_tensor = True
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                img_np = img.numpy()
            else:
                is_tensor = False
                if img.ndim == 2:
                    img_np = img[np.newaxis, ...]
                else:
                    img_np = img
                    
            h, w = img_np.shape[1], img_np.shape[2]
            channels = img_np.shape[0]
            transformed = np.zeros_like(img_np)
            
            # Génère un champ de déplacement aléatoire à la taille de l'image
            dx = np.random.rand(h, w) * 2 - 1
            dy = np.random.rand(h, w) * 2 - 1
            dx = self.alpha * scipy.ndimage.gaussian_filter(dx, self.sigma)
            dy = self.alpha * scipy.ndimage.gaussian_filter(dy, self.sigma)
            
            # Crée une grille de coordonnées
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            
            # Applique les mêmes déformations à tous les canaux
            for c in range(channels):
                transformed[c] = scipy.ndimage.map_coordinates(
                    img_np[c], 
                    indices, 
                    order=1,
                    mode='reflect'
                ).reshape(h, w)
            
            if is_tensor:
                if img.dim() == 3 and img.shape[0] == 1:
                    return torch.from_numpy(transformed.squeeze(0))
                return torch.from_numpy(transformed)
            else:
                if img.ndim == 2:
                    return transformed.squeeze(0)
                return transformed