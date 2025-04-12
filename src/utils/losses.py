import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Fonction de perte Dice pour YourUNet
    Combine la perte Dice avec la perte d'entropie croisée pour une meilleure stabilité
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # S'assurer que les prédictions ont la bonne forme.
        if len(predictions.shape) == 4:
            # Assurer que les cibles sont des entiers valides dans la plage [0, num_classes-1]
            num_classes = predictions.size(1)
            targets = torch.clamp(targets, min=0, max=num_classes - 1)

            # Appliquer softmax pour obtenir les probabilités
            predictions_softmax = F.softmax(predictions, dim=1)

            # Convertir les cibles en one-hot encoding avec permutation des dimensions
            # Reshape pour correspondre à la dimension attendue [B, C, H, W]
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Calculer la perte Dice pour chaque classe
            dice_loss = 0
            for i in range(predictions.size(1)):
                pred = predictions_softmax[:, i, :, :]
                target = targets_one_hot[:, i, :, :]

                intersection = (pred * target).sum()
                union = pred.sum() + target.sum()

                # Éviter la division par zéro
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                dice_loss += 1 - dice

            # Moyenne sur toutes les classes
            dice_loss = dice_loss / predictions.size(1)

            # Calculer l'entropie croisée avec les cibles vérifiées
            ce_loss = F.cross_entropy(predictions, targets.long())

            return 0.5 * dice_loss + 0.5 * ce_loss
        else:
            raise ValueError("Les prédictions doivent avoir 4 dimensions [B, C, H, W]")


class FocalLoss(nn.Module):
    """
    Fonction de perte Focal pour YourSegNet
    Utile pour gérer le déséquilibre des classes
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        # S'assurer que les cibles sont des entiers valides
        num_classes = predictions.size(1)
        targets = torch.clamp(targets, min=0, max=num_classes - 1)

        ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()