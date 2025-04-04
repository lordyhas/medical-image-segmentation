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
        self.smooth = smooth # pour éviter la division par zéro

    def forward(self, predictions, targets):

        predictions = F.softmax(predictions, dim=1)

        # cibles en one-hot encoding
        targets = F.one_hot(targets, num_classes=predictions.size(1)).float()

        # Calculer la perte Dice pour chaque classe
        dice_loss = 0
        for i in range(predictions.size(1)):
            pred = predictions[:, i:i + 1]
            target = targets[:, i:i + 1]

            intersection = (pred * target).sum()
            dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
            dice_loss += 1 - dice

        dice_loss = dice_loss / predictions.size(1)

        # Combiner avec l'entropie croisée
        ce_loss = F.cross_entropy(predictions, targets.argmax(dim=1))

        return 0.5 * dice_loss + 0.5 * ce_loss

class FocalLoss(nn.Module):
    """
    Fonction de perte Focal pour YourUNet
    Conçue pour traiter le déséquilibre entre les classes
    """

    def __init__(self, alpha=1.0, gamma=2.0, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # pour équilibrer les classes
        self.gamma = gamma # pour ajuster la focalisation
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = F.softmax(predictions, dim=1)
        targets = F.one_hot(targets, num_classes=predictions.size(1)).float()

        BCE_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()