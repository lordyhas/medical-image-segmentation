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

        predictions = F.softmax(predictions, dim=1)

        # cibles en one-hot encoding
        targets = F.one_hot(targets, num_classes=predictions.size(1)).float()


        dice_loss = 0
        for i in range(predictions.size(1)):
            pred = predictions[:, i:i + 1]
            target = targets[:, i:i + 1]

            intersection = (pred * target).sum()
            dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
            dice_loss += 1 - dice

        dice_loss = dice_loss / predictions.size(1)


        ce_loss = F.cross_entropy(predictions, targets.argmax(dim=1))

        return 0.5 * dice_loss + 0.5 * ce_loss
