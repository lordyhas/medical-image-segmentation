import torch
import torch.nn as nn
import torch.nn.functional as F

# Author: Hassan Kajila

class DiceLoss(nn.Module):
    """
    Dice loss function for YourUNet.
    Combines Dice loss with cross-entropy loss for better stability.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Ensure predictions have the correct shape
        if len(predictions.shape) == 4:
            # Ensure targets are valid integers in the range [0, num_classes - 1]
            num_classes = predictions.size(1)
            targets = torch.clamp(targets, min=0, max=num_classes - 1)

            # Apply softmax to obtain probabilities
            predictions_softmax = F.softmax(predictions, dim=1)

            # Convert targets to one-hot encoding with dimension permutation
            # Reshape to match the expected shape [B, C, H, W]
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Compute Dice loss for each class
            dice_loss = 0
            for i in range(predictions.size(1)):
                pred = predictions_softmax[:, i, :, :]
                target = targets_one_hot[:, i, :, :]

                intersection = (pred * target).sum()
                union = pred.sum() + target.sum()

                # Avoid division by zero
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                dice_loss += 1 - dice

            # Average across all classes
            dice_loss = dice_loss / predictions.size(1)

            # Compute cross-entropy with validated targets
            ce_loss = F.cross_entropy(predictions, targets.long())

            return 0.5 * dice_loss + 0.5 * ce_loss
        else:
            raise ValueError("Predictions must have 4 dimensions [B, C, H, W]")


class FocalLoss(nn.Module):
    """
    Focal loss function for YourSegNet.
    Useful for handling class imbalance.
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        # Ensure targets are valid integers
        num_classes = predictions.size(1)
        targets = torch.clamp(targets, min=0, max=num_classes - 1)

        ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()