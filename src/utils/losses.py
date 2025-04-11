import torch.nn as nn
import torch.nn.functional as F

from utils.utils import mean_dice


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - mean_dice(inputs, targets)



class DiceCELoss(nn.Module):

    def __init__(self):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets):
        dice_loss = DiceLoss()(inputs, targets)
        cross_entropy = F.cross_entropy(inputs, targets, reduction='mean')
        return cross_entropy + dice_loss
