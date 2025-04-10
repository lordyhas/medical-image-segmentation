import torch.nn as nn
import torch.nn.functional as F

from loss.DiceLoss import DiceLoss


# Idea from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

class DiceCELoss(nn.Module):

    def __init__(self):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets):
        dice_loss = DiceLoss()(inputs, targets)
        cross_entropy = F.cross_entropy(inputs, targets, reduction='mean')
        return cross_entropy + dice_loss
