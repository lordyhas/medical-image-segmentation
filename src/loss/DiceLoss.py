import torch.nn as nn

from utils.utils import mean_dice


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - mean_dice(inputs, targets)
