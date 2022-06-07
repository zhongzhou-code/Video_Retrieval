import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import itertools


class AveragePrecisionLoss(nn.Module):
    def __init__(self, margin, device):
        super(AveragePrecisionLoss, self).__init__()
        self.feature_size = None
        self.batch_size = None
        self.margin = margin
        self.device = device

    def similarity(self, label1, label2):
        return label1 == label2     # default with singe label

    def forward(self, x, labels):
        return