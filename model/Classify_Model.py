import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from torch.autograd import Variable

class Classify(nn.Module):

    def __init__(self, numbers_of_categories):
        super(Classify, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, numbers_of_categories),
            nn.ReLU(inplace=True),
            nn.Linear(numbers_of_categories, numbers_of_categories),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        #
        return x

