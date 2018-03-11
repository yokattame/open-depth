from __future__ import absolute_import

import torch
import numpy as np
from torch import nn


class Outliers(nn.Module):

  def __init__(self, absolute_threshold, relative_threshold):
    super(Outliers, self).__init__()
    self.absolute_threshold = absolute_threshold
    self.relative_threshold = relative_threshold

  def forward(self, outputs, targets):
    ground_truths, masks = targets
    differences = torch.abs(ground_truths - outputs) * masks
    is_outlier = np.logical_and.reduce((differences > self.absolute_threshold, differences > ground_truths * self.relative_threshold))
    outliers = (is_outlier.type(torch.FloatTensor).cuda().sum(3).sum(2).sum(1) / masks.sum(3).sum(2).sum(1)).mean()
    return outliers

