from __future__ import absolute_import

import torch

from .BaseLoss import BaseLoss
from .EPE import EPE
from .Outliers import Outliers

class L1ValidLoss(BaseLoss):
  
  def __init__(self, metrics={'EPE': EPE(), 'D1-all': Outliers(absolute_threshold=3, relative_threshold=0.05)}):
    super(L1ValidLoss, self).__init__(metrics)
    self.metrics = metrics

  def forward(self, outputs, targets):
    outputs = outputs[0]
    metric_values = self.get_metric_values(outputs, targets)
    loss = metric_values['EPE']
    return loss, metric_values

