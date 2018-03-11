from __future__ import absolute_import

import torch

from .BaseLoss import BaseLoss
from .EPE import EPE
from .Outliers import Outliers

class L1ValidLoss(BaseLoss):
  
  def __init__(self, metrics=None):
    if metrics is not None:
      super().__init__(metrics=metrics)
    else:
      super().__init__()

  def forward(self, outputs, targets):
    outputs = outputs[0]
    metric_values = self.get_metric_values(outputs, targets)
    loss = metric_values['EPE']
    return loss, metric_values

