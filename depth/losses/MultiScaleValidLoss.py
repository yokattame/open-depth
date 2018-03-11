from __future__ import absolute_import

import torch

from .BaseLoss import BaseLoss
from .EPE import EPE
from .Outliers import Outliers


class MultiScaleValidLoss(BaseLoss):

  def __init__(self, max_num_scales=5, initial_weight=0.32, loss_decay=0.5, metrics=None):
    
    if metrics is not None:
      super().__init__(metrics=metrics)
    else:
      super().__init__()

    self.max_num_scales = max_num_scales
    self.initial_weight = initial_weight
    self.loss_decay = loss_decay
    self.epe = EPE()

  def forward(self, outputs, targets):
    
    if self.initial_weight == 0:
      raise NotImplementedError('Automatic initial_weight deduction to be implemented.')
    loss_weights = [self.initial_weight * self.loss_decay ** i for i in range(len(outputs))]
    loss = torch.autograd.Variable(torch.zeros([1])).cuda() # A scalar Tensor with value 0
    for i, output in enumerate(outputs):
      loss += loss_weights[i] * self.epe(output, targets)

    metric_values = self.get_metric_values(outputs[0], targets) # Compute metrics using only the final disparity output

    return loss, metric_values

