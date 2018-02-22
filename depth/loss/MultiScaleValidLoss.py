from __future__ import absolute_import

import torch

from torch import nn
from .EPE import EPE

class MultiScaleValidLoss(nn.Module):

  def __init__(self, max_num_scales=5, loss_weight=0.32):
    
    super(MultiScaleValidLoss, self).__init__()

    self.loss_labels = ['MS', 'EPE']
    self.max_num_scales = max_num_scales
    self.loss_weight = loss_weight
    self.epe = EPE()

  def forward(self, outputs, targets):
    
    if self.loss_weight == 0:
      raise NotImplementedError('Automatic loss_weight deduction to be implemented.')
    loss_weights = [self.loss_weight / 2 ** i for i in range(len(outputs))]
    loss = torch.autograd.Variable(torch.zeros([1])).cuda() # A scalar Tensor with value 0
    for i, output in enumerate(outputs):
      loss += loss_weights[i] * self.epe(output, targets)

    epe = self.epe(outputs[0], targets)

    return loss, epe

