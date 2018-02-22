from __future__ import absolute_import

import torch

from torch import nn
from .EPE import EPE

class L1ValidLoss(nn.Module):
  
  def __init__(self):
    super(L1ValidLoss, self).__init__()
    self.epe = EPE()
    self.loss_labels = ['L1', 'EPE']

  def forward(self, outputs, targets):
    outputs = outputs[0]
    epe = self.epe(outputs, targets)
    loss = epe
    return loss, epe

