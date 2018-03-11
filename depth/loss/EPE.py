from __future__ import absolute_import

import torch

from torch import nn


class EPE(nn.Module):
  
  def __init__(self):
    super(EPE, self).__init__()

  def forward(self, outputs, targets):
    ground_truths, masks = targets
    epe = ((torch.abs(ground_truths - outputs) * masks).sum(3).sum(2).sum(1) / masks.sum(3).sum(2).sum(1)).mean()
    return epe

