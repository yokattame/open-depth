from __future__ import absolute_import

import torch

from torch import nn


class EPE(nn.Module):
  
  def __init__(self):
    super(EPE, self).__init__()

  def forward(self, outputs, targets):
    ground_truth, mask = targets
    epe = (torch.abs(groud_truth - outputs) * mask).sum(3).sum(2).mean()
    return epe

