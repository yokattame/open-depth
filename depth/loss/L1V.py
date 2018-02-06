from __future__ import absolute_import

import torch

from torch import nn

#def accuracy(output, target):
#    within_3px = (torch.abs(target - output) <= 3).type(torch.cuda.FloatTensor)
#    within_5perc = ((torch.abs((target - output) / target) <= 0.05)).type(torch.cuda.FloatTensor)
#    return (within_3px * within_5perc).mean()

class L1V(nn.Module):
  
  def __init__(self):
    super(L1V, self).__init__()
    #self.loss_labels = ['L1', 'EPE', 'D1all']
    self.loss_labels = ['L1', 'EPE']

  def forward(self, outputs, targets):
    ground_truth, mask = targets
    loss = (torch.abs(ground_truth - outputs) * mask).sum(3).sum(2).mean()
    epe = loss
    #return (loss, epe, d1all)
    return (loss, epe)

