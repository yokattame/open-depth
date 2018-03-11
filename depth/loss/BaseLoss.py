from __future__ import absolute_import

import torch
from torch import nn

from .EPE import EPE
from .Outliers import Outliers


class BaseLoss(nn.Module):

  def __init__(self, metrics={'EPE': EPE(), 'D1-all': Outliers(absolute_threshold=3, relative_threshold=0.05)}):
    super(BaseLoss, self).__init__()
    self.metrics = metrics
  
  def get_metric_keys(self):
    return self.metrics.keys()

  def get_metric_values(self, outputs, targets):
    metric_values = dict()
    for metric_key, metric in self.metrics.items():
      metric_values[metric_key] = metric(outputs, targets)
    return metric_values

  def get_num_metrics(self):
    return len(self.metrics)

