from __future__ import absolute_import

import torch
from torch import nn

from .MetricFactory import MetricFactory
from .EPE import EPE
from .Outliers import Outliers


class BaseLoss(nn.Module):

  def __init__(self, metrics=MetricFactory.create_metric_bundle(['EPE', 'D1'])):
    super().__init__()
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

