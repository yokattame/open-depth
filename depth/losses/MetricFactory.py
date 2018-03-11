from __future__ import absolute_import

from .EPE import EPE
from .Outliers import Outliers


class MetricFactory:
  
  @classmethod
  def create_metric(cls, metric_name):
    if metric_name == 'EPE':
      return EPE()
    elif metric_name == 'bad3':
      return Outliers(absolute_threshold=3, relative_threshold=0.0)
    elif metric_name == 'D1':
      return Outliers(absolute_threshold=3, relative_threshold=0.05)
    else:
      raise ValueError('Undefined metric: ' + metric_name)

  @classmethod
  def create_metric_bundle(cls, metric_names):
    metric_bundle = {}
    for metric_name in metric_names:
      metric_bundle[metric_name] = cls.create_metric(metric_name)
    return metric_bundle

