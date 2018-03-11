from __future__ import absolute_import

from .L1ValidLoss import L1ValidLoss
from .MultiScaleValidLoss import MultiScaleValidLoss


__factory = {
  'L1ValidLoss': L1ValidLoss,
  'MultiScaleValidLoss': MultiScaleValidLoss,
}


def names():
  return sorted(__factory.keys())


def create(name, metrics):
  if name not in __factory:
    raise KeyError('Undefined loss:', name)
  return __factory[name](metrics=metrics)

