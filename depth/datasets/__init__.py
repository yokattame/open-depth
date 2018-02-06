from __future__ import absolute_import

from .KittiStereo import KittiStereo
from .CityScapeDepth import CityScapeDepth

__factory = {
  'KittiStereo': KittiStereo,
  'CityScapeDepth': CityScapeDepth,
}

def create(name, root):
  if name not in __factory:
    raise KeyError("Unknown dataset:", name)
  return __factory[name](root)

