from __future__ import print_function, absolute_import
import os.path as osp
import shutil

import torch

from .osutils import mkdir_if_missing


def save_checkpoint(state, are_best, fpath='checkpoint.pth.tar'):
  mkdir_if_missing(osp.dirname(fpath))
  torch.save(state, fpath)
  for metric_name, is_best in are_bset.items():
    if is_best:
      shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_' + metric_name + '.pth.tar'))


def load_checkpoint(fpath):
  if osp.isfile(fpath):
    checkpoint = torch.load(fpath)
    print("=> Loaded checkpoint '{}'".format(fpath))
    return checkpoint
  else:
    raise ValueError("=> No checkpoint found at '{}'".format(fpath))

