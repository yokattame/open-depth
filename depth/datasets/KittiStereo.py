from __future__ import absolute_import

import os
import os.path as osp
import numpy as np

from glob import glob


class KittiStereo(object):
  
  def __init__(self, root):
    if osp.exists('data_lists/KittiStereo.txt'):
      return
    if not osp.exists('data_lists'):
      os.mkdir('data_lists')
    file_list = sorted(glob(osp.join(root, 'training', 'disp_occ_0', '*.png')))
    with open('data_lists/KittiStereo.txt', 'w') as output_file:
      for disp_image in file_list:
        fname = osp.basename(disp_image)
        left_image = osp.join(root, 'training', 'image_2', fname)
        right_image = osp.join(root, 'training', 'image_3', fname)
        print(left_image, right_image, disp_image, file=output_file)

    train_list = []
    with open('data_lists/KittiStereo.txt', 'r') as input_file:
      for line in input_file:
        train_list.append(line)
    train_list = np.asarray(train_list)
    num = train_list.shape[0]
    num_val = int(num * 0.2)
    np.random.shuffle(train_list)
    val_list = train_list[:num_val]
    train_list = train_list[num_val:]
    assert(num == val_list.shape[0] + train_list.shape[0])
    with open('data_lists/KittiStereo_train.txt', 'w') as output_file:
      for line in train_list:
        print(line[:-1], file=output_file)
    with open('data_lists/KittiStereo_val.txt', 'w') as output_file:
      for line in val_list:
        print(line[:-1], file=output_file)

