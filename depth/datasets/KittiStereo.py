from __future__ import absolute_import

import os
import os.path as osp

from glob import glob


class KittiStereo(object):
  
  def __init__(self, root):
    if osp.exists('data_lists/KittiStereo.txt'):
      return
    if not osp.exists('data_lists'):
      os.mkdir('data_lists')
    file_list = sorted(glob(osp.join(root, 'disp_occ_0', '*.png')))
    with open('data_lists/KittiStereo.txt', 'w') as output_file:
      for disp_image in file_list:
        fname = osp.basename(disp_image)
        left_image = osp.join(root, 'image_2', fname)
        right_image = osp.join(root, 'image_3', fname)
        print(left_image, right_image, disp_image, file=output_file)

