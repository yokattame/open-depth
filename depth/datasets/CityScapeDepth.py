from __future__ import absolute_import

import os
import os.path as osp


class CityScapeDepth(object):
  
  def __init__(self, root):
    if osp.exists('data_lists/CityScapeDepth.txt'):
      return
    if not osp.exists('data_lists'):
      os.mkdir('data_lists')
    with open('data_lists/CityScapeDepth.txt', 'w') as output_file:
      for dirs, roots, files in os.walk(osp.join(root, 'disparity')):
        for f in files:
          disp_image = osp.join(dirs, f)
          left_image = disp_image.replace('disparity', 'leftImg8bit')
          right_image = disp_image.replace('disparity', 'rightImg8bit')
          print(left_image, right_image, disp_image, file=output_file)

