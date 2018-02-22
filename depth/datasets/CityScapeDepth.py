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

    train_list = []
    with open('data_lists/CityScapeDepth.txt', 'r') as input_file:
      for line in input_file:
        train_list.append(line)
    train_list = np.asarray(train_list)
    num = train_list.shape[0]
    num_val = int(num * 0.2)
    np.random.shuffle(train_list)
    val_list = train_list[:num_val]
    train_list = train_list[num_val:]
    assert(num == val_list.shape[0] + train_list.shape[0])
    with open('data_lists/CityScapeDepth_train.txt', 'w') as output_file:
      for line in train_list:
        print(line[:-1], file=output_file)
    with open('data_lists/CityScapeDepth_val.txt', 'w') as output_file:
      for line in val_list:
        print(line[:-1], file=output_file)

