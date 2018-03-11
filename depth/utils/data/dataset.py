from __future__ import absolute_import
import torch
import numpy as np

from scipy.misc import imread
from torch.utils.data import Dataset

from .static_stereo_transforms import *


class DepthDataset(Dataset):

  def __init__(self, datalist, input_size, torch_transform=None, resize_method='Center Crop'):
    self.datalist = datalist
    self.torch_transform = torch_transform
    self.input_size = input_size
    self.resize_method = resize_method

  def __len__(self):
    return len(self.datalist)

  def __getitem__(self, index):
    line = self.datalist[index]
    left_image, right_image, disp_image = line.split(' ')
    left_image = imread(left_image)
    right_image = imread(right_image)
    disp_image = imread(disp_image[:-1])
    disp_image = np.array(disp_image, dtype=np.float32) / 256.0
    disp_image = np.expand_dims(disp_image, axis=2)

    image_size = left_image.shape[:2]
    input_size = image_size
    if self.input_size != (0, 0):
      input_size = self.input_size
    else:
      self.resize_method = 'Pad'
    
    if self.resize_method == 'Random Crop':
      transform = StaticStereoRandomCrop(image_size, input_size)
    elif self.resize_method == 'Center Crop':
      transform = StaticStereoCenterCrop(image_size, input_size)
    elif self.resize_method == 'Pad':
      transform = StaticStereoPadTo64x(image_size)
    elif self.resize_method == 'Resize':
      transform = StaticStereoResize(image_size, input_size)
    elif self.resize_method == 'Data Augment':
      transform = StaticStereoDataAugment(image_size, input_size)
    else:
      raise ValueError('Illegal parameter resize_method: ' + self.resize_method)
    
    left_image, right_image, disp_image = transform(left_image, right_image, disp_image)
    if self.torch_transform is not None:
      left_image = self.torch_transform(left_image)
      right_image = self.torch_transform(right_image)
    mask = np.copy(disp_image)

    axis = np.transpose(np.nonzero(mask))

    for x, y, z in axis:
      mask[x][y][z] = 1.0
          
    disp_image = torch.from_numpy(disp_image.transpose(2, 0, 1))
    mask = torch.from_numpy(mask.transpose(2, 0, 1))

    return left_image, right_image, disp_image, mask

