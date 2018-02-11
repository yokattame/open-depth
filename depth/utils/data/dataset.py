from __future__ import absolute_import
import torch
import numpy as np
import random
import math

from scipy.misc import imread
from torch.utils.data import Dataset


class StaticRandomCrop(object):

  def __init__(self, image_size, crop_size):
    self.crop_h, self.crop_w = crop_size
    h, w = image_size
    self.start_h = random.randint(0, h - self.crop_h)
    self.start_w = random.randint(0, w - self.crop_w)

  def __call__(self, image):
    return image[self.start_h:(self.start_h + self.crop_h), self.start_w:(self.start_w + self.crop_w)]


class StaticCenterCrop(object):

  def __init__(self, image_size, crop_size):
    self.crop_h, self.crop_w = crop_size
    self.h, self.w = image_size

  def __call__(self, image):
    return image[int((self.h - self.crop_h) // 2):int((self.h + self.crop_h) // 2), int((self.w - self.crop_w) // 2):int((self.w + self.crop_w) // 2)]


class StaticPadTo64x(object):

  def __init__(self, image_size):
    height, width = image_size
    padding = [0, 0, 0, 0]
    if width % 64 != 0:
      padding[0] += int(math.floor((64 - width % 64) / 2))
      padding[2] += int(math.ceil((64 - width % 64) / 2))
    if height % 64 != 0:
      padding[1] += int(math.floor((64 - height % 64) / 2))
      padding[3] += int(math.ceil((64 - height % 64) / 2))
    self.padding = tuple(padding)

  def __call__(self, image):
    height, width, channels = image.shape
    image = np.concatenate((np.zeros((height, self.padding[0], channels)), image, np.zeros((height, self.padding[2], channels))), axis=1)
    width = image.shape[1]
    image = np.concatenate((np.zeros((self.padding[1], width, channels)), image, np.zeros((self.padding[3], width, channels))), axis=0)
    return image.astype(np.float32)


class DepthDataset(Dataset):

  def __init__(self, datalist, input_size, transform=None, crop_method='Center'):
    self.datalist = datalist
    self.transform = transform
    self.input_size = input_size
    self.crop_method = crop_method

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
    crop_size = image_size
    if self.input_size != (0, 0):
      crop_size = self.input_size
    else:
      self.crop_method = 'Pad'
    
    if self.crop_method == 'Random':
      cropper = StaticRandomCrop(image_size, crop_size)
    elif self.crop_method == 'Center':
      cropper = StaticCenterCrop(image_size, crop_size)
    elif self.crop_method == 'Pad':
      cropper = StaticPadTo64x(image_size)
    else:
      raise ValueError('Illegal parameter crop_method: ' + self.crop_method)
    
    left_image = cropper(left_image)
    right_image = cropper(right_image)
    disp_image = cropper(disp_image)
    if self.transform is not None:
      left_image = self.transform(left_image)
      right_image = self.transform(right_image)
    mask = np.copy(disp_image)

    valid = np.count_nonzero(mask)
    axis = np.transpose(np.nonzero(mask))

    for x, y, z in axis:
      mask[x][y][z] = 1.0 / valid
          
    disp_image = torch.from_numpy(disp_image.transpose(2, 0, 1))
    mask = torch.from_numpy(mask.transpose(2, 0, 1))

    return left_image, right_image, disp_image, mask

