from __future__ import absolute_import
import torch
import numpy as np
import random

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


class DepthDataset(Dataset):

  def __init__(self, datalist, input_size, transform=None, random_crop=False):
    self.datalist = datalist
    self.transform = transform
    self.input_size = input_size
    self.random_crop = random_crop

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
    
    if self.random_crop:
      cropper = StaticRandomCrop(image_size, crop_size)
    else:
      cropper = StaticCenterCrop(image_size, crop_size)
    
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

