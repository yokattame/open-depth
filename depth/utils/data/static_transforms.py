from __future__ import absolute_import

import numpy as np
import cv2
import random
import math


class StaticRandomCrop(object):

  def __init__(self, original_size, new_size):
    self.crop_h, self.crop_w = new_size
    h, w = original_size
    self.start_h = random.randint(0, h - self.crop_h)
    self.start_w = random.randint(0, w - self.crop_w)

  def __call__(self, image):
    return image[self.start_h:(self.start_h + self.crop_h), self.start_w:(self.start_w + self.crop_w)]


class StaticCenterCrop(object):

  def __init__(self, original_size, new_size):
    self.crop_h, self.crop_w = new_size
    self.h, self.w = original_size

  def __call__(self, image):
    return image[int((self.h - self.crop_h) // 2):int((self.h + self.crop_h) // 2), int((self.w - self.crop_w) // 2):int((self.w + self.crop_w) // 2)]


class StaticResize(object):
  
  def __init__(self, original_size, new_size, data_type='rgb'):
    self.horizontal_scale = 1.0 * new_size[1] / original_size[1]
    self.new_size = tuple(reversed(new_size))
    self.data_type = data_type

  def __call__(self, image):
    channels = image.shape[2]

    if self.data_type == 'rgb':
      image = cv2.resize(image, self.new_size)
    elif self.data_type == 'disparity':
      image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_NEAREST)
      image = image * self.horizontal_scale
      image = image.reshape(image.shape + (channels,))
    else:
      raise ValueError('Unsupported data_type: ' + self.data_type)

    return image.astype(np.float32)


class StaticPadTo64x(object):

  def __init__(self, original_size):
    height, width = original_size
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

