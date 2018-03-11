from __future__ import absolute_import

import cv2

from .static_transforms import *


class StaticStereoParallelTransform:

  def __init__(self):
    raise NotImplementedError()

  def __call__(self, left_image, right_image, truth_image):
    return self.transform(left_image), self.transform(right_image), self.transform(truth_image)


class StaticStereoRandomCrop(StaticStereoParallelTransform):

  def __init__(self, original_size, new_size):
    self.transform = StaticRandomCrop(original_size, new_size)


class StaticStereoCenterCrop(StaticStereoParallelTransform):
  
  def __init__(self, original_size, new_size):
    self.transform = StaticCenterCrop(original_size, new_size)


class StaticStereoPadTo64x(StaticStereoParallelTransform):

  def __init__(self, original_size):
    self.transform = StaticPadTo64x(original_size)


class StaticStereoResize:

  def __init__(self, original_size, new_size):
    self.data_transform = StaticResize(original_size, new_size)
    self.label_transform = StaticResize(original_size, new_size, data_type='disparity')

  def __call__(self, left_image, right_image, truth_image):
    return self.data_transform(left_image), self.data_transform(right_image), self.label_transform(truth_image)


class StaticStereoDataAugment:

  def __init__(self, original_size, new_size, scale_factor_small=0.7, scale_factor_large=1.3):
    scale_factor = max(1.0 * new_size[0] / original_size[0], 1.0 * new_size[1] / original_size[1])
    
    resized_size = new_size
    if scale_factor < scale_factor_large:
      resize_scale = random.uniform(max(scale_factor, scale_factor_small), scale_factor_large)
      resized_size = (max(new_size[0], int(resize_scale * original_size[0])), max(new_size[1], int(resize_scale * original_size[1])))

    self.resize = StaticStereoResize(original_size, resized_size)
    self.random_crop = StaticStereoRandomCrop(resized_size, new_size)

  def __call__(self, left_image, right_image, truth_image):
    left_image, right_image, truth_image = self.resize(left_image, right_image, truth_image)
    left_image, right_image, truth_image = self.random_crop(left_image, right_image, truth_image)
    return left_image, right_image, truth_image

