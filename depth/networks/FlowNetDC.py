from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .FlowNetC import FlowNetC

from .submodules import *
'Parameter count = 162,518,834'

RGB_MAX = 255
RGB_SCALE = 1


class FlowNetDC(FlowNetC):

    def __init__(self, normalization='Example', batchNorm=False, div_flow=1):
        super(FlowNetDC, self).__init__(batchNorm=batchNorm, div_flow=1)
        self.normalization = normalization

    def forward(self, inputs):
        x1, x2 = inputs

        if self.normalization == 'Example':
          rgb_mean = (x1.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True) + x2.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)) / 2
          rgb_scale = RGB_SCALE
        elif self.normalization == 'Batch':
          rgb_mean = (x1.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True) + x2.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True)) / 2
          rgb_scale = RGB_SCALE
        elif self.normalization == 'ImageNet':
          rgb_mean = torch.autograd.Variable((torch.Tensor([123.680, 116.779, 103.939]) / RGB_MAX).view(1, -1, 1, 1)).cuda()
          rgb_scale = troch.autograd.Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)).cuda()
        else:
          raise ValueError('Undefined normalization method: ' + self.normalization)

        x1 = (x1 - rgb_mean) / rgb_scale
        x2 = (x2 - rgb_mean) / rgb_scale

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        #print(out_conv1a.size())
        out_conv2a = self.conv2(out_conv1a)
        #print(out_conv2a.size())
        out_conv3a = self.conv3(out_conv2a)
        #print(out_conv3a.size())

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        #print(out_conv3a.size())
        out_conv_redir = self.conv_redir(out_conv3a)
        #print(out_conv_redir.size())

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        #print(out_conv4.size(), out_conv6.size())
        depth6       = self.predict_depth6(out_conv6)
        depth6_up    = self.upsampled_depth6_to_5(depth6)
        out_deconv5 = self.deconv5(out_conv6)

        #print(out_conv5.size(), out_deconv5.size(), depth6_up.size())
        concat5 = torch.cat((out_conv5,out_deconv5,depth6_up),1)

        depth5       = self.predict_depth5(concat5)
        depth5_up    = self.upsampled_depth5_to_4(depth5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,depth5_up),1)

        depth4       = self.predict_depth4(concat4)
        depth4_up    = self.upsampled_depth4_to_3(depth4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,depth4_up),1)

        depth3       = self.predict_depth3(concat3)
        depth3_up    = self.upsampled_depth3_to_2(depth3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,depth3_up),1)

        depth2 = self.predict_depth2(concat2)
        
        upsampled_depth2 = self.upsample2(depth2 * self.div_flow)
        upsampled_depth3 = self.upsample3(depth3 * self.div_flow)
        upsampled_depth4 = self.upsample4(depth4 * self.div_flow)
        upsampled_depth5 = self.upsample5(depth5 * self.div_flow)
        upsampled_depth6 = self.upsample6(depth6 * self.div_flow)
        
        return upsampled_depth2, upsampled_depth3, upsampled_depth4, upsampled_depth5, upsampled_depth6

