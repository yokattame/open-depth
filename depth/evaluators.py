from __future__ import print_function, absolute_import

import time

from torch.autograd import Variable

from .utils.meters import AverageMeter


class Evaluator(object):
  
  def __init__(self, model, criterion, args):
    super(Evaluator, self).__init__()
    self.model = model
    self.criterion = criterion
    self.args = args

  def evaluate(self, data_loader, print_freq=1):
    self.model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epes = AverageMeter()

    end = time.time()
    for i, inputs in enumerate(data_loader):
      data_time.update(time.time() - end)

      inputs, targets = self._parse_data(inputs)
      loss, epe = self._forward(inputs, targets)

      losses.update(loss.data[0], self.args.batch_size)
      epes.update(epe.data[0], self.args.batch_size)

      batch_time.update(time.time() - end)

      if (i + 1) % print_freq == 0:
        print('Extract Features: [{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              'EPE {:.3f} ({:.3f})\t'
              .format(i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg,
                      epes.val, epes.avg))
      end = time.time()

    return epes.avg
  
  def _parse_data(self, inputs):
    left_image, right_image, disp_image, mask = inputs
    
    inputs = (Variable(left_image.cuda(async=True), volatile=True), Variable(right_image.cuda(async=True), volatile=True))
    targets = (Variable(disp_image.cuda(async=True), volatile=True), Variable(mask.cuda(async=True), volatile=True))
    return inputs, targets

  def _forward(self, inputs, targets):
    outputs = self.model(inputs)
    loss, epe = self.criterion(outputs, targets)
    return loss, epe


