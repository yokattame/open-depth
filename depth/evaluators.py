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
    metrics = dict()
    for metric_key in self.criterion.get_metric_keys():
      metrics[metric_key] = AverageMeter()

    end = time.time()
    for i, inputs in enumerate(data_loader):
      data_time.update(time.time() - end)

      inputs, targets = self._parse_data(inputs)
      loss, metric_values = self._forward(inputs, targets)

      losses.update(loss.data[0], self.args.batch_size)
      for metric in metrics:
        metrics[metric].update(metric_values[metric].data[0], self.args.batch_size)

      batch_time.update(time.time() - end)

      if (i + 1) % print_freq == 0:
        print('Extract Features: [{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              .format(i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg),
              end='')
      for metric_key, metric_value in metrics.items():
        print(metric_key, '{:.3f} ({:.3f})\t'.format(metric_value.val, metric_value.avg), end='')
      print()
      end = time.time()

    return metrics
  
  def _parse_data(self, inputs):
    left_image, right_image, disp_image, mask = inputs
    
    inputs = (Variable(left_image.cuda(async=True), volatile=True), Variable(right_image.cuda(async=True), volatile=True))
    targets = (Variable(disp_image.cuda(async=True), volatile=True), Variable(mask.cuda(async=True), volatile=True))
    return inputs, targets

  def _forward(self, inputs, targets):
    outputs = self.model(inputs)
    loss, metrics = self.criterion(outputs, targets)
    return loss, metrics

