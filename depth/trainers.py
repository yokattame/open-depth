from __future__ import print_function, absolute_import

import time

from torch.autograd import Variable

from .utils.meters import AverageMeter


class BaseTrainer(object):

  def __init__(self, model, criterion, args):
    super(BaseTrainer, self).__init__()
    self.model = model
    self.criterion = criterion
    self.args = args

  def _adjust_lr(self, epoch, optimizer):
    lr = self.args.lr * (0.1 ** (epoch // self.args.step_size))
    for g in optimizer.param_groups:
      g['lr'] = lr * g.get('lr_mult', 1)


  def train(self, epoch, data_loader, optimizer, print_freq=1):
    self.model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epes = AverageMeter()

    if self.args.optimizer == 'SGD':
      self._adjust_lr(epoch, optimizer)

    end = time.time()
    for i, inputs in enumerate(data_loader):
      # measure data loading time
      data_time.update(time.time() - end)

      inputs, targets = self._parse_data(inputs)
      loss, epe = self._forward(inputs, targets)

      losses.update(loss.data[0], self.args.batch_size)
      epes.update(epe.data[0], self.args.batch_size)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batch_time.update(time.time() - end)
      end = time.time()

      if (i + 1) % print_freq == 0:
        print('Epoch: [{}][{}/{}]\t'
            'Time {:.3f} ({:.3f})\t'
            'Data {:.3f} ({:.3f})\t'
            'Loss {:.3f} ({:.3f})\t'
            'EPE {:.3f} ({:.3f})\t'
            .format(epoch, i + 1, len(data_loader),
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.avg,
                    losses.val, losses.avg,
                    epes.val, epes.avg))

  def _parse_data(self, inputs):
    raise NotImplementedError

  def _forward(self, inputs, targets):
    raise NotImplementedError


class Trainer(BaseTrainer):

  def _parse_data(self, inputs):
    left_image, right_image, disp_image, mask = inputs
    
    inputs = (Variable(left_image.cuda(async=True)), Variable(right_image.cuda(async=True)))
    targets = (Variable(disp_image.cuda(async=True)), Variable(mask.cuda(async=True)))
    return inputs, targets

  def _forward(self, inputs, targets):
    outputs = self.model(inputs)
    loss, epe = self.criterion(outputs, targets)
    return loss, epe

