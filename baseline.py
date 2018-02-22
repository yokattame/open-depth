from __future__ import print_function, absolute_import
import os.path as osp
import argparse
import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from depth import datasets
from depth import networks
from depth import loss
from depth import optimizers
from depth.utils.data import dataset as D
from depth.trainers import Trainer
from depth.evaluators import Evaluator
from depth.utils.logging import Logger
from depth.utils.serialization import save_checkpoint, load_checkpoint


def main(args):
  #cudnn.benchmark = True
  torch.manual_seed(args.seed)
  sys.stdout=Logger(osp.join(args.logs_dir, 'log.txt'))
  
  datasets.create(args.dataset, args.dataset_root)
  train_list = []
  with open(osp.join('data_lists', args.dataset + '.txt'), 'r') as input_file:
    for line in input_file:
      train_list.append(line)
  train_list = np.asarray(train_list)
  if args.validation > 0:
    train_list = []
    with open(osp.join('data_lists', args.dataset + '_train.txt'), 'r') as input_file:
      for line in input_file:
        train_list.append(line)
    train_list = np.asarray(train_list)

    val_list = []
    with open(osp.join('data_lists', args.dataset + '_val.txt'), 'r') as input_file:
      for line in input_file:
        val_list.append(line)
    val_list = np.asarray(val_list)

    print('validation size: {}'.format(val_list.shape[0]))

  print('train size: {}'.format(train_list.shape[0]))
  
  train_transformer = transforms.Compose([
      transforms.ToTensor(),
  ])
  train_dataset = D.DepthDataset(train_list, (args.input_height, args.input_width), train_transformer, resize_method='Random Crop')
  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      num_workers=args.number_workers,
      shuffle=True,
      pin_memory=True,
      drop_last=True)

  if args.validation > 0:
    val_transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_dataset = D.DepthDataset(val_list, (args.input_height, args.input_width), val_transformer, resize_method='Center Crop')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.number_workers,
        shuffle=False,
        pin_memory=True)

  model = networks.FlowNetDC(normalization='Example')
  model = nn.DataParallel(model).cuda()

  if args.loss == 'L1Valid':
    criterion = loss.L1ValidLoss().cuda()
  elif args.loss == 'MultiScaleValid':
    criterion = loss.MultiScaleValidLoss().cuda()
  else:
    raise ValueError('Undefined loss: ' + args.loss)

  trainer = Trainer(model, criterion, args)
  evaluator = Evaluator(model, criterion, args)

  param_groups = model.parameters()
  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
  elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
  elif args.optimizer == 'SGDC':
    optimizer = optimizers.SGDC(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True, gradient_threshold=args.gradient_threshold)
  else:
    raise ValueError('Undefined optimizer: ' + args.optimizer)
  
  best_EPE = 1e9
  if args.resume:
    checkpoint = load_checkpoint(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_EPE = checkpoint['best_EPE']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> Start epoch: {:3d}\tbest_EPE: {:5.3}'
          .format(args.start_epoch, best_EPE))

  if args.evaluate:
    EPE = evaluator.evaluate(val_loader)
    print('EPE: {:5.3}'.format(EPE))
    return

  for epoch in range(args.start_epoch, args.epochs):
    trainer.train(epoch, train_loader, optimizer, print_freq=1)
    if (epoch + 1) % args.validation == 0:
      EPE = evaluator.evaluate(val_loader)
      is_best = EPE < best_EPE
      best_EPE = min(EPE, best_EPE)
      save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_EPE' : best_EPE,
          'optimizer': optimizer.state_dict(),
      }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
      
      print('\n * Finished epoch {:3d}\tEPE: {:5.3}\tbest: {:5.3}{}\n'.
            format(epoch, EPE, best_EPE, ' *' if is_best else ''))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset', type=str, default='KittiStereo')
  parser.add_argument('--dataset_root', type=str, default='../data/KittiStereo')
  parser.add_argument('--resume', type=str, default='')
  parser.add_argument('--evaluate', action='store_true')
  parser.add_argument('--validation', type=int, default=1)
  parser.add_argument('--input_height', type=int, default=320)
  parser.add_argument('--input_width', type=int, default=1216)
  parser.add_argument('--batch_size', type=int, default=20)
  parser.add_argument('--number_workers', type=int, default=4)
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=45)
  parser.add_argument('--logs_dir', type=str, default='logs/baseline')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--loss', type=str, default='L1Valid')
  parser.add_argument('--optimizer', type=str, default='SGD')
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--step_size', type=int, default=15)
  parser.add_argument('--beta1', type=float, default=0.9)
  parser.add_argument('--beta2', type=float, default=0.999)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight_decay', type=float, default=5e-4)
  parser.add_argument('--gradient_threshold', type=float, default=None)

  main(parser.parse_args())

