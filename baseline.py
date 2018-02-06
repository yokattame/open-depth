from __future__ import print_function, absolute_import
import os.path as osp
import argparse
import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import *

from depth import datasets
from depth import networks
from depth import loss
from depth.utils.data import dataset as D
#from depth.utils.data import transforms as T
from depth.trainers import Trainer
from depth.evaluators import Evaluator
from depth.utils.logging import Logger
from depth.utils.serialization import save_checkpoint


def main(args):
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  sys.stdout=Logger(osp.join(args.logs_dir, 'log.txt'))
  
  datasets.create(args.dataset, args.dataset_root)
  train_list = []
  with open(osp.join('data_lists', args.dataset + '.txt'), 'r') as input_file:
    for line in input_file:
      train_list.append(line)
  train_list = np.asarray(train_list)
  if args.validation > 0:
    num = train_list.shape[0]
    num_val = int(round(num * 0.2))
    np.random.shuffle(train_list)
    val_list = train_list[:num_val]
    train_list = train_list[num_val:]
    assert(num == val_list.shape[0] + train_list.shape[0])
    print('validation size: {}'.format(len(val_list)))

  print('train size: {}'.format(len(train_list)))
  
  normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  train_transformer = transforms.Compose([
      transforms.ToTensor(),
      normalizer,
  ])
  train_dataset = D.DepthDataset(train_list, (args.input_height, args.input_width), train_transformer, random_crop=True)
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
        normalizer,
    ])
    val_dataset = D.DepthDataset(val_list, (args.input_height, args.input_width), val_transformer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.number_workers,
        shuffle=False,
        pin_memory=True)

  model = networks.FlowNetDC()
  model = nn.DataParallel(model).cuda()


  criterion = loss.L1V().cuda()

  trainer = Trainer(model, criterion, args)
  evaluator = Evaluator(model, criterion, args)

  param_groups = model.parameters()
  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(args.beta1, 0.999))
  else:
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
  
  best_EPE = 1e9
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
  parser.add_argument('--dataset_root', type=str, default='../kitti/training')
  parser.add_argument('--validation', type=int, default=1)
  parser.add_argument('--input_height', type=int, default=320)
  parser.add_argument('--input_width', type=int, default=1216)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--number_workers', type=int, default=4)
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--logs_dir', type=str, default='logs')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--optimizer', type=str, default='Adam')
  parser.add_argument('--lr', type=float, default=0.0001)
  parser.add_argument('--step_size', type=int, default=50)
  parser.add_argument('--beta1', type=float, default=0.5)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight_decay', type=float, default=5e-4)

  main(parser.parse_args())
