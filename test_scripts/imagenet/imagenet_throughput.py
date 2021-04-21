from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
# import tensorboardX
import os
import math
from tqdm import tqdm
from logger import get_logger #, log_time, sync_e
import json
import time
import numpy as np

def sync_e():
    e = torch.cuda.Event()
    e.record()
    e.synchronize()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet152',
                    help='model to benchmark')
parser.add_argument('--train-dir', default='/data',
                    help='path to training data')
parser.add_argument('--val-dir', default='/data',
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=2,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--log-loc', type=str, default='log.txt')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

allreduce_batch_size = args.batch_size * args.batches_per_allreduce

torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    # torch.cuda.set_device(0)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
# for try_epoch in range(args.epochs, 0, -1):
#     if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
#         resume_from_epoch = try_epoch
#         break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.

# Horovod: print logs on the first worker.
verbose = 1

# Horovod: write TensorBoard logs on first worker.
# log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
log_writer = None

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=1, rank=0, shuffle=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

# Set up standard model.
model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
# compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
# optimizer = hvd.DistributedOptimizer(
#     optimizer, named_parameters=model.named_parameters(),
#     compression=compression,
#     backward_passes_per_step=args.batches_per_allreduce)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
# hvd.broadcast_parameters(model.state_dict(), root_rank=0)
# hvd.broadcast_optimizer_state(optimizer, root_rank=0)

f = open(args.log_loc, 'w')
now = time.time()
print("#time throughput", file=f)

def train(epoch):
    global now
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:

        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            # number of batchs limit
            if batch_idx >= 20000:
                return

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                sync_e()
                output = model(data_batch)

                _acc = accuracy(output, target_batch)
                train_accuracy.update(_acc)
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
                sync_e()
                last = now
                now = time.time()
                duration = now - last
                print('%lf %lf' % (now, args.batch_size / duration), file=f)

            # Gradient is applied across all ranks
            optimizer.step()

            t.set_postfix({'loss': train_loss.avg.item(),
                        'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)




# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1.
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.).cuda()
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)

f.close()
print('Finish!')