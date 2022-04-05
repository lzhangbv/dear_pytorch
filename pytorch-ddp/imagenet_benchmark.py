from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
import timeit
import numpy as np
import os
import time

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--fp16-allreduce', action='store_true', default=False,
#                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=5,
                    help='number of benchmark iterations')
parser.add_argument('--use-zero', type=int, default=0,
                    help='use ZeroRedundancy Optimizer')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# local_rank: (1) parse argument as folows in torch.distributed.launch; (2) read from environment in torch.distributed.run, i.e. local_rank=int(os.environ['LOCAL_RANK'])
parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank for distributed training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dist.init_process_group(backend='nccl', init_method='env://')
args.local_rank = int(os.environ['LOCAL_RANK'])

if args.cuda:
    # pin GPU to local rank.
    torch.cuda.set_device(args.local_rank)

cudnn.benchmark = True

# Set up standard model.
if args.model == 'inceptionv4':
    from inceptionv4 import inceptionv4
    model = inceptionv4()
else:
    model = getattr(models, args.model)()

if args.cuda:
    model.cuda()

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

if args.use_zero:
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optim.SGD, lr=0.01)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.01)


# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()


def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def log(s, nl=True):
    if dist.get_rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, dist.get_world_size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (dist.get_world_size(), device, dist.get_world_size() * img_sec_mean, dist.get_world_size() * img_sec_conf))

#time.sleep(3.)
