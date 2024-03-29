from __future__ import print_function

import argparse
import numpy as np
from transformers import BertTokenizer, BertForPreTraining, BertConfig
from transformers import AdamW
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import os
from compression import compressors
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'
import hv_distributed_optimizer as hvd
import timeit
from profiling import benchmark
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='bert',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=8,
                    help='input batch size')
parser.add_argument('--sentence-len', type=int, default=128,
                    help='input sentence len')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=5,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--mgwfbp', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')
parser.add_argument('--threshold', type=int, default=536870912, help='Set threshold if mgwfbp is False')
parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')
parser.add_argument('--compressor', type=str, default='none', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=1, help='Density for sparsification')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['HOROVOD_NUM_NCCL_STREAMS'] = str(args.nstreams)

hvd.init()
if args.cuda:
    # Horovod: pin GPU to local rank.
    #print('local rank: ', hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

DATAPATH='/datasets/shshi'
pretrained_path='%s/pretrained'%DATAPATH

if args.model == 'bert_base':
    config = BertConfig.from_json_file('bert_base_config.json')
else:
    config = BertConfig.from_json_file('bert_config.json')
# Padding for divisibility by 8
if config.vocab_size % 8 != 0:
    config.vocab_size += 8 - (config.vocab_size % 8)

vocab_size=config.vocab_size
#tokenizer = BertTokenizer.from_pretrained(pretrained_path)
#model = BertForPreTraining.from_pretrained(pretrained_path)
model = BertForPreTraining(config)

if args.cuda:
    model.cuda()

max_len = args.sentence_len
batch_size = args.batch_size
input_ids = (torch.rand(batch_size, max_len) * 2000).long()
attention_masks = torch.rand(batch_size, max_len).long()
token_type_ids = torch.rand(batch_size, max_len).long()
position_ids = (torch.rand(batch_size, max_len) * 10).long()
next_sentence_label = torch.rand(batch_size, 1).long()
masked_lm_labels = torch.rand(batch_size, max_len).long()
batch = (input_ids, attention_masks, token_type_ids, position_ids, next_sentence_label, masked_lm_labels)
if args.cuda:
    batch = tuple(item.cuda() for item in batch)
input_ids, attention_masks, token_type_ids, position_ids, next_sentence_label, masked_lm_labels = batch

class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss
criterion = BertPretrainingCriterion(vocab_size)

if args.mgwfbp:
    seq_layernames, layerwise_times, _ = benchmark(model, batch, criterion, task='bert')
    layerwise_times = comm.bcast(layerwise_times, root=0)
else:
    seq_layernames, layerwise_times = None, None



optimizer = AdamW(model.parameters(),
        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        )
#optimizer = optim.SGD(model.parameters(), lr=2e-5)

#compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
# Horovod: wrap optimizer with DistributedOptimizer.
#optimizer = hvd.DistributedOptimizer(optimizer,
#                                     named_parameters=model.named_parameters(),
#                                     compression=compression,
#                                     op=hvd.Average)
if hvd.size() > 1:
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compressors[args.compressor](), is_sparse=args.density<1, density=args.density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=args.threshold, writer=None, gradient_path='./', momentum_correction=False, fp16=args.fp16, mgwfbp=args.mgwfbp, rdma=args.rdma, asc=args.asc)
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def benchmark_step():
    optimizer.zero_grad()
    #loss, prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks, masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_label)
    prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)
    loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_label)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

benchmark_step()

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

log('BERT Large Pretraining, Sentence len: %d' % max_len)
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    sen_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f sentences/sec per GPU' % (x, sen_sec))
    img_secs.append(sen_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Sen/sec per %s: %.1f +-%.1f' % ('GPU', img_sec_mean, img_sec_conf))
log('Total sen/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), 'GPU', hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
