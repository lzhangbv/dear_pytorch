# Copyright 2020 HKBU. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import collections
import numpy as np

from comm_core import rank, size, init
from comm_core import Communicator
from comm_core import init as comm_init
from sklearn.linear_model import LinearRegression

from . import utils
from .profiling import CommunicationProfiler
import tcmm


comm_init()
comm = None
def init():
    global comm
    comm = Communicator(1)


ADAPTIVE_SPARSE = False
DEBUG = False


def gtopk_sparse_recursive_allreduce(comm, sparse_tensor, values, indexes):
    num_workers = size()
    r = rank()
    num_round = int(np.log2(num_workers))
    peer_masks = np.zeros(num_workers, dtype=np.int64)

    original_indexes = indexes
    sparse_tensor = torch.zeros_like(sparse_tensor)
    #sparse_tensor.fill_(0.0)

    k = values.numel()

    send_values = values
    send_indexes = indexes
    recv_values = torch.zeros_like(values)
    recv_indexes = torch.zeros_like(indexes)

    for i in range(num_round):
        peer_distance = 2**i
        for j in range(num_workers):
            local_rank = j % (2 * peer_distance)
            if local_rank < peer_distance:
                peer_masks[j] = 1
            else:
                peer_masks[j] = -1
        peer = peer_masks[r] * peer_distance + r
        comm.sendrecv(send_values, recv_values, peer)
        comm.sendrecv(send_indexes, recv_indexes, peer)
        comm.synchronize()

        if r < peer:
            first_indexes = send_indexes
            first_values = send_values
            second_indexes = recv_indexes
            second_values = recv_values
        else:
            first_indexes = recv_indexes
            first_values = recv_values
            second_indexes = send_indexes
            second_values = send_values
        sparse_tensor.scatter_(0, first_indexes, first_values)
        sparse_tensor.scatter_add_(0, second_indexes, second_values)

        #_, indexes = torch.topk(sparse_tensor.abs(), k=k, sorted=False)

        _, indexes = tcmm.f_topk(sparse_tensor.abs(), k)
        indexes = indexes.long()

        send_values = sparse_tensor[indexes]
        send_indexes = indexes
        #print('%d: [%d<->%d], %s, send: %s,%s' % (i, r, peer, sparse_tensor, send_values, send_indexes))
        sparse_tensor.fill_(0.0)
       
    included_indexes = indexes
    #cv, c1, c2 = np.intersect1d(original_indexes.cpu().numpy(), included_indexes.cpu().numpy(), return_indices=True)
    #included_indexes = torch.tensor(c1, device=original_indexes.device)
    return send_values, send_indexes, included_indexes 

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression,
                 is_sparse=False,
                 density=0.001,
                 seq_layernames=None,
                 layerwise_times=None,
                 norm_clip=None,
                 threshold=0,
                 writer=None,
                 gradient_path=None,
                 momentum_correction=False,
                 fp16=False,
                 mgwfbp=False,
                 rdma=False,
                 multi_job_scheduling=False):
        r"""Distributed optimizer with w/o or w/ MG-WFBP and/or Gradient Compression.

        Args:
            params (nn.Module): Torch optimizer parameters, e.g., optimizer.param_groups.
            named_parameters (nn.Module): Torch model parameters, e.g., model.named_parameters().
            compression (TopKCompressor): Object of TopKCompressor.
            is_sparse (bool): Use gradient sparsification.
            seq_layernames (list): Layer names from the 1st layer to the final layer.
            layerwise_times (list): Elapsed time per layer.
            norm_clip (float): Norm clip value for gradients.
            threshold (int): Threshold for merging gradients (tensor fusion).
            writer (Writer): TensorboardX object.
            gradient_path (str): File path for storing gradients.
            momentum_correction (bool): Momentum correction.
            fp16 (bool): Use mixed precision training.
            mgwfbp (bool): Use MG-WFBP.
            rdma (bool): Use RDMA alpha and beta.
            multi_job_scheduling (bool): Enable multi-job scheduling.

        Attributes:
            _comm (comm_core.Communicator): Communication utilities.
            _compression (TopKCompressor): The gradient compressor.
            _sparse (bool): Enable gradient sparsification.
            _multi_job_scheduling (bool): Enable multi-job scheduling.
            _density (float): The density for gradient sparsification.
            _seq_layernames (list): Layer names from the 1st layer to the final layer.
            _layerwise_times (list): Elapsed time per layer.
            _original_layerwise_times_kv (dict): TODO.
            _norm_clip (float): Norm clip value for gradients.
            _threshold (int): Threshold for merging gradients (tensor fusion).
            _writer (Writer): TensorboardX object.
            _gradient_path (str): File path for storing gradients.
            _fp16 (bool): Enable mixed-precision training.
            _mgwfbp (bool): Use MG-WFBP.
            _rdma (bool): Use RDMA alpha and beta.
            _sizes (list): Layer-wise gradient size.
            _final_name (str): Gradient tensor name for the single-layer communication.
            _final_tensor (Tensor): Gradient tensor for the single-layer communication.
            _compression_timers (dict): Layer-wise compression timer.
            _allreduce_timers (dict): Layer-wise all-reduce timer.
            _update_timers (dict): Layer-wise update timer.
            train_epoch (int): Current training epoch. 
            train_iter (int): Current training iteration. 
            momentum_correction (bool): Enable momentum correction.
            _layerwise_compressors (dict): Layer-wise compressors, key is tensor name, value is compressor.
            _dynamic_densities (list): Setting different densities for epochs.
            local (bool): Local update. True indicates not to aggregation.
        """
        super(self.__class__, self).__init__(params)


        self._compression = compression
        self._sparse = is_sparse
        self._multi_job_scheduling = multi_job_scheduling
        self._density = density
        self._profiling = False
        self._seq_layernames = seq_layernames
        self._layerwise_times = layerwise_times
        self._original_layerwise_times_kv = None
        self._norm_clip = norm_clip
        self._threshold = threshold
        self._writer = writer
        self._gradient_path = gradient_path
        self._fp16 = fp16
        self._mgwfbp = mgwfbp
        self._rdma = rdma
        self._sizes = None
        self._final_name = None
        self._final_tensor = None
        self.alpha = None
        self.beta = None
        if self._layerwise_times is not None and self._seq_layernames is not None:
            self._original_layerwise_times_kv = dict(zip(self._seq_layernames, \
                self._layerwise_times))
        self._compression_timers = {} # compression
        self._allreduce_timers = {} # allreduce times
        self._update_times = {} # allreduce times
        self.train_epoch = 0
        self.train_iter = 0
        self.momentum_correction = momentum_correction
        if density < 1:
            self._layerwise_compressors = {}
            #self._dynamic_densities = None
            self._dynamic_densities = [0.25, 0.0625, 0.015625, 0.004, 0.001] # the setting used in DGC
        else:
            self._dynamic_densities = None
            self._layerwise_compressors = None

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._named_parameters = {k: v for k, v
                                  in named_parameters}
        if self._seq_layernames is not None:
            self._sequential_keys = self._seq_layernames
        else:
            self._sequential_keys = [k for k, v in named_parameters]
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys]
        model_size = sum(sizes)
        if self._threshold == -1:
            self._threshold = model_size

        self.size_commtime_dict = None
        if self._mgwfbp and self._layerwise_times is None:
            self._benchmark_communication()

        self._debug_seq_keys = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self._generate_merged_parameters()

        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self.local = False
        self._hook_checked_idx = 0
        if size() > 1:
            self._register_hooks()

        #logger.info('layerwise compressors: %s', self._layerwise_compressors)

    def _benchmark_communication(self):
        """Benchmark communication performance to achieve alpha and beta.
        """
        #logger.info('Benchmarking communication performance...')
        comm_profiler = CommunicationProfiler(comm.allReduce, comm.synchronize)
        sizes, times = comm_profiler.benchmark(num_iters=10)
        def _fit_linear_function(x, y):
            X = np.array(x).reshape((-1, 1)) * 4
            Y = np.array(y)
            model = LinearRegression()
            model.fit(X, Y)
            alpha = model.intercept_
            beta = model.coef_[0]
            #A = np.vstack([X, np.ones(len(X))]).T
            #beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
            return alpha, beta
        alpha, beta = _fit_linear_function(sizes, times)
        self.alpha = alpha
        self.beta = beta
        alpha_tensor = torch.ones(1) * alpha
        beta_tensor = torch.ones(1) * beta
        alpha_tensor = broadcast(alpha_tensor, root_rank=0)
        beta_tensor = broadcast(beta_tensor, root_rank=0)
        if rank() != 0:
            self.alpha = float(alpha_tensor[0])
            self.beta = float(beta_tensor[0])

    def _benchmark_communication2(self):
        sizes = [self._named_parameters[k].data.numel() 
                 for k in self._sequential_keys][::-1] # reverse from L to 1
        all_combined_sizes = []
        for i in range(len(sizes)):
            s = sizes[i]
            all_combined_sizes.append(s)
            for j in range(i+1, len(sizes)):
                s += sizes[j]
                all_combined_sizes.append(s)
        comm_profiler = CommunicationProfiler(comm.allReduce, comm.synchronize, all_combined_sizes)
        sizes, times = comm_profiler.benchmark(num_iters=10)
        size_commtime_dict = {}
        for s, t in zip(sizes, times):
            if s not in size_commtime_dict:
                size_commtime_dict[s] = t
            else:
                if t > size_commtime_dict[s]:
                    size_commtime_dict[s] = t
        self.size_commtime_dict = size_commtime_dict

    def _register_hooks(self):
        r"""Register hooks.
        """
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _generate_groups_with_threshold(self, threshold):
        r"""Tensor fusion with the threshold.

        Args: 
            threshold (int): # of parameters for one single group.

        Returns: 
            groups (list): Groups with tensor fusion. One item is one or several layers.
            key_groupidx_maps (dict): Key: tensor name, Value: group index.
        """
        sizes = [self._named_parameters[k].data.numel() 
                 for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:
            numel = self._named_parameters[k].data.numel()
            sub_size += numel
            key_groupidx_maps[k] = idx
            if sub_size < threshold:
                group.append(k)
            else:
                idx += 1
                group.append(k)
                groups.append(group)
                group = []
                sub_size = 0
        if len(group) > 0:
            groups.append(group)
        if rank() == 0:
            print('# of groups: ', len(groups), ', # of layers: ', len(self._sequential_keys))
        return groups, key_groupidx_maps

    def get_current_density(self, name=None):
        """Get density for current iteration.
        Args:
            name (str): Tensor name.

        Returns:
            density (float): Density for current tensor.
        """
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        if name is not None and self._layerwise_compressors is not None:
            if name not in self._layerwise_compressors:
                errstr = 'compressor density not found at layer: %s' % name
                logger.error(errstr)
                raise Exception(errstr)
            ld = self._layerwise_compressors[name]
            density = max(ld, density)
        return density

    def _generate_groups_mgwfbp(self):
        r"""The MG-WFBP algorithm.
            Ref: S. Shi et al., MG-WFBP: Efficient data communication for distributed synchronous SGD algorithms, INFOCOM 2019.
        """
        num_of_workers = size()
        p_alpha_beta_56Gbps = {
                64: (0.00080632079996292579, 1.8*3.2713239529771973e-10),
                32: (0.00040632079996292579, 1.5*3.2713239529771973e-10),
                16: (0.00023583677659915685*3, 4.0594787739537565e-10),
                8: (9.75367204301171e-05, 3.0568230536676206e-10),
                4: (4.204298980348825e-05, 2.0589360830118177e-10),
                2: (2.554691138304671e-06, 9.837548167872609e-11)
                }
        p_alpha_beta_10Gbps = {
                64: (0.0023476410788581382*3, 9.643300782166769e-10),
                32: (0.0013476410788581382*3, 8.643300782166769e-10),
                16: (0.0009080981007148093, 7.395651186836712e-10),
                8: (0.0005230272768511732, 8.570746975492128e-10),
                4: (4.204298980348825e-05, 2.0589360830118177e-10),
                2: (2.554691138304671e-06, 9.837548167872609e-11)
                }
        if self.alpha is not None:
            alpha, beta = self.alpha, self.beta
        else:
            if self._rdma:
                alpha, beta = p_alpha_beta_56Gbps[num_of_workers]
            else:
                alpha, beta = p_alpha_beta_10Gbps[num_of_workers]
        nbytes = 2 if self._fp16 else 4
        def __calculate_comm_start(tc, tb, taob, L):
            taoc = [0] * L 
            taoc[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taob[l] + tb[l])
            return taoc
        def __merge(taob, tc, p, l):
            tc[l] = 0
            p[l-1] = p[l-1]+p[l]
            p[l] = 0
            if self.size_commtime_dict is not None:
                tc[l-1] = self.size_commtime_dict[l-1]
            else:
                tc[l-1] = utils.predict_allreduce_time_with_size(alpha,\
                    beta, p[l-1]*nbytes, num_of_workers)
        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        if not utils.check_unique(seq_layernames):
            raise ValueError
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        if self.size_commtime_dict is not None:
            tc = [self.size_commtime_dict[s] for s in sizes]
        else:
            tc = [utils.predict_allreduce_time_with_size(alpha, beta, s*nbytes, num_of_workers) 
                  for s in sizes]
        tb = list(self._layerwise_times)
        taob = [0]*L
        for l in range(0,L-1)[::-1]:
            taob[l] = taob[l+1] + tb[l+1]
        taoc = __calculate_comm_start(tc, tb, taob, L)
        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        for l in range(1, L)[::-1]:
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l-1] + tb[l-1]
            merged=False
            if current_taob < taoc[l]+tc[l]:
                if taoc[l] > current_taob:
                    __merge(taob, tc, p, l)
                    taoc = __calculate_comm_start(tc, tb, taob, L)
                    merged=True
                else:
                    t_wait = current_taob - taoc[l]
                    t_saved = alpha
                    if t_wait < t_saved:
                        __merge(taob, tc, p, l)
                        taoc = __calculate_comm_start(tc, tb, taob, L)
                        merged=True
            #if not merged and (key.find('bn') >= 0 or key.find('bias') >= 0):
            if not merged and p[l] < 8192: 
                __merge(taob, tc, p, l)
                taoc = __calculate_comm_start(tc, tb, taob, L)
                merged=True
            if not merged:
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)

        if rank() == 0:
            print('Merged sizes: ', p[::-1])
            print('# of parameters: ', np.sum(p[::-1]))

        return groups, key_groupidx_maps

    def _generate_groups_mgs(self):
        r"""The MGS-SGD algorithm.
            Ref: S. Shi et al., Communication-efficient distributed deep learning with merged gradient sparsification on GPUs, INFOCOM 2020.
        """
        P = size() # number of wokers

        def __calculate_sparse_and_backward_start(tb, sizes, L, start=0):
            taos = [start] * L 
            ts = [utils.topk_perf_model(s) for s in sizes]
            taob = [start] * L 
            taob[L-1] = start 
            taos[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taob[l] = taos[l+1] + ts[l+1]
                taos[l] = taob[l] + tb[l]
            return taob, taos, ts

        def __calculate_comm_start(ts, taos, sizes, L):
            taoc = [0] * L 
            tc = [utils.allgather_perf_model(s, P, self._density) for s in sizes]
            taoc[L-1] = taos[L-1] + ts[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taos[l] + ts[l])
            return taoc, tc

        def __merge(tb, ts, tc, p, l):
            tb[l-1] += tb[l]
            tb[l] = 0

            p[l-1] = p[l-1]+p[l]
            p[l] = 0

            tc[l-1] = utils.allgather_perf_model(p[l-1], P, self._density) 
            tc[l] = 0

            ts[l-1] = utils.topk_perf_model(p[l-1])
            ts[l] = 0

        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        tb = list(self._layerwise_times)
        taob, taos, ts = __calculate_sparse_and_backward_start(tb, p, L)
        taoc, tc = __calculate_comm_start(ts, taos, p, L)

        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        group.append(key)
        for l in range(1, L-1)[::-1]:
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx

            tw = tb[l-1]+utils.topk_perf_model(p[l]+p[l-1])\
                - utils.topk_perf_model(p[l]) - utils.topk_perf_model(p[l-1])\
                - (taoc[l] - (taos[l]+ts[l]))
            tsave = utils.allgather_perf_model(p[l], P, self._density)+utils.allgather_perf_model(p[l-1], P, self._density)-\
                    utils.allgather_perf_model((p[l]+p[l-1]), P, self._density)
            if tw < tsave:
                __merge(tb, ts, tc, p, l)
                taob2, taos2, ts2 = __calculate_sparse_and_backward_start(tb[:l], p[:l], l, start=taob[l]+tb[l])
                taob[:l] = taob2
                taos[:l] = taos2
                taoc, tc = __calculate_comm_start(ts, taos, p, L)
            else:
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_merged_parameters(self):
        """Generate merged gradients.
        """
        self._merged_parameters = {}
        self._merged_parameter_names = {}

        if self._mgwfbp and self._layerwise_times is not None:
            if self._density < 1: # MGS 
                groups, key_groupidx_maps = self._generate_groups_mgs()
            else:
                groups, key_groupidx_maps = self._generate_groups_mgwfbp()
        else:
            groups, key_groupidx_maps = self._generate_groups_with_threshold(self._threshold)
        new_keys = []
        self._merged_parameter_offsets = {}
        self._layerwise_compressors = None
        self._layerwise_compressors = {}
        num_of_workers = size()
        for g in groups:
            sub_size = 0
            offsets = []
            computation_time = 0
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
                if self._original_layerwise_times_kv is not None and k in self._original_layerwise_times_kv and ADAPTIVE_SPARSE:
                    computation_time += self._original_layerwise_times_kv[k]
            new_key = ':'.join(g)
            new_keys.append(new_key)
            t = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False)
            self._merged_parameters[new_key] = t
            self._merged_parameter_names[t] = new_key
            self._merged_parameter_offsets[new_key] = offsets
            if self._density < 1 and ADAPTIVE_SPARSE:
                _density = utils.predict_density_with_size_and_computation(sub_size, computation_time, num_of_workers)
                density = max(_density, self._density)
            else:
                density = self._density
            if self._layerwise_compressors is not None:
                self._layerwise_compressors[new_key] = density
        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)


    def _push_to_buffer(self, name, tensor):
        r"""Push the ``tensor`` with ``name`` into buffer.
        
        Args:
            name (str): Tensor name.
            tensor (Tensor): Tensor.

        Returns:
            new_key (str): New generated name.
            tensor (Tensor): New merged tensor.
        """
        with torch.no_grad():
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                return name, new_tensor 
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()
            self._merged_parameters[new_key].data[offset:offset+numel].copy_(tensor.view(numel))
            self._groups_flags[group_idx][layer_idx] = 1
            for idx in self._groups_flags[group_idx]:
                if idx == 0:
                    return name, None
            return new_key, self._merged_parameters[new_key]

    def _pull_from_buffer(self, name, merged_tensor):
        r"""Split the merged tensors into multiple original tensors.
        
        Args:
            name (str): Merged tensor name.
            merged_tensor (Tensor): Merged tensor.

        Returns:
            tensors (list): The original tensors.
        """
        if len(self._groups) == len(self._sequential_keys):
            shape = self._named_parameters[name].data.shape
            return {name: merged_tensor.view(shape)} 
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx])
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensors[k] = merged_tensor.data[offset:offset+numel].view(original_tensor.shape)
        return tensors

    def _allreduce_grad_async(self, p, name):
        r"""Invoke a dense allreduce operation asynchronizely.

        Args: 
            p (Tensor): Tensor to be allreduced.
            name (str): Tensor name.

        Returns:
            handle (Handler): Horovod allreduce handler.
            ctx (tuple): Shape (None here).
        """
        tensor = p.data.view(-1)
        if False and rank() == 0 and self.train_iter % 200 == 0 and self.train_iter < 3000:
            grads = tensor.cpu().numpy()
            layer_idx = self._sequential_keys.index(name)
            np.save('%s/r%d_gradients_iter_%d::%s::%d' % (self._gradient_path, rank(), self.train_iter, name, layer_idx), grads)
        allreduce_name = name
        if len(name) > 200:
            allreduce_name = name[0:100]+'...'+name[-100:]
        handle = comm.allReduce(tensor)
        return (tensor,), None

    def _sparse_allreduce_async(self, p, name, density):
        r"""Invoke a sparse allreduce operation asynchronizely.

        Args: 
            p (Tensor): Tensor to be allreduced.
            name (str): Tensor name.
            density (float): Density.

        Returns:
            handle (Handler): Horovod allreduce handler.
            ctx (tuple): The original tensor shape.
        """
        stime = time.time()
        tensor = p.data.view(-1)
        tensor_compressed, ctx, selected_values = self._compression.compress(tensor, name, ratio=density)

        if False and rank() == 0 and self.train_iter % 200 == 0 and self.train_iter < 3000:
            grads = tensor.cpu().numpy()
            layer_idx = self._sequential_keys.index(name)
            np.save('%s/r%d_gradients_iter_%d::%s::%d' % (self._gradient_path, rank(), self.train_iter, name, layer_idx), grads)
        indexes = ctx

        if 'gtopk' in self._compression.name: # in ['gtopk', 'gtopkef', 'gtopkefddr4']:
            val_results, idx_results, included_indexes = gtopk_sparse_recursive_allreduce(comm, tensor_compressed, selected_values, ctx)
            if included_indexes is not None:
                self._compression.add_residuals(included_indexes, name)
        else:
            if indexes is None:
                val_results = torch.zeros(tensor_compressed.numel() * size(), device=tensor_compressed.device)
                handle = comm.allGather(tensor_compressed, val_results)
                idx_results = None
            else:
                val_results = torch.zeros(selected_values.numel() * size(), device=selected_values.device)
                handle = comm.allGather(selected_values, val_results)
                idx_results = torch.zeros(indexes.numel() * size(), device=indexes.device, dtype=indexes.dtype)
                handle_idx = comm.allGather(indexes, idx_results)

        if self._profiling:
            utils.force_insert_item(self._compression_timers, name, time.time()-stime)
        return (val_results, idx_results), ctx 

    def check_hooked_tensor_sequence(self, name):
        r"""Check the sequence of the backward tensors.
        """
        if self._seq_layernames is None:
            return
        ntensors = len(self._seq_layernames)
        idx = self._seq_layernames.index(name)
        if idx == ntensors-self._hook_checked_idx-1:
            self._hook_checked_idx += 1
            if idx == 0:
                self._hook_checked_idx = 0
        else:
            raise

    def _make_hook(self, p):
        r"""Add hooks for backward propagation.
        """
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local:
                name = self._parameter_names.get(p)
                #self.check_hooked_tensor_sequence(name)
                d_p = p.grad.data

                if self.momentum_correction and self._sparse:
                    param_state = self.state[p]
                    momentum = 0.9
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                    d_p = buf

                new_name, new_tensor = self._push_to_buffer(name, d_p)
                self._final_name = new_name
                self._final_tensor = new_tensor
                if not self._multi_job_scheduling and new_tensor is not None:
                    density = self.get_current_density(name=new_name)
                    if self._sparse and density < 1:
                        handle, ctx = self._sparse_allreduce_async(new_tensor, new_name, density)
                        self._handles[new_tensor] = (handle, ctx, density)
                    else:
                        handle, ctx = self._allreduce_grad_async(new_tensor, new_name)
                        self._handles[new_tensor] = (handle, ctx, 1)
        return hook

    def issue_allreduce(self, name, tensor):
        r"""Issue an allreduce operation for a tensor.

        Args:
            name (str): Tensor name.
            tensor (Tensor): The tensor for allreduce.
        """
        density = self.get_current_density(name=name)
        if self._sparse and density < 1:
            handle, ctx = self._sparse_allreduce_async(tensor, name, density)
            self._handles[tensor] = (handle, ctx, density)
        else:
            handle, ctx = self._allreduce_grad_async(tensor, name)
            self._handles[tensor] = (handle, ctx, 1)

    def synchronize(self):
        r"""Synchronize the allreduce operations.
        """

        num_of_workers = size()
        comm.synchronize()
        for p, value in self._handles.items():
            name = self._merged_parameter_names.get(p)
            handle, ctx, density = value
            if self._sparse and density < 1:
                stime = time.time()
                handle_idx = None
                all_indexes = None
                if type(handle) is tuple:
                    output, all_indexes = handle[0], handle[1]

                if self._profiling:
                    utils.force_insert_item(self._allreduce_timers, name, time.time()-stime)
                stime = time.time()
                new_grad = p.data.view(-1)
                new_grad.fill_(0.0)
                numel = output.size(0)
                real_num_values = numel//num_of_workers
                for i in range(num_of_workers):
                    values_and_indexes = output.data[i*real_num_values:(i+1)*real_num_values]
                    if all_indexes is None:
                        values = values_and_indexes
                        indexes = None
                        per_values = values
                        per_values = self._compression.decompress(per_values, p.size())
                        new_grad += per_values.view(-1)
                    else:
                        values = values_and_indexes
                        indexes = all_indexes.data[i*real_num_values:(i+1)*real_num_values].long()
                        per_values = values[0:indexes.numel()]
                        per_values = self._compression.decompress(per_values, p.size())
                        new_grad[indexes[0:indexes.numel()]] += per_values
                new_grad /= num_of_workers

                if self._profiling:
                    utils.force_insert_item(self._update_times, name, time.time()-stime)
            else:
                stime = time.time()
                output = handle[0] 
                if self._profiling:
                    utils.force_insert_item(self._allreduce_timers, name, time.time()-stime)
                stime = time.time()

                if self._norm_clip is not None:
                    norm_clip = np.sqrt(1.0/size()) * self._norm_clip
                    norm_type = 2.0
                    param_norm = output.norm(norm_type)
                    total_norm = param_norm.item() 
                    clip_coef = norm_clip / (total_norm + 1e-6)
                    if clip_coef < 1:
                        output.mul_(clip_coef)
                if self._compression:
                    output = self._compression.decompress(output, p.size())
                p.set_(output/size())
                if self._profiling:
                    utils.force_insert_item(self._update_times, name, time.time()-stime)
        if len(self._groups) != len(self._sequential_keys):
            for merged_p, value in self._handles.items():
                new_name = self._merged_parameter_names.get(merged_p)
                tensors = self._pull_from_buffer(new_name, merged_p)
                for n in tensors:
                    p = self._named_parameters.get(n)
                    if self._fp16:
                        p.grad.set_(tensors[n].data.type(p.grad.type()))
                    else:
                        p.grad.set_(tensors[n].data)
        self.train_iter += 1
        self._handles.clear()
        self._print_profiling()


    def _print_profiling(self):
        r"""Print time profiler.
        """
        if self._profiling and rank() == 0 and len(self._allreduce_timers.keys()) > 0 and len(self._allreduce_timers.get(self._allreduce_timers.keys()[0], [])) ==  40:
            cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times
            ups = self._update_times # update times
            r = rank()
            tcp = 0.0; tar = 0.0; tup = 0.0; total=0.0
            for k in cps:
                acp = np.mean(cps[k])
                tcp += acp
                aar = np.mean(ars[k])
                tar += aar
                aup = np.mean(ups[k])
                tup += aup
                #logger.info('[%d][%s]: %f, %f, %f', r, k, acp, aar, aup)
            total = tcp+tar+tup
            cps.clear()
            ars.clear()
            ups.clear()


    def _step_with_mc(self, closure=None):
        r"""Performs a single optimization step with momentum correction.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        offset = 0
        density = self.get_current_density()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                name = self._parameter_names.get(p)
                if weight_decay != 0:
                    wd = p.data
                    d_p.add_(weight_decay, wd)
                if momentum != 0 and not self.momentum_correction:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                if momentum != 0 and self.momentum_correction and density < 1:
                    param_state = self.state[p]
                    buf = param_state['momentum_buffer']
                    if self._compression.zc is not None:
                        buf.view(-1).mul_(self._compression.zc[offset:offset+d_p.numel()])
                        offset += d_p.numel()
        return loss

    def step(self, closure=None):
        r"""Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
        """
        if self._multi_job_scheduling:
            self.issue_allreduce(self._final_name, self._final_tensor)
        if not self.local:
            self.synchronize()
        if self.momentum_correction and self._sparse:
            return self._step_with_mc(closure)
        return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, named_parameters=None, compression=None, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, momentum_correction=False, fp16=False, mgwfbp=False, rdma=False, multi_job_scheduling=False):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, compression, is_sparse, density,
               seq_layernames=seq_layernames,
               layerwise_times=layerwise_times,
               norm_clip=None,
               threshold=threshold,
               writer=writer,
               gradient_path=gradient_path,
               momentum_correction=momentum_correction,
               fp16=fp16,
               mgwfbp=mgwfbp,
               rdma=rdma,
               multi_job_scheduling=multi_job_scheduling)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    for name, p in params:
        comm.bcast(p, root_rank)

    comm.synchronize()


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, _ in params:
        if key in callbacks:
            callbacks[key]()
