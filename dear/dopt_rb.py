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

"""
Experimental: decompose all-reduce primitive into reduce and broadcast
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import collections
import numpy as np

from comm_core import rank, size, Communicator, init as comm_init
#from acomm import rank, size, Communicator, init as comm_init

from tensorfusion import MergedCommCollective, CollectiveOp, MergedCommReduce

import utils

MERGE = True
comm_init()
comm = None
bcastcomm = None
reducecomm = None

NSTREAMS=1
def init():
    global comm
    global bcastcomm 
    global reducecomm 
    comm = Communicator(NSTREAMS)
    reducecomm = MergedCommReduce(merge=MERGE, op=CollectiveOp.REDUCE)
    bcastcomm = MergedCommCollective(merge=False, single_layer=False, op=CollectiveOp.BCAST, nstreams=1)


ADAPTIVE_SPARSE = False
DEBUG = False


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, model, compression,
                 is_sparse=False,
                 density=0.001,
                 seq_layernames=None,
                 layerwise_times=None,
                 norm_clip=None,
                 threshold=0,
                 writer=None,
                 gradient_path=None,
                 fp16=False,
                 mgwfbp=False,
                 rdma=False,
                 multi_job_scheduling=False,
                 exclude_parts=''):
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
            _compression_timers (dict): Layer-wise compression timer.
            _allreduce_timers (dict): Layer-wise all-reduce timer.
            _update_timers (dict): Layer-wise update timer.
            train_epoch (int): Current training epoch. 
            train_iter (int): Current training iteration. 
            local (bool): Local update. True indicates not to aggregation.
        """
        super(self.__class__, self).__init__(params)

        self._compression = compression
        self._sparse = is_sparse
        self._multi_job_scheduling = multi_job_scheduling
        self._density = density
        self._profiling = False
        self._seq_layernames = None
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
        self.alpha = None
        self.beta = None
        if self._layerwise_times is not None and self._seq_layernames is not None:
            self._original_layerwise_times_kv = dict(zip(self._seq_layernames, \
                self._layerwise_times))
        self._compression_timers = {} # compression
        self._allreduce_timers = {} # allreduce times
        self._update_times = {} # allreduce times
        self._model = model
        self._inner_iter = 0
        self.train_epoch = 0
        self.train_iter = 0
        self.exclude_reduce = True if exclude_parts.find('reduce') >=0 else False
        self.exclude_bcast = True if exclude_parts.find('bcast') >=0 else False

        named_parameters = list(model.named_parameters())

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
        self._handles = {}
        self._bcast_handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self.local = False
        self._hook_checked_idx = 0
        self._synced = False

        self._modules = [] # From the first layer to the last 
        self._module_names = {}
        self._module_name_list = []
        self._module_tensors_name_list = []
        self._module_ranks = {}
        self._module_to_next_module = {}
        self._module_to_gradients = {}
        if size() > 1:
            self._register_hooks()
            self._generate_module_ranks(self._modules)

        self._timeline = os.environ.get('WFSGD_TIMELINE', '')

    def _register_hooks(self):
        r"""Register hooks.
        """
        model = self._model
        name_idx = 0
        if not self.exclude_reduce:
            self._register_bp_hooks()

        for module in model.modules():
            classname = module.__class__.__name__
            module_name = 'module_name_%s_%d' % (classname, name_idx)
            if hasattr(module, 'weight'):
                module.register_forward_pre_hook(self._forward_pre_hook)
                name_idx += 1
                self._module_names[module] = module_name
                self._modules.append(module)
                self._module_name_list.append(module_name)
                self._module_tensors_name_list.append(module_name+'-weight')
            if hasattr(module, 'bias'):
                self._module_tensors_name_list.append(module_name+'-bias')


    def _make_hook(self, p):
        r"""Add hooks for backward propagation.
        """
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local:
                name = self._parameter_names.get(p)
                if self._inner_iter == 0:
                    if self._seq_layernames is None:
                        self._seq_layernames = []
                    self._seq_layernames.append(name)
                #else:
                #    if rank() == 0:
                #        print(self._seq_layernames.index(name), name)
                d_p = p.grad

                #current_stream = torch.cuda.current_stream()
                #current_stream.synchronize()

                root_rank = 0 #self._module_ranks[module]
                handle, ctx = self._reduce_grad_async(d_p, name, root_rank)
                if name not in self._handles:
                    self._handles[name] = []
                self._handles[name].append((handle, d_p, ctx, 1))
        return hook

    def _register_bp_hooks(self):
        r"""Register bp hooks.
        """
        self._seq_layernames = []
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)
                    name = self._parameter_names.get(p)
                    #self._seq_layernames.append(name)
        #if rank() == 0:
        #    print('_seq_layernames: ', self._seq_layernames)
        #reducecomm.init_tensor_group(self._seq_layernames[::-1])

    def _forward_pre_hook(self, module, input):
        """Hook for pre-fowarding"""
        if torch.is_grad_enabled():
            if not self.local and hasattr(module, 'weight'):
                if module not in self._bcast_handles:
                    return
                handles = self._bcast_handles[module]
                if handles is not None:
                    grads = []
                    for handle, new_tensor, ctx, _ in handles:
                        if handle is not None and handle >= 0:
                            comm.syncStream(handle)
                        grads.append(new_tensor)
                    self.step_one_module(module, grads)

                    if module not in self._module_to_next_module:
                        for m in self._modules:
                            if m in self._bcast_handles:
                                continue
                            tensors = self._get_grad(m)
                            if tensors is not None:
                                self._module_to_next_module[module] = m
                                self._module_to_gradients[m] = tensors
                                break
                    next_module = self._module_to_next_module.get(module, None)
                    if next_module:
                        self.bcast_one_module(next_module)

    def _generate_module_ranks(self, modules):
        nworkers = size()
        rank_list = list(range(nworkers))
        for i, module in enumerate(modules):
            rank = rank_list[i % nworkers]
            self._module_ranks[module] = 0 #rank

        #reducecomm.init_tensor_group(self._module_tensors_name_list[::-1])
        #bcastcomm.init_tensor_group(self._module_tensors_name_list)

    def _get_grad(self, module):
        """Get formated gradient of module

        Args:
          module: module/layer to get gradient of

        Returns:
          Formatted gradient with shape [output_dim, input_dim] for module
        """
        if not hasattr(module, 'weight') or module.weight.grad is None:
            return None
        grad = module.weight.grad.data.view(-1)
        tensors = [grad]
        if hasattr(module, 'bias') and module.bias is not None:
            tensors.append(module.bias.grad.data.view(-1))
        return tensors

    def _set_grad(self, module, p):
        if not hasattr(module, 'weight') or module.weight.grad is None:
            return None

        if hasattr(module, 'bias') and module.bias is not None:
            grad = p[0]
            bias = p[1]
        else:
            grad = p[0]
        module.weight.grad.data.view(-1).copy_(grad)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.grad.data.view(-1).copy_(bias)

    def _reduce_grad_async(self, p, name, r):
        r"""Invoke a dense reduce operation asynchronizely.

        Args: 
            p (Tensor): Tensor to be allreduced.
            name (str): Tensor name.
            rank (int): Rank for reduction.

        Returns:
            handle (Handler): NCCL reduce handler.
            ctx (tuple): Shape (None here).
        """
        tensor = p.data.view(-1)
        #handle = comm.reduce(tensor, r)
        handle = reducecomm.reduce_async_(name, tensor, r)
        return handle, None

    def bcast_one_module(self, module):
        if self.exclude_bcast:
            return
        name = self._module_names[module]
        root_rank = self._module_ranks[module]
        next_tensors = self._module_to_gradients[module]

        if module not in self._bcast_handles:
            self._bcast_handles[module] = []
        for i, next_tensor in enumerate(next_tensors):
            tensor_name = name
            if i == 0 and hasattr(module, 'weight'):
                tensor_name = name+'-weight'
            if i == 1 and hasattr(module, 'bias'):
                tensor_name = name+'-bias'

            if next_tensor is not None:
                #handle, ctx = bcastcomm.bcast(next_tensor, root_rank), None
                handle, ctx = bcastcomm.collective_async_(tensor_name, next_tensor, root_rank), None
                self._bcast_handles[module].append((handle, next_tensor, ctx, 1))

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

    def force_sync(self):
        """Should be invoked before inference
        """
        comm.synchronize()
        self._handles.clear()
        self._synced = True

    def synchronize(self):
        r"""Synchronize the allreduce operations.
        """
        reducecomm.synchronize()
        self._inner_iter += 1
        if self._inner_iter == 1 and not self.exclude_reduce:
            reducecomm.init_tensor_group(self._seq_layernames)
        self._handles.clear()
        self._bcast_handles.clear()

        first_module = None
        for m in self._modules:
            if m in self._bcast_handles:
                continue
            tensors = self._get_grad(m)
            if tensors is not None:
                first_module = m
                self._module_to_gradients[m] = tensors
                break
        if first_module:
            self.bcast_one_module(first_module)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
        """
        if not self.local:
            self.synchronize()

    def zero_grad(self):
        pass

    def step_one_module(self, module, grad, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
    
        #self._set_grad(module, grad)
        name = self._module_names[module]
    
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            target_p = []
            target_p.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                target_p.append(module.bias)
                
            for p in target_p:
                d_p = p.grad.data
                if weight_decay != 0:
                    wd = p.data
                    d_p.add_(wd, alpha=weight_decay)
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(d_p, alpha=-group['lr'])
                p.grad.fill_(0.0)
        return loss

def DistributedOptimizer(optimizer, model, compression=None, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, fp16=False, mgwfbp=False, rdma=False, multi_job_scheduling=False, exclude_parts=''):
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

    return cls(optimizer.param_groups, model, compression, is_sparse, density,
               seq_layernames=seq_layernames,
               layerwise_times=layerwise_times,
               norm_clip=None,
               threshold=threshold,
               writer=writer,
               gradient_path=gradient_path,
               fp16=fp16,
               mgwfbp=mgwfbp,
               rdma=rdma,
               multi_job_scheduling=multi_job_scheduling, 
               exclude_parts=exclude_parts)


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
        if p is not None:
            comm.bcast(p.view(-1), root_rank)
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

                if p is not None and not torch.is_tensor(p):
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

def allreduce(tensor, name=None):
    comm.allReduce(tensor.view(-1))
    comm.synchronize()
    return tensor/size()
