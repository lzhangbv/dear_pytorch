# DeAR: <u>de</u>coupling the <u>a</u>ll-<u>r</u>educe primitive to accelerate distributed deep learning

## Introduction 
We propose a new optimization algorithm called DeAR, that decouples the all-reduce primitive to two operations, so as to enable fine-grained scheduling without introducing extra communication overhead. This repository contains DeAR's source code, as well as a set of benchmarking scripts for evaluating the training performance of popular distributed deep learning methods with data parallelism. Currently, it covers: 
### Optimization algorithms without Tensor Fusion
- Wait-free backpropagation (WFBP), which is also known as the technique of pipelining the backward computations with gradient communications. 
- [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler), which uses tensor partition and priority schedule to overlap some communication tasks with
feed-forward computing tasks. 
- DeAR w/o TF, which disables the tensor fusion technique by setting THRESHOLD=None and NUM_NEARBY_LAYERS=1. 
### Optimization algorithms with Tensor Fusion
- [Horovod](https://github.com/horovod/horovod). 
- [PyTorch-DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
- [MG-WFBP](https://github.com/HKBU-HPML/MG-WFBP), which determines fusion tensors by measuring the backward computation time and communication time. 
- DeAR, which supports tuning tensor fusion with [Bayesian optimization](https://github.com/fmfn/BayesianOptimization). 

### Deep Neural Networks
- [Convolutional neural networks (CNNs)](https://pytorch.org/vision/stable/models.html) on a fake ImageNet data set (i.e., randomly generate the input image of 224\*224\*3)
- [Transformers](https://github.com/huggingface/transformers): BERT-Base and BERT-Large pretraining models.

## Installation
### Prerequisites
- Python 3.6+
- CUDA-10.+
- NCCL-2.4.+
- [PyTorch-1.8.+](https://download.pytorch.org/whl/torch_stable.html)
- [OpenMPI-4.0.+](https://www.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.19.+](https://github.com/horovod/horovod)
- [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler)

### Get the code
```
$git clone https://github.com/lzhangbv/dear_pytorch.git
$cd dear_pytorch
$pip install -r requirements.txt
```

### Configure the cluster settings
Before running the scripts, please carefully configure the configuration files in the directory of `configs`.
- configs/cluster\*: configure the host files for MPI
- configs/envs.conf: configure the cluster environments

Compile the communication package:
```
$ bash common/comm_core/compile.sh
```

Create a log folder, e.g., 
```
$mkdir -p logs/sc22tf
```

### Run benchmarks
- The batch mode
```
$python benchmarks.py
```

For different experimental settings, users can modify the DNN model, batch size, the number of GPUs, and network configurations in the benckmarks.py script. 


- The individual mode, e.g.,
```
$cd dear
$dnn=resnet50 bs=64 nworkers=64 ./horovod_mpi_cj.sh
```

Before running DeAR w/o tensor fusion, please set THRESHOLD=None and NUM_NEARBY_LAYERS=1 in the DeAR's dopt_rsag.py script. For DeAR with tensor fusion, we use THRESHOLD=25MB by default. To support Bayesian optimization, please import dopt_rsag_bo and increase the num-warmup-batches to at least 60 to tune buffer size in DeAR's benchmark scripts. 

## DeAR Usage
The DeAR distributed optimizer can be easily used like `horovod.DistributedOptimizer()`.
```Python
import dear
dear.init()
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = dear.DistributedOptimizer(optimizer, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
...
```

### DeAR Example
Example script for training on MNIST was provided.
```
$ bash mnist.sh
```

<!-- ## Paper
If you are using this repository for your paper, please cite our work
```
@article{shi2020ddlsurvey,
    author = {Shi, Shaohuai and Tang, Zhenheng and Chu, Xiaowen and Liu, Chengjian and Wang, Wei and Li, Bo},
    title = {Communication-Efficient Distributed Deep Learning: Survey, Evaluation, and Challenges},
    journal = {arXiv},
    year = {2020}
}
``` -->
