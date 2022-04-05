# BO
#rdma=0 nworkers=64 dnn=resnet50 bs=64 ./horovod_mpi_cj.sh
#rdma=0 nworkers=64 dnn=densenet201 bs=32 ./horovod_mpi_cj.sh
#rdma=0 nworkers=64 dnn=bert_base bs=64 ./horovod_mpi_cj.sh
#rdma=1 nworkers=64 dnn=resnet50 bs=64 ./horovod_mpi_cj.sh
#rdma=1 nworkers=64 dnn=densenet201 bs=32 ./horovod_mpi_cj.sh
#rdma=1 nworkers=64 dnn=bert_base bs=64 ./horovod_mpi_cj.sh

# single-GPU time
rdma=0 nworkers=1 dnn=resnet50 bs=64 ./horovod_mpi_cj.sh
rdma=0 nworkers=1 dnn=densenet201 bs=32 ./horovod_mpi_cj.sh
rdma=0 nworkers=1 dnn=bert_base bs=64 ./horovod_mpi_cj.sh
