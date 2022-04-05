# BO
#rdma=0 nworkers=64 dnn=resnet50 bs=64 ./horovod_mpi_cj.sh
#rdma=0 nworkers=64 dnn=densenet201 bs=32 ./horovod_mpi_cj.sh
#rdma=0 nworkers=64 dnn=bert_base bs=64 ./horovod_mpi_cj.sh
#rdma=1 nworkers=64 dnn=resnet50 bs=64 ./horovod_mpi_cj.sh
#rdma=1 nworkers=64 dnn=densenet201 bs=32 ./horovod_mpi_cj.sh
#rdma=1 nworkers=64 dnn=bert_base bs=64 ./horovod_mpi_cj.sh


# time breakdowns
nworkers=64
rdma=0

dnn=resnet50
bs=64
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=allgather ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=reducescatter ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma ./horovod_mpi_cj.sh

dnn=densenet201
bs=32
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=allgather ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=reducescatter ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma ./horovod_mpi_cj.sh

dnn=inceptionv4
bs=64
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=allgather ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=reducescatter ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma ./horovod_mpi_cj.sh

dnn=bert_base
bs=64
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=allgather ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=reducescatter ./horovod_mpi_cj.sh
#dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma ./horovod_mpi_cj.sh

dnn=bert
bs=32
dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=allgather ./horovod_mpi_cj.sh
dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma exclude_parts=reducescatter ./horovod_mpi_cj.sh
dnn=$dnn bs=$bs nworkers=$nworkers rdma=$rdma ./horovod_mpi_cj.sh

