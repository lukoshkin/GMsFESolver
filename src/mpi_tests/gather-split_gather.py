import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

kcomm = comm.Split(rank%2, rank)

experiment = 3

if experiment == 1:
    sendbuf = np.ones(5) + rank
    recvbuf = None
    if rank == 4:
        recvbuf = np.empty((size, 5))
    print('RECEIVING DATA')
    comm.Gather(sendbuf, recvbuf, root=0)
    print('RECEIVED SUCCESSFULLY')

    if rank == 0:
        print(len(recvbuf))
        print(len(recvbuf[0]))
        print(recvbuf)

if experiment == 2:
    data = rank
    data = kcomm.gather(data, root=0)
    if rank in [0, 1]:
        print(data)

if experiment == 3:
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((6, 5, 5), 'f8')
    if rank < 6:
        sendbuf = np.ones((5, 5)) * rank
    else:
        sendbuf = np.array([], 'f8')
    sendcounts = [25, 25, 25, 25, 25, 25, 0, 0]
    comm.Gatherv(sendbuf, (recvbuf, sendcounts), root=0)

    if rank == 0: print(recvbuf)

#MPI.Finalize()
