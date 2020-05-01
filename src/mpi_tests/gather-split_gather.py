import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

kcomm = comm.Split(rank%2, rank)

if False:
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

if True:
    data = rank
    data = kcomm.gather(data, root=0)
    if rank in [0, 1]:
        print(data)

MPI.Finalize()
