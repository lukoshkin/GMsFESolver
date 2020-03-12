import numpy as np
from mpi4py import MPI

comm = MPI.Comm.Dup(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()

ksize = 4

kcomm = MPI.Comm.Split(comm, rank//ksize, rank)
krank = kcomm.Get_rank()

census = kcomm.gather(rank, root=0)
if krank == 0:
    print(census)

group = MPI.Comm.Get_group(comm)
kgroup = MPI.Comm.Get_group(kcomm)
ranks = [1, 2, 3, 5, 7]
pgroup = group.Incl(ranks)

prank = 'not defined'
pcomm = MPI.Comm.Create(comm, pgroup)
if pcomm != MPI.COMM_NULL: prank = pcomm.Get_rank()

ranks = np.arange(0, size, ksize)
cgroup = group.Incl(ranks)
ccomm = MPI.Comm.Create_group(comm, cgroup)


census


print(rank, krank, prank)
