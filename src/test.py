import numpy as np

from mpi_gmsfem import *
from utils import get_simple_kappa
from mpi4py import MPI

scomm = MPI.COMM_SELF
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_el = 8
n_blocks = 3
N_el = n_el * n_blocks
N_c = (n_blocks-1)*(n_blocks-1)
M_off = 10
M_on = 10


world_group = comm.Get_group()
# krank = kcomm.Get_rank()
# kcomm = MPI.Comm.Split(world_group, rank//3, rank)
# print('-------')
# print(kcomm.Get_rank())
# print('-------')
args = (n_el, n_blocks, scomm)

eta = [10, 1e2, 1e3, 4e3]
K = get_simple_kappa(eta, N_el, comm=scomm, seed=123)
N_v = len(eta)

def get_dofs(fns):
    out = []
    for fn in fns:
        dofs = fn.compute_vertex_values()
        out.append(dofs)
    return np.array(out)

def restore_fns(vdofs, V):
    out = []
    sh = vdofs.shape
    v2d = vertex_to_dof_map(V)
    for dofs in vdofs.reshape(-1, sh[-1]):
        fn = Function(V)
        fn.vector()[v2d] = dofs
        out.append(fn)
    return np.array(out).reshape(*sh)

print('BUILDING BLOCKS')
nodeload = []
for r in range(rank, 2, size):
    ms_solver = SpaceBlocks(rank, *args)
    psi_snap_k = ms_solver.buildSnapshotSpace(K[rank])
    nodeload.append(psi_snap_k)

recvbuf = None
if rank == 0:
    recvbuf = np.empty([8, n_el*n_el, ms_solver.V.dim()])

print('SENDING DATA')
sendbuf = []
for snap_i in nodeload:
    dofs_i = get_dofs(snap_i)
    sendbuf.append(dofs_i)
sendbuf = np.array(sendbuf)
print('SENT SUCCESSFULLY')

print('RECEIVING DATA')
t1 =- MPI.Wtime()
comm.Gather(sendbuf, recvbuf, root=0)
t1 += MPI.Wtime()
print('RECEIVED SUCCESSFULLY')

t2 =- MPI.Wtime()
kcomm = comm.Create_group(world_group.Incl([0, 1]))
if (kcomm != MPI.COMM_NULL):
    print('RECEIVING DATA')
    kcomm.Gather(sendbuf, recvbuf, root=0)
    print('RECEIVED SUCCESSFULLY')
t2 += MPI.Wtime()

if rank == 0:
    print(len(recvbuf))
    print(len(recvbuf[0]))


# recvbuf = None
# sendbuf = np.ones(5) + rank
# if rank == 0:
#     print('finilizing')
#     recvbuf = np.empty((size, 5))
# comm.Gather(sendbuf, recvbuf, root=0)
# 
# if rank == 0: print(recvbuf)

if rank == 0:
    print('whole group', t1)
    print('subgroup', t2)
    print('t1/t2', t1/t2)

if kcomm != MPI.COMM_NULL: kcomm.Free()
MPI.Finalize()
