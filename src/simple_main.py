import numpy as np

from mpi4py import MPI
from mpi_gmsfem import *
from optimal_distribution import ColoredPartition
from utils import get_simple_kappa
                
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

args = (n_el, n_blocks, scomm)

eta = [10, 1e2, 1e3, 4e3]
K = get_simple_kappa(eta, N_el, comm=scomm, seed=123)
N_v = len(eta)

calc_weigths = lambda k: k.vector().get_local().sum()
weights = np.vectorize(calc_weigths, 'O')(K)
avg_K = np.sum(K/weights)


print("---- BUILDING SNAPSHOT SPACE ----")
ms_solver = SpaceBlocks(rank, *args)
psi_snap = np.empty([N_v, n_el*n_el], 'O')
for i,kappa in enumerate(K):
    psi_snap[i] = ms_solver.buildSnapshotSpace(kappa)
psi_snap = psi_snap.reshape(-1)

print("==== BUILDING OFFLINE SPACE ====")
psi_off,_ = ms_solver.buildOfflineSpace(avg_K, psi_snap, M_off)

print("#### BUILDING ONLINE SPACE ####")
for _ in range(rank, N_c*N_v, size):
    psi_ms,_ = ms_solver.buildOnlineSpace(psi_off)

recvbuf, received = None, None
if rank == 0:
    recvbuf = np.empty([size, M_on, ms_solver.V.dim()])

sendbuf = get_dofs(psi_ms)
comm.Allgather(sendbuf, recvbuf)
psi_ms = restore_fns(recvbuf)
