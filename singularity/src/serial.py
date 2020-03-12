import numpy as np

from mpi_gmsfem import *
from utils import get_simple_kappa
from mpi4py import MPI

n_el = 8
n_blocks = 3
N_el = n_el * n_blocks
N_c = (n_blocks-1)*(n_blocks-1)
M_off = 10
M_on = 10

args = (n_el, n_blocks, M_off, M_on)

eta = [10, 1e2, 1e3, 4e3]
K = get_simple_kappa(eta, N_el, comm=scomm, seed=123)
N_v = len(eta)

calc_weigths = lambda k: k.vector().get_local().sum()
weights = np.vectorize(calc_weigths, 'O')(K)
avg_K = np.sum(K/weights)

kcomm = MPI.Comm.Split(comm, rank//N_v, rank)
krank = kcomm.Get_rank()

major_ranks = np.arange(0, size, N_v)

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

print("---- BUILDING SNAPSHOT SPACE ----")
for r in range(rank, N_c*N_v, size):
    ms_solver = SpaceBlocks(r%N_c, *args)
    psi_snap_k = ms_solver.buildSnapshotSpace(K[rank])
    break

recvbuf = None
if rank == 0:
    recvbuf = np.empty([N_v, n_el, ms_solver.V.dim()])

sendbuf = get_dofs(psi_snap_k)
kcomm.Gather(sendbuf, recvbuf, root=0)

print("==== BUILDING OFFLINE SPACE ====")
for r in range(rank, N_c, size):
    if r not in major_ranks: continue
    recvbuf = recvbuf.reshape(-1, recvbuf.shape[-1])
    psi_snap = restore_fns(recvbuf, ms_solver.V)
    psi_off,_ = ms_solver.buildOfflineSpace(avg_K, psi_snap)
    break

data = get_dofs(psi_off)
kcomm.Bcast(data, root=0)
psi_off = restore_fns(psi_off, ms_solver.V)

print("#### BUILDING ONLINE SPACE ####")
for _ in range(rank, N_c*N_v, size):
    psi_ms,_ = ms_solver.buildOnlineSpace(psi_off)
    break

recvbuf, received = None, None
if rank == 0:
    recvbuf = np.empty([size, M_on, ms_solver.V.dim()])

sendbuf = get_dofs(psi_ms)
comm.Allgather(sendbuf, recvbuf)
psi_ms = restore_fns(recvbuf)

#times, left = divmod(N_c*N_v, size)
#if not times: times += 1
#
#for _ in range(times):
#    comm.Gather(sendbuf,  recvbuf, root=0)
#    if rank==0: received = np.vstack([received, recvbuf])


#u = ms_solver.globalCoupling(kappa, rhs)
