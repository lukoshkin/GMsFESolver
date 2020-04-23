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

def get_dofs(fns):
    out = []
    for fn in fns:
        dofs = fn.compute_vertex_values()
        out.append(dofs)
    return np.array(out)

def restore_fns(vdofs, V):
    out = []
    v2d = vertex_to_dof_map(V)
    for dofs in vdofs:
        fn = Function(V)
        fn.vector()[v2d] = dofs
        out.append(fn)
    return np.array(out)


print("---- BUILDING SNAPSHOT SPACE ----")
nodebuf, sendbuf = [], []
cp = ColoredPartition(size, N_v)
n_comms,_ = cp.partition(N_c)

world_group = comm.Get_group()
subcomm = comm.Create_group(world_group.Incl(np.arange(n_comms)))
kcomm = MPI.Comm.Split(subcomm, rank//N_v, rank)
if kcomm != MPI.COMM_NULL: krank = kcomm.Get_rank()

ms_solver = {}
for r in cp.map['r'][rank]:
    if r is not None and r not in ms_solver.keys():
        ms_solver[r] = SpaceBlocks(r, *args)

for c, r, f in cp.getZip(rank):
    psi_snap_k = ms_solver[r].buildSnapshotSpace(K[c])
    if f: sendbuf.append(get_dofs(psi_snap_k))
    else: nodebuf.append(psi_snap_k)

recvbuf = None
if kcomm != MPI.COMM_NULL and kcomm.Get_rank() == 0:
    pack_size = cp.countMSGTransfers()[rank//N_v]
    recvbuf = np.empty([pack_size, n_el*n_el, ms_solver.V.dim()])

if kcomm != MPI.COMM_NULL:
    kcomm.Gather(np.array(sendbuf), recvbuf, root=0)

print("==== BUILDING OFFLINE SPACE ====")
for r in range(rank, N_c*N_v, size):
    if r not in major_ranks: continue
    recvbuf = recvbuf.reshape(N_v*n_el*n_el, -1)
    psi_snap = restore_fns(recvbuf, ms_solver.V)
    psi_off,_ = ms_solver.buildOfflineSpace(avg_K, psi_snap)

data = get_dofs(psi_off)
kcomm.Bcast(data, root=0)
psi_off = restore_fns(psi_off, ms_solver.V)

print("#### BUILDING ONLINE SPACE ####")
for _ in range(rank, N_c*N_v, size):
    psi_ms,_ = ms_solver.buildOnlineSpace(psi_off)

recvbuf, received = None, None
if rank == 0:
    recvbuf = np.empty([size, M_on, ms_solver.V.dim()])

sendbuf = get_dofs(psi_ms)
comm.Allgather(sendbuf, recvbuf)
psi_ms = restore_fns(recvbuf)


