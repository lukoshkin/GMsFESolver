from core_components import *
from utils import get_simple_kappa
from mpi4py import MPI
                
scomm = MPI.COMM_SELF
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Init Stage
# ---------------
n_el = 8            # num of cells in a coarse block along 1D 
n_blocks = 4        # num of coarse blocks along 1D

M_off = 10          # dimensionality of offline (online) space
eta = 1e3           # permeability coefficient's bursts
# ---------------

col = rank % (n_blocks-1)
row = rank // (n_blocks-1)

mesh = UnitSquareMesh(scomm, 2*n_el, 2*n_el)
mesh.translate(Point(col*.5, row*.5))
mesh.scale(2./n_blocks)

RHS = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', pi=np.pi, degree=1)
K = get_simple_kappa(eta, n_el*n_blocks, comm=scomm, seed=123)
V = FunctionSpace(mesh, 'P', 1)
W = K.function_space()

rhs = project(RHS, W)
k = project(K, V)

with open('cutils/pymodule2.cpp', 'r') as fp:
    cutils = compile_cpp_code(fp.read(), include_dirs=['cutils'])
core = GMsFEUnit(rank, n_el, n_blocks, cutils)

##  Offline Stage
Nv = core.snapshotSpace(k)
print('HERE1')
Nv,_ = core.modelReduction(k, Nv, M_off)
print('HERE2')

## Online Stage
Nv_i = core.multiscaleFunctions(k, Nv, W)

pairs = overlap_map(n_blocks-1)
size_mask = (pairs >= size).sum(1).astype(bool)
pairs = pairs[~size_mask]

imask = (pairs[:, 0] == rank)
omask = (pairs[:, 1] == rank)
send_to = pairs[omask][:, 0]
recv_from = pairs[imask][:, 1]

recvbuf = np.empty((len(recv_from), *Nv_i.shape))
for dest in send_to:
    comm.Isend([Nv_i, MPI.DOUBLE], dest)
for i, src in enumerate(recv_from):
    comm.Recv(recvbuf[i], src)

## Global Coupling
A_ii, b_i = core.diagonalBlock(k, Nv, rhs)
for i, Nv_j in enumerate(recvbuf):
    pos = recv_from[i] - i
    A_ij = core.offdiagonalBlock(K, Nv_i, Nv_j, pos)

all_offdiagA, all_diagA, all_b = None, None, None
if rank == 0:
    all_offdiagA = np.empty([len(pairs), *A_ii.shape])
    all_diagA = np.empty([(n_blocks-1)**2, *A_ij.shape])
    all_b = np.empty([(n_blocks-1)**2, len(b_i)])

comm.Gather(A_ij, all_offdiagA, root=0)
comm.Gather(A_ii, all_diagA, root=0)
comm.Gather(b_i, all_b, root=0)

## Assembling on the root
if rank == 0:
    height = M_off*(n_blocks-1)**2
    A = np.zeros([height, height])
    b = np.zeros(height)

    for i, A_ii in enumerate(all_diagA):
        A[i*M_off:(i+1)*M_off, i*M_off:(i+1)*M_off] = A_ii
        b[i*M_off:(i+1)*M_off] = all_b[i]

    for i, A_i in enumerate(all_offdiagA):
        for j in pairs[i]:
            A[i*M_off:(i+1)*M_off, j*M_off:(j+1)*M_off] = A_i[j]

    A[ij_lower] = np.tril_indices_from(A, -1)
    A[ij_lower] = A.T[ij_lower]

    u = scipy.linalg.solve(A, b, assume_a='pos')

MPI.Finalize()
