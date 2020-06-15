from core_components import *
from utils import get_simple_kappa
from mpi4py import MPI
                
scomm = MPI.COMM_SELF
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Init Stage
# ---------------
n_el = 32            # num of cells in a coarse block along 1D
n_blocks = 4         # num of coarse blocks along 1D

M_off = 10           # dimensionality of offline (online) space
eta = 1e3            # permeability coefficient's bursts
# ---------------
num_cnbh = (n_blocks-1)**2

assert size == num_cnbh, ('This is the implementation with equal'
        ' number of processes and number of coarse neighborhoods')

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
t_off =- MPI.Wtime()
# -----------------
Nv = core.snapshotSpace(k)
Nv,_ = core.modelReduction(k, Nv, M_off)
# -----------------
t_off += MPI.Wtime()

## Online Stage
t_on =- MPI.Wtime()
# -----------------
Nv_i = core.multiscaleFunctions(k, Nv, W)

pairs = overlap_map(n_blocks-1)
def communications(rank):
    imask = (pairs[:, 0] == rank)
    omask = (pairs[:, 1] == rank)
    send_to = pairs[omask][:, 0]
    recv_from = pairs[imask][:, 1]
    return send_to, recv_from

send_to, recv_from = communications(rank)
recvbuf = np.empty((len(recv_from), *Nv_i.shape))
for dest in send_to:
    comm.Isend([Nv_i, MPI.DOUBLE], dest)
for i, src in enumerate(recv_from):
    comm.Recv(recvbuf[i], src)

## Global Coupling
A_ii, b_i = core.diagonalBlock(k, Nv, rhs)

nodebuf = []
for Nv_j in recvbuf:
    pos = recv_from[i] - rank
    A_ij = core.offdiagonalBlock(K, Nv_i, Nv_j, pos)
    nodebuf.append(A_ij)
nodebuf = np.array(nodebuf)

all_offdiagA, all_diagA, all_b = None, None, None
if rank == 0:
    all_offdiagA = np.empty([len(pairs), *A_ii.shape])
    all_diagA = np.empty([num_cnbh, *A_ij.shape])
    all_b = np.empty([num_cnbh, len(b_i)])

sendcounts = np.full(num_cnbh, M_off*M_off)
for r in range(num_cnbh):
    sendcounts[r] *= len(communications(r)[1])

comm.Gatherv(nodebuf, (all_offdiagA, sendcounts), root=0)
comm.Gather(A_ii, all_diagA, root=0)
comm.Gather(b_i, all_b, root=0)
# -----------------
t_on += MPI.Wtime()

### Assembling on the root
if rank == 0:
    height = M_off*num_cnbh
    A = np.zeros([height, height])
    b = np.zeros(height)

    for i, A_ii in enumerate(all_diagA):
        A[i*M_off:(i+1)*M_off, i*M_off:(i+1)*M_off] = A_ii
        b[i*M_off:(i+1)*M_off] = all_b[i]

    for i, A_i in enumerate(all_offdiagA):
        for j in pairs[i]:
            A[i*M_off:(i+1)*M_off, j*M_off:(j+1)*M_off] = A_i[j]

    ij_lower = np.tril_indices_from(A, -1)
    A[ij_lower] = A.T[ij_lower]

    print(f'OFFLINE STAGE: {t_off:.3}s')
    print(f'ONLINE STAGE: {t_on:.3}s')


ms_dofs = None
if rank == 0:
    ms_dofs = np.empty((num_cnbh, Nv_i.shape)) 

comm.Gather(Nv_i, ms_dofs, root=0)
if rank == 0:
    u = scipy.linalg.solve(A, b, assume_a='pos')
    Psi_ms = cutils.compose(ms_dofs, W._cpp_object)
    sol_dofs = project(np.dot(u, Psi_ms), W).vector()[:]
    np.save('solution', sol_dofs)
