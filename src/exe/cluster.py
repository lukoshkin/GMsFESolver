import re
import sys
from pathlib import Path
sys.path.append('..')

from gmsfem.petsc.p1case import * 
from gmsfem.subdiv import *
from gmsfem.utils import get_simple_kappa
from mpi4py import MPI
from petsc4py import PETSc
                
scomm = MPI.COMM_SELF
wcomm = MPI.COMM_WORLD
rank = wcomm.Get_rank()
size = wcomm.Get_size()

## Init Stage
# ---------------
n_el = 32            # num of cells in a coarse block along 1D
n_blocks = 4         # num of coarse blocks along 1D

M_off = 10           # dimensionality of offline (online) space
eta = 1e3            # permeability coefficient's bursts
# ---------------
n_nbh = (n_blocks-1)**2
height = M_off*n_nbh

#assert size == n_nbh, ('This is the implementation with equal'
#        ' number of processes and number of coarse neighborhoods')

col = rank % (n_blocks-1)
row = rank // (n_blocks-1)

mesh = UnitSquareMesh(scomm, 2*n_el, 2*n_el)
mesh.translate(Point(col*.5, row*.5))
mesh.scale(2./n_blocks)
V = FunctionSpace(mesh, 'P', 1)

RHS = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', pi=pi, degree=1)
K = get_simple_kappa(eta, n_el*n_blocks, comm=scomm, seed=123)
W = K.function_space()

rhs = project(RHS, V)
k = project(K, V)

if rank == 0: core = GMsFEUnit(0, n_el, n_blocks)
else: core = GMsFEUnit(rank, n_el, n_blocks)

wcomm.barrier()
if rank == 0: 
    for p in Path('.').iterdir():
        if re.search('mesh[0-9]+\.(xdmf|h5)', str(p)):
            p.unlink()
wcomm.barrier()

##  Offline Stage
t_off =- MPI.Wtime()
# -----------------
Nv = core.snapshotSpace(k)
# -----------------
t_off += MPI.Wtime()

## Online Stage
t_on =- MPI.Wtime()
# -----------------
Nv,_ = core.modelReduction(k, Nv, M_off)
Nv_i = core.multiscaleDOFs(k, Nv)
ms_dofs = zero_extrapolate(Nv_i, V, W, col, row)
## -----------------
t_on += MPI.Wtime()

pairs = overlap_map(n_blocks-1)
mask = (pairs >= size).sum(1).astype(bool)
pairs = pairs[~mask]

def communications(rank):
    imask = (pairs[:, 0] == rank)
    omask = (pairs[:, 1] == rank)
    send_to = pairs[omask][:, 0]
    recv_from = pairs[imask][:, 1]
    return send_to, recv_from

## Global Coupling
t_sr =- MPI.Wtime()
## ----------------
send_to, recv_from = communications(rank)
recvbuf = np.zeros((len(recv_from), *Nv_i.shape)) # empty

for dest in send_to:
    wcomm.Isend([Nv_i, MPI.DOUBLE], dest)
for i, src in enumerate(recv_from):
    wcomm.Recv(recvbuf[i], src)
## ----------------
t_sr += MPI.Wtime()

t_gc =- MPI.Wtime()
## ----------------
A = PETSc.Mat(wcomm)
A.createAIJ((height, height))
A.setUp()

b = PETScVector(wcomm, height)

A_ii, b_i = core.diagonalBlock(k, Nv_i, rhs)
I = np.arange(rank*M_off, (rank+1)*M_off, dtype='i4')
A.setValues(I, I, A_ii.getDenseArray().flatten())
b.set_local(b_i.getArray())

for j, Nv_j in zip(recv_from, recvbuf):
    if rank >= size or j >= size: continue
    A_ij = core.offdiagonalBlock(k, Nv_i, Nv_j, j-rank)
    I = np.arange(rank*M_off, (rank+1)*M_off, dtype='i4')
    J = np.arange(j*M_off, (j+1)*M_off, dtype='i4')
    A.setValues(I, J, A_ij.getDenseArray().flatten())
    A.setValues(J, I, A_ij.getDenseArray().T.flatten())

A.assemble()
A = PETScMatrix(A)
## ----------------
t_gc += MPI.Wtime()

# The next line supposes that you have petsc with superlu_dist installed
# `LUSolver` in dolfinx may not exist - the code below has never been tested
solver = LUSolver(wcomm, A, 'superlu_dist')
solver.parameters['symmetric'] = True

u = PETScVector(wcomm, height)
solver.solve(u, b)
u = u.gather_on_zero()

if rank == 0:
    np.save('solution.bin', u)
    print(f'OFFLINE STAGE: {t_off:.3}s')
    print(f'ONLINE STAGE: {t_on:.3}s')
    print(f'SEND-RECV: {t_sr:.3}s')
    print(f'GLOBAL COUPLING: {t_gc:.3}s')
