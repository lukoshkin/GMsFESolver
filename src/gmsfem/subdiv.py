import time
import math
import numpy as np
from pathlib import Path

from dolfin import *
comm = MPI.comm_self


def zero_extrapolate(Nv, V, W, j, i):
    """
    A faster version of the equivalent
    operation with `LagrangeInterpolator`
    """
    N = W.dim()
    n = int(math.sqrt(N))
    I = np.arange(N).reshape(n, n)

    p = int(math.sqrt(Nv.shape[-1])/2)
    vertices = I[i*p:(i+2)*p+1, j*p:(j+2)*p+1].flatten()

    v2d_V = vertex_to_dof_map(V)
    v2d_W = vertex_to_dof_map(W)
    Nv_W = np.zeros((len(Nv), N))

    Nv_W[:, v2d_W[vertices]] = Nv[:, v2d_V]
    return Nv_W


def overlap_map(n):
    """
    n - number of coarse neighborhoods along 1D

    Returns id pairs of overlapping regions
    """
    N = n*n
    # neighbors above
    I = np.arange(N-n)
    pairs = np.column_stack((I, I+n))

    # neighbors on the right
    I = np.arange(N).reshape(n, n)[:, :-1].flatten()
    new_pairs = np.column_stack((I, I+1))
    pairs = np.row_stack((pairs, new_pairs))

    # neighbors on the right, above
    I = np.arange(N-n).reshape(n-1, n)[:, :-1].flatten()
    new_pairs = np.column_stack((I, I+n+1))
    pairs = np.row_stack((pairs, new_pairs))

    # neighbors on the left, above
    I = np.arange(N-n).reshape(n-1, n)[:, 1:].flatten()
    new_pairs = np.column_stack((I, I+n-1))
    pairs = np.row_stack((pairs, new_pairs))
    return pairs


def _mm1(m, overlap, pos):
    subm_i = SubMesh(m, overlap[pos])
    subm_j = SubMesh(m, overlap[-pos])

    mask_i = subm_i.data().array('parent_vertex_indices', 0)
    mask_j = subm_j.data().array('parent_vertex_indices', 0)
    return subm_i, mask_i, mask_j


def _mm2(m, overlap, pos):
    markers = MeshFunction('size_t', m, 2, 0)
    overlap[pos].mark(markers, 1)
    overlap[-pos].mark(markers, 2)

    subm_i = MeshView.create(markers, 1)
    subm_j = MeshView.create(markers, 2)
    mask_i = subm_i.topology().mapping()[m.id()].vertex_map()
    mask_j = subm_j.topology().mapping()[m.id()].vertex_map()
    return subm_i, mask_i, mask_j


def triplets(n_el, n_blocks):
    subd = lambda x_l, x_r, y_l, y_r: CompiledSubDomain(
            '((x[0] >= x_l-tol) && (x[0] <= x_r+tol))'
            '&& ((x[1] >= y_l-tol) && (x[1] <= y_r+tol))',
            x_l=x_l, x_r=x_r, y_l=y_l, y_r=y_r, tol=1e-12)
    tau = 1./n_blocks
    x_m, y_m = tau, tau
    x_l, x_r = 0, 2*tau
    y_l, y_r = 0, 2*tau
    overlap = {}

    rel_id = [1, n_blocks-2, n_blocks-1, n_blocks]
    overlap[rel_id[0]] = subd(x_m, x_r, y_l, y_r)
    overlap[rel_id[1]] = subd(x_l, x_m, y_m, y_r)
    overlap[rel_id[2]] = subd(x_l, x_r, y_m, y_r)
    overlap[rel_id[3]] = subd(x_m, x_r, y_m, y_r)

    # mirrored overlaps
    overlap[-rel_id[0]] = subd(x_l, x_m, y_l, y_r)
    overlap[-rel_id[1]] = subd(x_m, x_r, y_l, y_m)
    overlap[-rel_id[2]] = subd(x_l, x_r, y_l, y_m)
    overlap[-rel_id[3]] = subd(x_l, x_m, y_l, y_m)

    m = UnitSquareMesh(comm, 2*n_el, 2*n_el)
    m.scale(2./n_blocks)

    V = FunctionSpace(m, 'P', 1)
    v2d = vertex_to_dof_map(V)
    struct = {}

    base = 'mesh'
    ext = 'xdmf'
    for pos in rel_id:
        subm_i, mask_i, mask_j = _mm1(m, overlap, pos)
        subV = FunctionSpace(subm_i, 'P', 1)
        d2v = dof_to_vertex_map(subV)
        mask_i = v2d[mask_i][d2v]
        mask_j = v2d[mask_j][d2v]
       
        # SubMesh and MeshView's bug:
        # The functions' call spawns a mesh instance
        # with MPI_COMM_WORLD communicator instead of
        # the one associated with the original mesh.
        # So, the following workaround is used:

        fname = f'{base}{pos}.{ext}'
        if not Path(fname).exists():
            with XDMFFile(comm, fname) as fp:
                fp.write(subm_i)
        mesh = Mesh(comm)
        with XDMFFile(comm, fname) as fp:
            fp.read(mesh)

        # Though it does not raise an error,
        # it unreliable way and is only used
        # to perform time measurements

        subV = FunctionSpace(mesh, 'P', 1)
        k = Function(subV)
        struct[pos] = k, mask_i, mask_j
    return struct 
