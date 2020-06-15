# Version dependent lines contain:
#  -  `._cpp_object`
#  -  `.cpp_object()`
# This is valid for FEniCS-2019.1.0.
# There is no information about newer versions

import math
import numpy as np
import scipy.linalg

from ufl import *
from dolfin import *


def zero_extrapolate(Nv, V, W, j, i):
    """
    Many times faster (40-300x) than
    equivalent operation with LagrangeInterpolator
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
    # neighbors below
    I = np.arange(N-n)
    pairs = np.column_stack((I, I+n))

    # neighbors on the right
    I = np.arange(N).reshape(n, n)[:, :-1].flatten()
    new_pairs = np.column_stack((I, I+1))
    pairs = np.row_stack((pairs, new_pairs))

    # neighbors on the right, below
    I = np.arange(N-n).reshape(n-1, n)[:, :-1].flatten()
    new_pairs = np.column_stack((I, I+n+1))
    pairs = np.row_stack((pairs, new_pairs))

    # neighbors on the left, below
    I = np.arange(N-n).reshape(n-1, n)[:, 1:].flatten()
    new_pairs = np.column_stack((I, I+n-1))
    pairs = np.row_stack((pairs, new_pairs))

    return pairs


def _setBC(x, src, tol):
    """
    BC in a point 'src' with the tolerance 'tol'
    """
    predicate_1 = near(x[0], src[0], tol)
    predicate_2 = near(x[1], src[1], tol)
    return predicate_1 and predicate_2


class GMsFEUnit:
    """
    n_el        number of elements in a coarse block along one dimension
    n_blocks    number of coarse blocks along one dimension
    """
    def __init__(self, reg_id, n_el, n_blocks, cutils):
        tol = 1./(n_el*n_blocks)
        subd = lambda x_l, x_r, y_l, y_r: (
            CompiledSubDomain(
                '((x[0] >= x_l-tol) && (x[0] <= x_r+tol))'
                '&& ((x[1] >= y_l-tol) && (x[1] <= y_r+tol))',
                x_l=x_l, x_r=x_r, y_l=y_l, y_r=y_r, tol=1e-8));
        rel_id = np.array([0, 1, n_blocks-2, n_blocks-1, n_blocks])

        tau = 1./n_blocks
        col = reg_id % (n_blocks-1)
        row = reg_id // (n_blocks-1)
        x_m, y_m = (col+1)*tau, (row+1)*tau
        x_l, x_r = x_m - tau, x_m + tau
        y_l, y_r = y_m - tau, y_m + tau
        self._midp = x_m, y_m

        self._subd = {}
        self._subd[rel_id[0]] = subd(x_l, x_r, y_l, y_r)
        self._subd[rel_id[1]] = subd(x_m, x_r, y_l, y_r)
        self._subd[rel_id[2]] = subd(x_l, x_m, y_m, y_r)
        self._subd[rel_id[3]] = subd(x_l, x_r, y_m, y_r)
        self._subd[rel_id[4]] = subd(x_m, x_r, y_m, y_r)

        # mirrored overlaps
        self._subd[-rel_id[1]] = subd(x_l, x_m, y_l, y_r)
        self._subd[-rel_id[2]] = subd(x_m, x_r, y_l, y_m)
        self._subd[-rel_id[3]] = subd(x_l, x_r, y_l, y_m)
        self._subd[-rel_id[4]] = subd(x_l, x_m, y_l, y_m)

        self._nblocks = n_blocks
        self._cutils = cutils
        self._tol = tau/n_el

    def snapshotSpace(self, k):
        """
        Returns: Nv - Nodal Values, 2D np.ndarray
        """
        A, b = self._cutils.assemble_Ab(k.cpp_object())

        V = k.function_space()
        bmesh = BoundaryMesh(V.mesh(), 'local')
        Nv = np.empty((bmesh.num_cells(), V.dim()))

        for i, src in enumerate(bmesh.coordinates()):
            single = lambda x: _setBC(x, src, self._tol)
            bc1 = DirichletBC(V, Constant(0.), lambda x,on: on)
            bc2 = DirichletBC(V, Constant(1.), single, 'pointwise')

            _A = A.copy()
            _b = b.copy()
            bc1.apply(_A, _b)
            bc2.apply(_A, _b)

            u = b.copy()
            solve(_A, u, _b)
            Nv[i] = u.get_local()
        return Nv

    def modelReduction(self, k, Nv, n_eig=10, eps=1e-12, km=None):
        """
        n_eig - number of dominant eigenvalues to be kept
        """
        if km is not None:
            M, S = self._cutils.unloaded_matrices(
                    k.cpp_object(), km.cpp_object())
        else:
            M, S = self._cutils.unloaded_matrices(k.cpp_object())

        M = Nv @ M.array() @ Nv.T + eps*np.identity(len(Nv))
        S = Nv @ S.array() @ Nv.T + eps*np.identity(len(Nv))

        which = (len(Nv)-n_eig, len(Nv)-1) if n_eig else None
        w, h = scipy.linalg.eigh(M, S, eigvals=which)
        return (h.T @ Nv), w

    def multiscaleFunctions(self, k, Nv):
        """
        Multiplies the dofs of online functions
        by those of unity partition
        """
        V = k.function_space()
        xi = self.partitionFunction(k)
        return Nv * xi.vector().get_local()

    def partitionFunction(self, k):
        """
        Returns partition unity of function
        for a region specified by `k`
        """
        x_m, y_m = self._midp
        u_mv = Expression(
                '1 - k*fabs(x[1]-y_m)',
                k=self._nblocks, y_m=y_m, degree=1)
        u_mh = Expression(
                '1 - k*fabs(x[0]-x_m)',
                k=self._nblocks, x_m=x_m, degree=1)

        V = k.function_space()
        bc0 = DirichletBC(V, Constant(0.), lambda x,on: on)
        bc1 = DirichletBC(V, u_mv, lambda x: near(x[0], x_m, eps=1e-8))
        bc2 = DirichletBC(V, u_mh, lambda x: near(x[1], y_m, eps=1e-8))

        A, b = self._cutils.assemble_Ab(k.cpp_object())
        for bc in [bc0, bc1, bc2]: bc.apply(A, b)

        xi = Function(V)
        solve(A, xi.vector(), b)
        return xi

    def diagonalBlock(self, k, Nv, RHS):
        """
        Returns diagonal block `A_ii` and vector slice `b_i` in
        the global matrix `A` and load vector `b`, repspectively
        """
        A, b = self._cutils.assemble_Ab(
                k.cpp_object(), RHS.cpp_object())

        A = Nv @ A.array() @ Nv.T
        b = Nv @ b.get_local()
        return A, b

    def offdiagonalBlock(self, k_i, Nv_i, Nv_j, pos):
        """
        pos - position of j-th block relative to i-th block
        Returns off-diagonal block `A_ij` in the global matrix `A`
        """
        V = k_i.function_space()
        subm_i = SubMesh(V.mesh(), self._subd[pos])
        subm_j = SubMesh(V.mesh(), self._subd[-pos])

        v2d = vertex_to_dof_map(V)
        mask_i = v2d[subm_i.data().array('parent_vertex_indices', 0)]
        mask_j = v2d[subm_j.data().array('parent_vertex_indices', 0)]

        subV = FunctionSpace(subm_i, 'P', 1)
        k = Function(subV)

        d2v = dof_to_vertex_map(subV)
        k.vector()[:] = k_i.vector()[mask_i[d2v]]

        A,_ = self._cutils.assemble_Ab(k.cpp_object())
        A = Nv_i[:, mask_i[d2v]] @ A.array() @ Nv_j.T[mask_j[d2v]]
        return A
