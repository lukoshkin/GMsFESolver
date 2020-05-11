# Version dependent lines contain:
#  -  `._cpp_object`
#  -  `.cpp_object()`
# This is valid for FEniCS-2019.1.0.
# There is no information about newer versions

import numpy as np
import scipy.linalg

from ufl import *
from dolfin import *


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
        subd = lambda x_l, x_r, y_l, y_r: (
            CompiledSubDomain(
                '((x[0] >= x_l-tol) && (x[0] <= x_r+tol))'
                '&& ((x[1] >= y_l-tol) && (x[1] <= y_r+tol))',
                x_l=x_l, x_r=x_r, y_l=y_l, y_r=y_r, tol=1e-12));

        tau = 1./n_blocks
        col = reg_id % (n_blocks-1)
        row = reg_id // (n_blocks-1)
        x_m, y_m = (col+1)*tau, (row+1)*tau
        x_l, x_r = x_m - tau, x_m + tau
        y_l, y_r = y_m - tau, y_m + tau
        self._midp = (x_m, y_m)

        self._subd = {}
        self._subd[0] = subd(x_l, x_r, y_l, y_r)
        self._subd[1] = subd(x_m, x_r, y_l, y_r)
        self._subd[n_blocks-2] = subd(x_l, x_m, y_m, y_r)
        self._subd[n_blocks-1] = subd(x_l, x_r, y_m, y_r)
        self._subd[n_blocks] = subd(x_m, x_r, y_m, y_r)

        self._nblocks = n_blocks
        self._cutils = cutils
        self._tol = tau/n_el

    def snapshotSpace(self, k):
        """
        Nv - Nodal values
        """
        V = k.function_space()
        bmesh = BoundaryMesh(V.mesh(), 'local')
        Nv = np.empty((bmesh.num_cells(), V.dim()))
        A, b = self._cutils.assemble_Ab(k.cpp_object())

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

    def modelReduction(self, k, Nv, n_eig=10):
        """
        n_eig - number of dominant eigenvalues to be kept
        """
        M, S = self._cutils.unloaded_matrices(k.cpp_object())
        M = Nv @ M.array() @ Nv.T
        S = Nv @ S.array() @ Nv.T

        which = (len(Nv)-n_eig, len(Nv)-1) if n_eig else None
        w, h = scipy.linalg.eigh(M, S, eigvals=which)
        return (h.T @ Nv), w

    def multiscaleFunctions(self, k, Nv, W):
        """
        Returns dofs of multiscale basis in the form: list[list[double]]
        """
        V = k.function_space()
        xi = self.partitionFunction(k)
        Nv = self._cutils.multiply_project(xi.cpp_object(), Nv)
        Nv = self._cutils.zero_extrapolation(
                Nv, V._cpp_object, W._cpp_object)

        return np.array(Nv)

    def partitionFunction(self, k):
        x_m, y_m = self._midp
        u_mv = Expression(
                '1 - b*fabs(x[1]-y_m)',
                b=self._nblocks, y_m=y_m, degree=1)
        u_mh = Expression(
                '1 - b*fabs(x[0]-x_m)',
                b=self._nblocks, x_m=x_m, degree=1)

        V = k.function_space()
        bc1 = DirichletBC(V, u_mv, lambda x: near(x[0], x_m, eps=1e-8))
        bc2 = DirichletBC(V, u_mh, lambda x: near(x[1], y_m, eps=1e-8))
        bc3 = DirichletBC(V, Constant(0.), lambda x,on: on)

        A, b = self._cutils.assemble_Ab(k.cpp_object())
        for bc in [bc1, bc2, bc3]: bc.apply(A, b)

        xi = Function(V)
        solve(A, xi.vector(), b)
        return xi

    def diagonalBlock(self, K, Nv, RHS):
        """
        pos - position of j-th block relative to i-th block
        """
        fine_mesh = K.function_space().mesh()
        markers = MeshFunction('size_t', fine_mesh, 2, 0) 
        self._subd[0].mark(markers, 1)

        A, b = self._cutils.integral_assembling(
                K.cpp_object(), Nv, RHS.cpp_object(), markers)
        ij_lower = np.tril_indices(len(Nv), -1)
        A[ij_lower] = A.T[ij_lower]
        return A, b

    def offdiagonalBlock(self, K, Nv_i, Nv_j, pos):
        """
        pos - position of j-th block relative to i-th block
        """
        fine_mesh = K.function_space().mesh()
        markers = MeshFunction('size_t', fine_mesh, 2, 0) 
        self._subd[pos].mark(markers, 1)

        A = self._cutils.stiffness_integral_matrix(
                K.cpp_object(), Nv_i, Nv_j, markers)
        return A
