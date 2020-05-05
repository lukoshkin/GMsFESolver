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
    def __init__(self, middle_point, subdomains, cutils):
        self._subd = subdomains
        self._cutils = cutils
        self._middle_point = middle_point

    def snapshotSpace(self, k):
        """
        Nv - Nodal values
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

    def multiscaleFunctionsDofs(self, k, Nv, W):
        """
        Returns dofs of multiscale basis in the form: list[list[double]]
        """
        xi = self.partitionFunction(k)
        return self._cutils.project_embed(
                Nv, xi.cpp_object(), W._cpp_object)

    def partitionFunction(self, k):
        x_m, y_m = self._middle_point
        u_mv = Expression('1 - 4*fabs(x[1]-y_m)', y_m=y_m, degree=1)
        u_mh = Expression('1 - 4*fabs(x[0]-x_m)', x_m=x_m, degree=1)

        V = k.function_space()
        bc1 = DirichletBC(V, u_mv, lambda x: near(x[0], x_m, eps=1e-8))
        bc2 = DirichletBC(V, u_mh, lambda x: near(x[1], y_m, eps=1e-8))
        bc3 = DirichletBC(V, Constant(0.), lambda x,on: on)

        A, b = self._cutils.assemble_Ab(k.cpp_object())
        for bc in [bc1, bc2, bc3]: bc.apply(A, b)

        xi = Function(V)
        solve(A, xi.vector(), b)
        return xi

    def diagonalBlock(self, k, Nv, RHS):
        xi = self.partitionFunction(k)
        A, b = self._cutils.diagonal_coupling(
                k.cpp_object(), xi.cpp_object(), RHS.cpp_object())

        A = Nv @ A.array() @ Nv.T
        b = Nv @ b.get_local()
        return A, b

    def offdiagonalBlock(self, K, Psi_i, Psi_j, pos):
        """
        pos - overlap position of two adjacent regions
              from the view point of the one with higher id
        """
        fine_mesh = K.function_space().mesh()
        markers = MeshFunction('size_t', fine_mesh, 2, 0) 
        self._subd[pos].mark(markers, 1)

        A = self._cutils.stiffness_integral_matrix(
                K[-1].cpp_object(), Psi_i, Psi_j, markers)
        ij_lower = np.tril_indices(len(Psi_i), -1)
        A[ij_lower] = A.T[ij_lower]
        return A
