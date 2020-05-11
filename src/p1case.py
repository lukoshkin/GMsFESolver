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
        def middle_point(reg_id):
            col = reg_id % (n_blocks-1)
            row = reg_id // (n_blocks-1)
            x_m, y_m = (col+1)*tau, (row+1)*tau
            return np.column_stack([x_m, y_m])

        rel_id = np.array([0, 1, n_blocks-2, n_blocks-1, n_blocks])
        abs_id = reg_id + rel_id

        self._midp = dict(zip(rel_id, middle_point(abs_id)))
        x_m, y_m = self._midp[0]
        x_l, x_r = x_m - tau, x_m + tau
        y_l, y_r = y_m - tau, y_m + tau

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
        self._subd[-rel_id[4]] = subd(x_m, x_r, y_l, y_m)

        self._nblocks = n_blocks
        self._cutils = cutils
        self._tol = tau/n_el
        self._nel = n_el

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

    def partitionFunction(self, k, x_m, y_m):
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
        xi = self.partitionFunction(k, *self._midp[0])

        #A, b = self._cutils.diagonal_coupling(
        #    k.cpp_object(), xi.cpp_object(), RHS.cpp_object())

        A, b = self._cutils.assemble_Ab(
                k.cpp_object(), RHS.cpp_object())
        A = Nv @ A.array() @ Nv.T
        b = Nv @ b.get_local()
        return A, b

    def offdiagonalBlock(self, k_i, Nv_i, Nv_j, pos):
        """
        pos - overlap position of two adjacent regions
              from the view point of the one with higher id

        Returns A_ij
        """
        V = k_i.function_space()
        subm_i = SubMesh(V.mesh(), self._subd[pos])
        subm_j = SubMesh(V.mesh(), self._subd[-pos])

        mask_i = vertex_to_dof_map(V)[
                subm_i.data().array('parent_vertex_indices', 0)]
        mask_j = vertex_to_dof_map(V)[
                subm_j.data().array('parent_vertex_indices', 0)]

        subV = FunctionSpace(subm_i, 'P', 1)
        k = Function(subV)
        #k = project(k_i, subV)

        #Nv_subi = []
        #for v_dofs in Nv_i:
        #    v = Function(V)
        #    v.vector().set_local(v_dofs)
        #    v = project(v, subV)
        #    vn_dofs = v.vector().get_local()
        #    Nv_subi.append(vn_dofs)
        #Nv_subi = np.array(Nv_subi)

        #mesh = UnitSquareMesh(2*self._nel, 2*self._nel)
        #mesh.scale(2./self._nblocks)
        #leftcorner = self._midp[pos] - 1./self._nblocks
        #mesh.translate(Point(*leftcorner))
        #Va = FunctionSpace(mesh, 'P', 1) 
        ##return V, Va, subV

        #Nv_subj = []
        #for i, v_dofs in enumerate(Nv_j):
        #    v = Function(Va)
        #    v.vector().set_local(v_dofs)
        #    v = project(v, subV)
        #    vn_dofs = v.vector().get_local()
        #    Nv_subj.append(vn_dofs)
        #Nv_subj = np.array(Nv_subj)

        k.vector()[vertex_to_dof_map(subV)] = k_i.vector()[mask_i]
        d2v = dof_to_vertex_map(subV)

        xi_i = self.partitionFunction(k, *self._midp[0])
        xi_j = self.partitionFunction(k, *self._midp[pos])

        #A,_ = self._cutils.assemble_Ab(k.cpp_object())  #del
        A = self._cutils.offdiagonal_coupling(
                k.cpp_object(), xi_i.cpp_object(), xi_j.cpp_object())
        #A = Nv_subi @ A.array() @ Nv_subj.T

        A = Nv_i[:, mask_i][:, d2v] @ A.array() @ Nv_j.T[mask_j][d2v]
        return A
