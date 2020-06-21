import numpy as np
import scipy.linalg

from ..forms import *


class GMsFEUnit:
    """
    n_el        number of elements in a coarse block along 1D
    n_blocks    number of coarse blocks along one of dimensions
    """
    def __init__(self, reg_id, n_el, n_blocks):
        self._nblocks = n_blocks
        subd = lambda x_l, x_r, y_l, y_r: CompiledSubDomain(
                '((x[0] >= x_l-tol) && (x[0] <= x_r+tol))'
                '&& ((x[1] >= y_l-tol) && (x[1] <= y_r+tol))',
                x_l=x_l, x_r=x_r, y_l=y_l, y_r=y_r, tol=1e-8)
        tau = 1./n_blocks
        self._bc = lambda x0, x1: CompiledSubDomain(
                'near(x[0], x0, tol) && near(x[1], x1, tol)',
                x0=x0, x1=x1, tol=tau/n_el)
        self._pv = lambda x_m: CompiledSubDomain(
                'near(x[0], x_m, tol)', x_m=x_m, tol=1e-8)
        self._ph = lambda y_m: CompiledSubDomain(
                'near(x[1], y_m, tol)', y_m=y_m, tol=1e-8)

        col = reg_id % (n_blocks-1)
        row = reg_id // (n_blocks-1)
        x_m, y_m = (col+1)*tau, (row+1)*tau
        x_l, x_r = x_m - tau, x_m + tau
        y_l, y_r = y_m - tau, y_m + tau
        self._midp = x_m, y_m

        self._subd = {}
        rel_id = [1, n_blocks-2, n_blocks-1, n_blocks]
        self._subd[rel_id[0]] = subd(x_m, x_r, y_l, y_r)
        self._subd[rel_id[1]] = subd(x_l, x_m, y_m, y_r)
        self._subd[rel_id[2]] = subd(x_l, x_r, y_m, y_r)
        self._subd[rel_id[3]] = subd(x_m, x_r, y_m, y_r)

        # mirrored overlaps
        self._subd[-rel_id[0]] = subd(x_l, x_m, y_l, y_r)
        self._subd[-rel_id[1]] = subd(x_m, x_r, y_l, y_m)
        self._subd[-rel_id[2]] = subd(x_l, x_r, y_l, y_m)
        self._subd[-rel_id[3]] = subd(x_l, x_m, y_l, y_m)

    def snapshotSpace(self, k):
        """
        Returns: Nv - Nodal Values, 2D np.ndarray
        """
        A, b = assemble_Ab(k)
        V = k.function_space()
        bmesh = BoundaryMesh(V.mesh(), 'local')
        Nv = np.empty((bmesh.num_cells(), V.dim()))
        bc0 = DirichletBC(V, Constant(0.), 'on_boundary')
        solver = KrylovSolver('bicgstab', 'ilu')
        for i, src in enumerate(bmesh.coordinates()):
            bc1 = DirichletBC(
                V, Constant(1.), self._bc(*src), 'pointwise')

            _A = A.copy()
            _b = b.copy()
            bc0.apply(_A, _b)
            bc1.apply(_A, _b)

            u = b.copy()
            solver.solve(_A, u, _b)
            Nv[i] = u.get_local()
        return Nv

    def modelReduction(self, k, Nv, n_eig=10, eps=1e-12, km=None):
        """
        n_eig - number of dominant eigenvalues to preserve
        """
        M, S = unloaded_matrices(k, km)
        diag = eps*np.identity(len(Nv))
        M = Nv @ M.array() @ Nv.T + diag
        S = Nv @ S.array() @ Nv.T + diag

        which = (len(Nv)-n_eig, len(Nv)-1) if n_eig else None
        w, h = scipy.linalg.eigh(M, S, eigvals=which)
        return (h.T @ Nv), w

    def multiscaleDOFs(self, k, Nv):
        """
        Multiplies the dofs of online functions
        by those of unity partition
        """
        V = k.function_space()
        xi = self.partitionFunction(k)
        return Nv * xi.get_local()

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
        bc0 = DirichletBC(V, Constant(0.), 'on_boundary')
        bc1 = DirichletBC(V, u_mv, self._pv(x_m))
        bc2 = DirichletBC(V, u_mh, self._ph(y_m))

        a, L = form_MSL(k, f=Constant(0.))
        A, b = assemble_system(a, L, [bc0,bc1,bc2])
        solver = KrylovSolver('cg', 'ilu')

        xi = b.copy()
        solver.solve(A, xi, b)
        return xi

    def diagonalBlock(self, k, Nv, RHS):
        """
        Returns diagonal block `A_ii` and vector slice `b_i` in
        the global matrix `A` and load vector `b`, repspectively
        """
        A, b = assemble_Ab(k, RHS)

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

        A,_ = assemble_Ab(k)
        A = Nv_i[:, mask_i[d2v]] @ A.array() @ Nv_j.T[mask_j[d2v]]
        return A
