import numpy as np

from dolfin import *
from petsc4py import PETSc

from ..subdiv import triplets
from ..forms import form_MSL, assemble_Ab, unloaded_matrices

comm = MPI.comm_self


class GMsFEUnit:
    """
    n_el        number of elements in a coarse block along one dimension
    n_blocks    number of coarse blocks along one dimension
    """
    def __init__(self, reg_id, n_el, n_blocks):
        tau = 1./n_blocks
        col = reg_id % (n_blocks-1)
        row = reg_id // (n_blocks-1)

        self._midp = (col+1)*tau, (row+1)*tau
        self._overlap = triplets(n_el, n_blocks)
        self._nblocks = n_blocks

        self._pv = lambda x_m: CompiledSubDomain(
                'near(x[0], x_m, tol)', x_m=x_m, tol=1e-12)
        self._ph = lambda y_m: CompiledSubDomain(
                'near(x[1], y_m, tol)', y_m=y_m, tol=1e-12)
        self._bc = lambda x0, x1: CompiledSubDomain(
                'near(x[0], x0, tol) && near(x[1], x1, tol)',
                x0=x0, x1=x1, tol=tau/n_el)

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
        n_eig - number of dominant eigenvalues to be kept
        """
        # petsc4py bug:
        # ```
        # P = PETSc.Mat(comm)
        # P.createDense(Nv.shape, array=Nv)
        # ```
        # - no way to work

        P = PETSc.Mat()
        P.createDense(Nv.shape, array=Nv, comm=comm)
        P.transpose()

        M, S = unloaded_matrices(k, km)
        M = M.mat().PtAP(P)
        S = S.mat().PtAP(P)

        diag = PETSc.Vec(comm)
        diag.createWithArray(eps*np.ones(len(Nv)))
        M.setDiagonal(diag, addv=True)
        S.setDiagonal(diag, addv=True)
        M, S = map(PETScMatrix, [M, S])

        esolver = SLEPcEigenSolver(M, S)
        esolver.parameters["problem_type"] = "gen_hermitian"
        esolver.parameters["spectrum"] = "largest magnitude"
        esolver.parameters["solver"] = "lapack"
        esolver.solve(n_eig)

        w, h = np.empty(n_eig), np.empty([n_eig, len(Nv)])
        for i in range(n_eig): w[i],_,h[i],_ = esolver.get_eigenpair(i)
        return (h[::-1] @ Nv), w

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
        A, _b = assemble_Ab(k, RHS)
        b = PETSc.Vec()
        b.createSeq(len(Nv), comm=comm)

        P = PETSc.Mat()
        P.createDense(Nv.shape, array=Nv, comm=comm)
        P.mult(_b.vec(), b)
        P.transpose()

        A = A.mat().PtAP(P)
        return A, b

    def offdiagonalBlock(self, k_i, Nv_i, Nv_j, pos):
        """
        pos - position of j-th block relative to i-th block
        Returns off-diagonal block `A_ij` in the global matrix `A`
        """
        k, mask_i, mask_j = self._overlap[pos]
        k.vector()[:] = k_i.vector()[mask_i]

        P1 = PETSc.Mat()
        Nv = Nv_i[:, mask_i]
        P1.createDense(Nv.shape, array=Nv, comm=comm)

        P2 = PETSc.Mat()
        Nv = Nv_j[:, mask_j]
        P2.createDense(Nv.shape, array=Nv, comm=comm)

        A,_ = assemble_Ab(k)
        A = P1.matMult(A.mat())
        A = A.matTransposeMult(P2)
        return A
