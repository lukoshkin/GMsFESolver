import numpy as np
import scipy.linalg
import scipy.sparse

from ufl import *
from dolfin import *
from mpi4py import MPI


class GMsFEM:
    """
    n_el        number of elements in a coarse block along one dimension
    n_blocks    number of coarse blocks along one dimension

    kappa       list of permeability functions: the i-th fn is k(x,u_i)
                discretization over the 2nd argument has to be done manually

    M_off       number of eigenvalues in the offline space
    M_on        number of eigenvalues in the online space
    """
    def __init__(self, n_el, n_blocks, kappa, M_off=10, M_on=10):
        fine_mesh = UnitSquareMesh(n_el*n_blocks, n_el*n_blocks)
        self.W = FunctionSpace(fine_mesh, 'P', 1)

        self.N_c = (n_blocks-1)*(n_blocks-1)
        self.M_off = M_off
        self.M_on = M_on
        self.V = []

        self._n_c = int(self.N_c**.5)
        self._tol = 1./(n_el * n_blocks)

        self._kappa, self._xi = [], []
        for k in range(self.N_c):
            i = k // (n_blocks-1)
            j = k % (n_blocks-1)
            nbh = UnitSquareMesh(2*n_el, 2*n_el)
            nbh.translate(Point(j*.5, i*.5))
            nbh.scale(2./n_blocks)

            self.V.append(FunctionSpace(nbh,'P',1))
            K = np.vectorize(project, 'O')(kappa, self.V[-1])
            self._kappa.append(K)

        self.V, self._kappa = map(np.array, (self.V, self._kappa))
        self._markers = np.empty((self.N_c, self.N_c), 'O')

    def buildSnapshotSpace(self):
        self._U = [[] for _ in range(self.N_c)]
        self._DU = [[] for _ in range(self.N_c)]

        comm = MPI.COMM_WORLD
        prank = comm.Get_rank()
        psize = comm.Get_size()

        # for kappa in self._kappa.T:
        #     self._buildSnapshotSpace(K[i])
        K = self._kappa.T
        for i in range(prank, self.N_c, psize):
            self._buildSnapshotSpace(K[i])

    def _buildSnapshotSpace(self, kappa):
        def setBC(x, src):
            predicate_1 = near(x[0], src[0], self._tol)
            predicate_2 = near(x[1], src[1], self._tol)
            return predicate_1 and predicate_2

        every = lambda x, on: on
        for i, V in enumerate(self.V):
            u = TrialFunction(V)
            v = TestFunction(V)
            b = assemble(Constant(0.)*v*dx)
            A = assemble(kappa[i]*dot(grad(u), grad(v))*dx)

            bmesh = BoundaryMesh(V.mesh(), 'local')
            for src in bmesh.coordinates():
                single = lambda x: setBC(x, src)
                bc1 = DirichletBC(V, Constant(0.), every)
                bc2 = DirichletBC(V, Constant(1.), single, 'pointwise')

                _A = A.copy()
                _b = b.copy()
                bc1.apply(_A, _b)
                bc2.apply(_A, _b)

                u = Function(V)
                solve(_A, u.vector(), _b)

                self._U[i].append(u)
                self._DU[i].append(grad(u))

    def buildOfflineSpace(self):
        build = np.vectorize(assemble, [float])

        def calc_weigths(K):
            return np.sum(K.vector().get_local())

        y = np.vectorize(calc_weigths, [float])(self._kappa)
        K = np.sum(np.array(self._kappa,'O')/y, axis=1)
        K = np.vectorize(project, 'O')(K, self.V)

        self.psi_off = []
        for kappa, U, DU  in zip(K, self._U, self._DU):
            A = build(kappa * np.outer(U,U) * dx)
            S = build(kappa * np.tensordot(DU,DU,[1, 1]) * dx)

            w, v = scipy.linalg.eigh(A, S)
            self.psi_off.append(np.dot(U, v[:, -self.M_off:]))

    def buildOnlineSpace(self):
        self.psi_ms = []
        for kappa, psi, V in zip(self._kappa, self.psi_off, self.V):
            psi,_ = self._buildOnlineSpace(kappa, psi, V)
            self.psi_ms.append(psi)

    def _buildOnlineSpace(self, kappa, psi, V):
        build = np.vectorize(assemble, [float])
        A = build(kappa[:,None,None] * np.outer(psi,psi) * dx)

        form = lambda u,v,k: assemble(k*dot(grad(u),grad(v))*dx)
        S = np.vectorize(form, [float])(
                kappa[:,None,None], psi[:,None], psi[None])

        psi_ms = []
        for i in range(len(A)):
            w, v = scipy.linalg.eigh(A[i], S[i])
            psi = np.dot(psi.flatten(), v[:,-self.M_on:])
            psi_ms.append(psi)
        return psi_ms, w

    def _overlap(self, a, b):
        if self._markers[a, b] is not None: return self._markers[a, b]
        if b - a == self._n_c-1: return self._overlap(a-1, b+1)
        mesh, mesh1, mesh2 = self.W.mesh(), self.V[a].mesh(), self.V[b].mesh()
        V1, V2, V = (FunctionSpace(m, 'DG', 0) for m in (mesh1, mesh2, mesh))
        chi1, chi2 = Function(V), Function(V)

        LagrangeInterpolator.interpolate(chi1, interpolate(Constant(1), V1))
        LagrangeInterpolator.interpolate(chi2, interpolate(Constant(1), V2))
        chi = chi1.vector().get_local() * chi2.vector().get_local()

        markers = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        markers.array()[:] = chi
        self._markers[a, b] = markers
        return markers

    def globalCoupling(self, kappa, rhs):
        # >> shorthands
        M = self.M_off
        N_c = self.N_c
        n_c = self._n_c

        I = np.arange(N_c-n_c)
        pairs = np.column_stack((I, I+n_c))

        I = np.arange(N_c).reshape(n_c, n_c)[:, :-1].flatten()
        new_pairs = np.column_stack((I, I+1))
        pairs = np.row_stack((pairs, new_pairs))

        I = np.arange(N_c-n_c).reshape(n_c-1, n_c)[:, :-1].flatten()
        new_pairs = np.column_stack((I, I+n_c+1))
        pairs = np.row_stack((pairs, new_pairs))

        I = np.arange(N_c-n_c).reshape(n_c-1, n_c)[:, 1:].flatten()
        new_pairs = np.column_stack((I, I+n_c-1))
        pairs = np.row_stack((pairs, new_pairs))

        n_blocks = int(self.N_c**.5)
        for k, V in enumerate(self.V):
            i = k // (n_blocks-1)
            j = k % (n_blocks-1)
            x_m, y_m = .25*(j+1), .25*(i+1)
            self._addToPartition(x_m, y_m, kappa, V)

        psi = np.vectorize(project, 'O')(
                self.psi_ms*np.array(self._xi)[:, None], self.W)

        b = np.zeros(N_c*M)
        A = np.zeros((N_c*M, N_c*M))
        for i in range(N_c):
            chi = self._overlap(i, i)
            dx = Measure('dx', domain=self.W, subdomain_data=chi)

            form = lambda u,v: assemble(kappa*dot(grad(u),grad(v))*dx(1))
            A_ii = np.vectorize(form, [float])(psi[i,:,None], psi[i,None])
            A[i*M:(i+1)*M, i*M:(i+1)*M] = A_ii

            b_i = np.vectorize(assemble, [float])(rhs*psi[i]*dx(1))
            b[i*M:(i+1)*M] = b_i

        for i, j in pairs:
            chi = self._overlap(i, j)
            dx = Measure('dx', domain=self.W, subdomain_data=chi)

            form = lambda u,v: assemble(kappa*dot(grad(u),grad(v))*dx(1))
            A_ij = np.vectorize(form, [float])(psi[i,:,None], psi[j,None])
            A[i*M:(i+1)*M, j*M:(j+1)*M] = A_ij

        i_lower = np.tril_indices(len(A), -1)
        A[i_lower] = A.T[i_lower]

        u = scipy.linalg.solve(A, b, assume_a='pos')
        return project((u * psi.flatten()).sum(), self.W)

    def _addToPartition(self, x_m, y_m, kappa, V):
        u_mv = Expression('1 - 4*fabs(x[1]-y_m)', y_m=y_m, degree=1)
        u_mh = Expression('1 - 4*fabs(x[0]-x_m)', x_m=x_m, degree=1)
        kappa = project(kappa, V)

        tol = 1e-8
        def boundary_Mv(x):
            return near(x[0], x_m, tol)

        def boundary_Mh(x):
            return near(x[1], y_m, tol)

        bc1 = DirichletBC(V, u_mv, boundary_Mv)
        bc2 = DirichletBC(V, u_mh, boundary_Mh)
        bc3 = DirichletBC(V, Constant(0.), lambda x,on: on)
        bcs = [bc1, bc2, bc3]

        u = TrialFunction(V)
        v = TestFunction(V)
        L = Constant(0.) * v * dx
        a = kappa * dot(grad(u), grad(v)) * dx

        u = Function(V)
        solve(a==L, u, bcs)
        self._xi.append(u)

    def getFineMeshSolution(self, kappa, rhs):
        kappa = project(kappa, self.W)
        rhs = project(rhs, self.W)
        u = TrialFunction(self.W)
        v = TestFunction(self.W)
        a = kappa * dot(grad(u), grad(v)) * dx
        L = rhs * v * dx

        bc = DirichletBC(self.W, Constant(0.), lambda x, on: on)
        u = Function(self.W)
        solve(a==L, u, bc)
        return u
