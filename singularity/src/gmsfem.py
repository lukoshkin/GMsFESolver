import numpy as np
import scipy.linalg
import scipy.sparse

from ufl import *
from dolfin import *
from utils import set_kappa


class MsFEM:
    def __init__(
            self, n_el, n_blocks, mask, rhs,
            eta=1e3, n_eighs=10, **rhs_kwargs):
        N_el = n_el * n_blocks
        self._tol = 1./N_el
        self.M_off = n_eighs
        self.N_c = (n_blocks-1)*(n_blocks-1)
        self._n_c = int(self.N_c**.5)

        self._kappa, self._xi, self.V = [], [], []
        for k in range(self.N_c):
            i = k // (n_blocks-1)
            j = k % (n_blocks-1)
            neighborhood = UnitSquareMesh(2*n_el, 2*n_el)
            neighborhood.translate(Point(j*.5, i*.5))
            neighborhood.scale(2./n_blocks)

            self.V.append(FunctionSpace(neighborhood,'P',1))
            _mask = mask[i*n_el:(i+2)*n_el+1, j*n_el:(j+2)*n_el+1]
            self._kappa.append(set_kappa(eta, _mask, self.V[-1]))
            self._addToPartition(.25*(j+1), .25*(i+1))

        self._U = [[] for _ in range(self.N_c)]
        self._DU = [[] for _ in range(self.N_c)]
        fine_mesh = UnitSquareMesh(N_el, N_el)
        self.W = FunctionSpace(fine_mesh, 'P', 1)
        self._kappa.append(set_kappa(eta, mask, self.W))
        self._rhs = Expression(rhs, **rhs_kwargs, degree=1)
        self._markers = np.empty((self.N_c, self.N_c), 'O')

    def _addToPartition(self, x_m, y_m):
        u_mv = Expression('1 - 4*fabs(x[1]-y_m)', y_m=y_m, degree=1)
        u_mh = Expression('1 - 4*fabs(x[0]-x_m)', x_m=x_m, degree=1)

        tol = 1e-8
        def boundary_Mv(x):
            return near(x[0], x_m, tol)

        def boundary_Mh(x):
            return near(x[1], y_m, tol)

        kappa, V = self._kappa[-1], self.V[-1]
        bc1 = DirichletBC(V, u_mv, boundary_Mv)
        bc2 = DirichletBC(V, u_mh, boundary_Mh)
        bc3 = DirichletBC(V, Constant(0.), lambda x, on: on)
        bcs = [bc1, bc2, bc3]

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * dot(grad(u), grad(v)) * dx
        L = Constant(0.) * v * dx

        u = Function(V)
        solve(a==L, u, bcs)
        self._xi.append(u)

    def buildSnapshotSpace(self):
        def setBC(x, src):
            predicate_1 = near(x[0], src[0], self._tol)
            predicate_2 = near(x[1], src[1], self._tol)
            return predicate_1 and predicate_2

        every = lambda x, on: on
        for k, V in enumerate(self.V):
            u = TrialFunction(V)
            v = TestFunction(V)
            b = assemble(Constant(0.)*v*dx)
            A = assemble(self._kappa[k]*dot(grad(u), grad(v))*dx)

            bmesh = BoundaryMesh(V.mesh(), 'local')
            for src in bmesh.coordinates():
                single = lambda x: setBC(x, src)
                bc1 = DirichletBC(V, Constant(0.), every)
                bc2 = DirichletBC(V, Constant(1.), single, 'pointwise')

                bcs = [bc1, bc2]
                _A = A.copy()
                _b = b.copy()
                for bc in bcs:
                    bc.apply(_A, _b)

                u = Function(V)
                solve(_A, u.vector(), _b)
                self._U[k].append(u)
                self._DU[k].append(grad(u))

    def buildOfflineSpace(self):
        build = np.vectorize(assemble, [float])
        self.psi_ms = []
        for kappa, U, DU  in zip(self._kappa, self._U, self._DU):
            # >> mass matrix
            A = build(kappa * np.outer(U, U) * dx)
            # >> stiffness matrix
            S = build(kappa * np.tensordot(DU, DU, [1, 1]) * dx)

            N = len(A) - 1
            w, v = scipy.linalg.eigh(A, S, eigvals=(N-self.M_off,N))
            self.psi_ms.append(np.dot(U, v))

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

    def globalCoupling(self):
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

        def embed(psi, xi, V):
            embedding = Function(self.W)
            projection = project(psi * xi, V)
            LagrangeInterpolator.interpolate(embedding, projection)
            return embedding 

        kappa = self._kappa[-1]
        psi = np.vectorize(embed, 'O')(
                self.psi_ms, np.array(self._xi)[:, None],
                np.array(self.V)[:, None])

        b = np.zeros(N_c*M)
        A = np.zeros((N_c*M, N_c*M))
        for i in range(N_c):
            chi = self._overlap(i, i)
            dx = Measure('dx', domain=self.W, subdomain_data=chi)
            form = lambda u, v: assemble(kappa*dot(grad(u),grad(v))*dx(1))
            A_ii = np.vectorize(form, [float])(psi[i,:,None], psi[i,None])
            b_i = np.vectorize(assemble, [float])(self._rhs*psi[i]*dx(1))
            A[i*M:(i+1)*M, i*M:(i+1)*M] = A_ii
            b[i*M:(i+1)*M] = b_i

        for i, j in pairs:
            chi = self._overlap(i, j)
            dx = Measure('dx', domain=self.W, subdomain_data=chi)
            form = lambda u, v: assemble(kappa*dot(grad(u),grad(v))*dx(1))
            A_ij = np.vectorize(form, [float])(psi[i,:,None], psi[j,None])
            A[i*M:(i+1)*M, j*M:(j+1)*M] = A_ij

        i_lower = np.tril_indices(len(A), -1)
        A[i_lower] = A.T[i_lower]

        self._psi_ms, self.psi_ms = self.psi_ms, psi
        self.A = A

        u = scipy.linalg.solve(A, b, assume_a='pos')
        return project((u * psi.flatten()).sum(), self.W)

    def getFineMeshSolution(self):
        u = TrialFunction(self.W)
        v = TestFunction(self.W)
        a = self._kappa[-1] * dot(grad(u), grad(v)) * dx
        L = self._rhs * v * dx
        u = Function(self.W)

        bc = DirichletBC(self.W, Constant(0.), lambda x, on: on)
        solve(a==L, u, bc)
        return u
