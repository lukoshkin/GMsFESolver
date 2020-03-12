import numpy as np
import scipy.linalg

from ufl import *
from dolfin import *


def setBC(x, src, tol):
    predicate_1 = near(x[0], src[0], tol)
    predicate_2 = near(x[1], src[1], tol)
    return predicate_1 and predicate_2

on_boundary = lambda x, on: on
build = np.vectorize(assemble, [float])


class SpaceBlocks:
    """
    n_el        number of elements in a coarse block along one dimension
    n_blocks    number of coarse blocks along one dimension
    """
    def __init__(self, kappa, reg_id, n_el, n_blocks, comm=MPI.comm_world):
        self.N_c = (n_blocks-1)*(n_blocks-1)

        self._n_c = int(self.N_c**.5)
        self._tol = 1./(n_el * n_blocks)
        self._U, self._DU = [], []

        i = reg_id // (n_blocks-1)
        j = reg_id % (n_blocks-1)
        mesh = UnitSquareMesh(comm, 2*n_el, 2*n_el)
        mesh.translate(Point(j*.5, i*.5))
        mesh.scale(2./n_blocks)
        self.V = FunctionSpace(mesh,'P',1)

    def buildSnapshotSpace(self, K):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        b = assemble(Constant(0.)*v*dx)
        A = assemble(K*dot(grad(u), grad(v))*dx)

        psi_snap = []
        bmesh = BoundaryMesh(self.V.mesh(), 'local')
        for src in bmesh.coordinates():
            single = lambda x: setBC(x, src, self._tol)
            bc1 = DirichletBC(self.V, Constant(0.), on_boundary)
            bc2 = DirichletBC(self.V, Constant(1.), single, 'pointwise')

            _A = A.copy()
            _b = b.copy()
            bc1.apply(_A, _b)
            bc2.apply(_A, _b)

            u = Function(self.V)
            solve(_A, u.vector(), _b)
            psi_snap.append(u)

        return psi_snap

    def buildOfflineSpace(self, avg_K, psi_snap, M_off=10):
        """
        M_off       number of eigenvalues in the offline space
        """
        avg_K = project(avg_K, self.V)
        A = build(avg_K * np.outer(psi_snap,psi_snap) * dx)
        form = lambda u,v: assemble(avg_K*dot(grad(u),grad(v))*dx)
        S = np.vectorize(form, [float])(psi_snap[:,None], psi_snap[None])

        N = len(A)
        which = (N-M_off, N-1) if M_off else None
        w, v = scipy.linalg.eigh(A, S, eigvals=which)
        psi_off = np.dot(psi_snap, v)
        return psi_off, w

    def buildOnlineSpace(self, K, psi_off, M_on=10):
        """
        M_on        number of eigenvalues in the online space
        """
        build = np.vectorize(assemble, [float])
        A = build(K * np.outer(psi_off,psi_off) * dx)
        form = lambda u,v: assemble(K*dot(grad(u),grad(v))*dx)
        S = np.vectorize(form, [float])(psi_off[:,None], psi_off[None])

        N = len(A)
        which = (N-M_on, N-1) if M_on else None
        w, v = scipy.linalg.eigh(A, S, eigvals=which)
        psi_ms = np.dot(psi_off.flatten(), v)
        return psi_ms, w









class SeparateClassForIt:
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
        M = self.M_on
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
