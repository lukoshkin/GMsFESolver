import numpy as np
import scipy.linalg

from ufl import *
from dolfin import *


def setBC(x, src, tol):
    predicate_1 = near(x[0], src[0], tol)
    predicate_2 = near(x[1], src[1], tol)
    return predicate_1 and predicate_2

build = np.vectorize(assemble, [float])


def find_block_ids(N_c, n_c):
    """
    Returns the array of pairs `(row, column)`,
    where `(row, column)` is the coordinates
    of top-left corner of a block with values
    inside the matrix resulted during `globalCoupling`
    """
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
    return pairs


class SpaceBlocks:
    """
    n_el        number of elements in a coarse block along one dimension
    n_blocks    number of coarse blocks along one dimension
    """
    def __init__(self, reg_id, n_el, n_blocks, pairs, comm=MPI.comm_world):
        self._tol = 1./(n_el * n_blocks)

        col = reg_id % (n_blocks-1)
        row = reg_id // (n_blocks-1)
        self.reg_id = col, row

        mesh = UnitSquareMesh(comm, 2*n_el, 2*n_el)
        mesh.translate(Point(col*.5, row*.5))
        mesh.scale(2./n_blocks)

        self.V = FunctionSpace(mesh,'P',1)

        n_c = n_blocks - 1  # num. of coarse NBHs along 1D
        N_c = n_c*n_c       # total

        n_f = n_el*n_blocks + 1     # fine mesh size along 1D
        N_f = n_f*n_f               # its full discretization size

        self.pairs = pairs
        self.struct_info = {'NBH union': (N_c, n_c),
                            'coarse block': n_el,
                            'fine mesh': (N_f, n_f)}


    def _fillExtrapolate(self, f, W, fill_value=0):
        """
        Extrapolate function `f` defined on `reg_id`
        with `fill_value` to function space `W` built
        on fine mesh
        """
        j, i = self.reg_id
        N, n = self.struct_info['fine mesh']
        p = self.struct_info['coarse block']

        I = np.arange(N).reshape(n, n)
        vertices = I[i*p:(i+2)*p+1, j*p:(j+2)*p+1].flatten()

        F = Function(W)
        F.vector()[:] = fill_value
        v2d = vertex_to_dof_map(W)
        F.vector()[v2d[vertices]] = f.compute_vertex_values()
        return F

    def _restrictToSubdomain(self, F):
        """
        Restricts function `F`defined on fine mesh
        to subdomain `self.V`
        """
        j, i = self.reg_id
        _, n = self.struct_info['fine mesh']
        p = self.struct_info['coarse block']
        subarea = F.compute_vertex_values().reshape(n, n)

        f = Function(self.V)
        v2d = vertex_to_dof_map(self.V)
        f.vector()[v2d] = subarea[i*p:(i+2)*p+1, j*p:(j+2)*p+1].flatten()
        return f

    def buildSnapshotSpace(self, K):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        b = assemble(Constant(0.)*v*dx)
        K = self._restrictToSubdomain(K)
        A = assemble(K*dot(grad(u), grad(v))*dx)
        A_ret = assemble(dot(grad(u), grad(v))*dx)
        bc1.apply(A_ret)
        bc2.apply(A_ret)

        psi_snap = []
        bmesh = BoundaryMesh(self.V.mesh(), 'local')
        for src in bmesh.coordinates():
            single = lambda x: setBC(x, src, self._tol)
            bc1 = DirichletBC(self.V, Constant(0.), lambda x,on: on)
            bc2 = DirichletBC(self.V, Constant(1.), single, 'pointwise')

            _A = A.copy()
            _b = b.copy()
            bc1.apply(_A, _b)
            bc2.apply(_A, _b)

            u = Function(self.V)
            solve(_A, u.vector(), _b)
            psi_snap.append(u)

        C = np.empty((len(psi_snap), self.V.dim()))
        for i, psi in enumerate(psi_snap):
            C[i] = psi.compute_vertex_values()
            
        return psi_snap, C, A_ret.array()

    def buildOfflineSpace(self, avg_K, psi_snap, M_off=10):
        """
        M_off       number of eigenvalues in the offline space
        """
        avg_K = self._restrictToSubdomain(avg_K)
        A = build(np.outer(psi_snap,psi_snap) * dx)
        form = lambda u,v: assemble(avg_K*dot(grad(u),grad(v))*dx)
        S = np.vectorize(form, [float])(psi_snap[:,None], psi_snap[None])
        return A, S

        N = len(A)
        which = (N-M_off, N-1) if M_off else None
        w, v = scipy.linalg.eigh(A, S, eigvals=which)
        psi_off = np.dot(psi_snap, v)
        return psi_off, w

    def buildOnlineSpace(self, K, psi_off, M_on=10):
        """
        M_on        number of eigenvalues in the online space
        """
        K = extract_subdomain(K, self.V)
        A = build(K * np.outer(psi_off,psi_off) * dx)
        form = lambda u,v: assemble(K*dot(grad(u),grad(v))*dx)
        S = np.vectorize(form, [float])(psi_off[:,None], psi_off[None])

        N = len(A)
        which = (N-M_on, N-1) if M_on else None
        w, v = scipy.linalg.eigh(A, S, eigvals=which)
        psi_ms = np.dot(psi_off.flatten(), v)
        xi = self._createPartitionFunction(K)
        # define xi by psi_ms product
        return psi_ms, w

    def _createPartitionFunction(self, K):
        x_m, y_m = map(lambda x: .25*(x+1), self.reg_id)
        u_mv = Expression('1 - 4*fabs(x[1]-y_m)', y_m=y_m, degree=1)
        u_mh = Expression('1 - 4*fabs(x[0]-x_m)', x_m=x_m, degree=1)

        tol = 1e-8
        def boundary_Mv(x):
            return near(x[0], x_m, tol)

        def boundary_Mh(x):
            return near(x[1], y_m, tol)

        bc1 = DirichletBC(self.V, u_mv, boundary_Mv)
        bc2 = DirichletBC(self.V, u_mh, boundary_Mh)
        bc3 = DirichletBC(self.V, Constant(0.), lambda x,on: on)
        bcs = [bc1, bc2, bc3]

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        L = Constant(0.) * v * dx
        a = K * dot(grad(u), grad(v)) * dx

        xi = Function(self.V)
        solve(a==L, xi, bcs)
        return xi


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

    def globalCoupling(self, K, RHS, psi_ms):
        M = psi_ms.shape[1]
        N_c,_ = self.struct_info['NBH union']

        b = np.zeros(N_c*M)
        A = np.zeros((N_c*M, N_c*M))

        for i in range(self.N_c):
            chi = self._overlap(i, i)
            dx = Measure('dx', domain=self.W, subdomain_data=chi)

            form = lambda u,v: assemble(K*dot(grad(u),grad(v))*dx(1))
            A_ii = np.vectorize(form, [float])(psi[i,:,None], psi[i,None])
            A[i*M:(i+1)*M, i*M:(i+1)*M] = A_ii

            b_i = np.vectorize(assemble, [float])(RHS*psi[i]*dx(1))
            b[i*M:(i+1)*M] = b_i

        for i, j in pairs:
            chi = self._overlap(i, j)
            dx = Measure('dx', domain=self.W, subdomain_data=chi)

            form = lambda u,v: assemble(K*dot(grad(u),grad(v))*dx(1))
            A_ij = np.vectorize(form, [float])(psi[i,:,None], psi[j,None])
            A[i*M:(i+1)*M, j*M:(j+1)*M] = A_ij

        i_lower = np.tril_indices(len(A), -1)
        A[i_lower] = A.T[i_lower]

        u = scipy.linalg.solve(A, b, assume_a='pos')
        return project((u * psi.flatten()).sum(), self.W)

