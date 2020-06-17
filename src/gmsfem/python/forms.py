from dolfin import *

comm = MPI.comm_self


def form_MSL(ks, km=None, f=None):
    V = ks.function_space()
    v = TestFunction(V)
    u = TrialFunction(V)
    forms = []

    forms += [ks*dot(grad(u),grad(v))*dx]
    if km: forms += [km*u*v*dx]
    if f: forms += [f*v*dx]
    return forms


def assemble_Ab(k, f=None):
    A = PETScMatrix(comm)
    b = PETScVector(comm)
    if f: a, L = form_MSL(k, f=f)
    else: a, L = form_MSL(k, Constant(0.))
    assemble(a, A)
    assemble(L, b)
    return A, b


def unloaded_matrices(ks, km=None):
    km = km if km else ks
    a, q = form_MSL(ks, km)

    M = PETScMatrix(comm)
    S = PETScMatrix(comm)
    assemble(a, S)
    assemble(q, M)
    return M, S
