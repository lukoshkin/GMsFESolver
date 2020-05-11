import math
import numpy as np
from dolfin import *


def get_simple_kappa(
        eta, N_el, gap=2, rho=10,
        comm=MPI.comm_world, seed=None):
    """
    eta     list of floats (eta[i] - value of the i-th permeability func.)
    N_el    number of elements in the mesh along one dim.
    gap     indent from the edges where no strip is drawn
    """
    mesh = UnitSquareMesh(comm, N_el, N_el)
    V = FunctionSpace(mesh, 'P', 1)

    if seed is not None:
        np.random.seed(seed)

    K = []
    mask = np.zeros((N_el+1, N_el+1))

    flag = False
    try:
        iter(eta)
    except TypeError:
        eta = [eta]
        flag = True

    for i in range(len(eta)):
        n_strips = np.random.randint(rho//3, rho)
        mask_i = generate_mask(n_strips, N_el, gap, i+1)
        mask[mask_i>0] = mask_i[mask_i>0]

    for i in range(len(eta)):
        K.append(set_kappa(eta[i], mask, V, i+1))

    if flag: return K[0]
    return K


def set_kappa(eta, mask, V, tag=1):
    """
    eta     float
    mask    boolean 2D array
    V       FunctionSpace
    """
    v2d = vertex_to_dof_map(V)
    vertices = mask.flatten()==tag

    kappa = Function(V)
    kappa.vector()[:] = 1.
    kappa.vector()[v2d[vertices]] = eta
    return kappa


def generate_mask(n_strips, N_el, gap, tag=1, seed=None):
    """
    n_strips    number of strips (vertical and horizontal)
    N_el        number of elements in the mesh along one dim.
    gap         indent from the edges where no strip is drawn
    """
    if seed is not None:
        np.random.seed(seed)
    side = N_el + 1 - 2 * gap 
    n_ver = np.random.binomial(n_strips, p=.5)
    ends = [None] * 2
    mask = np.zeros((N_el+1, N_el+1))
    for pos, n_lines in zip((0, 1), (n_ver, n_strips - n_ver)):
        ends[0] = np.random.choice(side, n_lines, replace=False) + gap 
        ends[1] = np.random.choice(side, n_lines, replace=False) + gap 
        lengths = np.ceil(
            np.random.rand(n_lines) * (side+gap-ends[pos])).astype('int')
        for l, x1, x2 in zip(lengths, *ends):
            # >> vertical lines
            if pos: mask[x2:x2+l, x1] = tag
            # >> horizontal lines
            else: mask[x2, x1:x1+l] = tag
    return mask

def fill_extrapolate(f, V, i, j, fill_value=0):
    """
    Many times faster (40-300x) than
    equivalent operation with LagrangeInterpolator
    """
    N = V.dim()
    n = int(math.sqrt(N))
    I = np.arange(N).reshape(n, n)

    p = int(math.sqrt(f.function_space().dim())/2)
    vertices = I[i*p:(i+2)*p+1, j*p:(j+2)*p+1].flatten()

    F = Function(V)
    F.vector()[:] = fill_value
    v2d = vertex_to_dof_map(V)
    F.vector()[v2d[vertices]] = f.compute_vertex_values()
    return F

def extract_subdomain(F, V, i, j):
    """
    Slightly faster (3-10x) than `project(F, V)`
    """
    p = int(math.sqrt(V.dim())/2)
    n = int(math.sqrt(F.function_space().dim()))
    subarea = F.compute_vertex_values().reshape(n, n)

    f = Function(V)
    v2d = vertex_to_dof_map(V)
    f.vector()[v2d] = subarea[i*p:(i+2)*p+1, j*p:(j+2)*p+1].flatten()
    return f
