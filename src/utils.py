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
    for i in range(len(eta)):
        n_strips = np.random.randint(rho//3, rho)
        mask_i = generate_mask(n_strips, N_el, gap, i+1)
        mask[mask_i>0] = mask_i[mask_i>0]

    for i in range(len(eta)):
        K.append(set_kappa(eta[i], mask, V, i+1))

    return K


def set_kappa(eta, mask, V, tag=1):
    """
    eta     float
    mask    boolean 2D array
    V       FunctionSpace
    """
    v2d = vertex_to_dof_map(V)
    dofs = mask.flatten()==tag

    kappa = Function(V)
    kappa.vector()[:] = 1.
    kappa.vector()[v2d[dofs]] = eta
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
