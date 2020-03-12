import numpy as np
from dolfin import *


def set_kappa(eta, mask, V):
    v2d = vertex_to_dof_map(V)
    dofs, = mask.flatten().nonzero()

    kappa = Function(V)
    kappa.vector()[:] = 1.
    kappa.vector()[v2d[dofs]] = eta
    return kappa

def generate_mask(n_strips, N_el, gap, seed=None):
    if seed is not None:
        np.random.seed(seed)
    side = N_el + 1 - 2 * gap 
    n_ver = np.random.binomial(n_strips, p=.5)
    ends = [None] * 2
    mask = np.zeros((N_el+1, N_el+1), dtype='bool')
    for pos, n_lines in zip((0, 1), (n_ver, n_strips - n_ver)):
        ends[0] = np.random.choice(side, n_lines, replace=False) + gap 
        ends[1] = np.random.choice(side, n_lines, replace=False) + gap 
        lengths = np.ceil(
            np.random.rand(n_lines) * (side+gap-ends[pos])).astype('int')
        for l, x1, x2 in zip(lengths, *ends):
            # >> vertical lines
            if pos: mask[x2:x2+l, x1] = True
            # >> horizontal lines
            else: mask[x2, x1:x1+l] = True

    return mask
