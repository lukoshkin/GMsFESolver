import numpy as np

from functools import partial
from mpi4py import MPI

from core_components import *
from optimal_distribution import ColoredPartition
                

class GMsFESolver:
    def __init__(
            self, K_mu, mu=None, n_el=16, n_blocks=4, M_off=16, M_on=8):
        """
        Parameters:
            K_mu : list[Function] / Expression
                hyperparameter-dependent permeability coefficient

            n_el : int
                number of cells in a coarse block along 1D 

            n_blocks : int
                number of coarse blocks along 1D

            M_off : int
                dimensionality of the offline space

            M_on : int
                dimensionality of the online space
        """
        if mu is None:
            n_colors = len(K_mu)
            pass
        else:
            n_colors = len(mu)
            fine_mesh = UnitSquareMesh(n_el, n_el)
            W = FunctionSpace(mesh, 'P', 1)
            pass

        self.gcomm = MPI.COMM_WORLD
        self._rank = self.comm.Get_rank()
        self._size = self.comm.Get_size()

        self.cp = ColoredPartition(self._size, n_colors)
        n_comms,_ = self.cp.partition((n_blocks-1)*(n_blocks-1))

        ggroup = self.gcomm.Get_group()
        active = self.gcomm.Create_group(
                ggroup.Incl(np.arange(n_comms*n_colors)))
        self.lcomm = MPI.Comm.Split(
                active, self._rank//n_colors, self._rank)

        rmap = self.cp.map['r'][self._rank]
        cmap = self.cp.map['c'][self._rank]
        self.K, self.avg_K = project_kappa(K_mu, cmap, rmap)

        self._cpty = np.unique(rmap[~np.isnan(rmap)])
        self.vdim = (2*n_el+1)*(2*n_el+1)
        self.snap_dim = 8*n_el

        with open('cutils/pymodule2.cpp', 'r') as fp:
            cutils = compile_cpp_code(fp.read(), include_dirs=['cutils'])

        pairs = overlap_map(n_blocks-1)
        imask = (pairs[:, 0] == reg_id)
        omask = (pairs[:, 1] == reg_id)
        self.st = pairs[omask][:, 0]
        self.rf = pairs[imask][:, 1]

        self.cores = {}
        for r in rank_map:
            if r is not None and r not in cores.keys():
                self.cores[r] = GMsFEUnit(reg_id, n_el, n_blocks, cutils)

    def buildOnlineSpace(self):
        nodebuf, sendbuf = [], []
        for c, r, f in self.cp.getZip(self._rank):
            Nv_snap = self.cores[r].snapshotSpace(self.K[c,r])
            if f: sendbuf.append(Nv_snap)
            else: nodebuf.append(Nv_snap)

        recvbuf = None
        if (self.lcomm != MPI.COMM_NULL
                and self.lcomm.Get_rank() == 0):
            pack_size = cp.countMSGTransfers()[self._rank//self.cp.dimC]
            recvbuf = np.empty([pack_size, self.snap_dim, self.vdim])

        if kcomm != MPI.COMM_NULL:
            self.lcomm.Gather(np.array(sendbuf), recvbuf, root=0)
            Nv = np.concatenate(nodebuf, recvbuf) 
            Nv = Nv.reshape(-1, self.cp.dimC, *Nv.shape[1:])
            return Nv
