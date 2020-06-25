import math
import numpy as np
from warnings import warn


def decompose(N, m, b=8):
    """
    Decomposes N into m terms: a_i, i=1,m;
    so that: max_i a_i - min_i a_i <= 1 applies
    --------
    b is a number of bits used for an integer
    in the output array
    """
    terms = np.empty(m, f'i{b}')
    terms[:] = math.floor(N/m)
    terms[:N-terms.sum()] += 1
    return terms 


class ColoredPartition:
    """
    Solves the distribution problem of jar tares
    inside a large container. It is equivalent to 
    unimodal Tetris game with one additional condition:
    it is better to put tares upright
    """
    def __init__(self, size, n_colors):
        """
        Args:
        ----
        size        'bottom' size of the large container
        n_colors    size of tares/batches 
        """
        self.dimC = n_colors
        self.size = size                   # number of all processes
        self.map = {}                      # rank map

        self._ncom = size//n_colors        # max. number of comm-s (per row)
        self._size = self._ncom*n_colors   # number of active processes
        if size == 1: warn(
            'Better to use just for-loop'
            '`for r in range(rank, n_tasks, size): YourCode`')

    def getZip(self, rank, comm_flags=True):
        """
        Returns an iterator over the pairs:
        color - 1st pos, and region - 2nd.
        All `None`-s are dropped
        """
        try:
            cs = self.map['c'][rank]
            rs = self.map['r'][rank]
        except KeyError:
            warn('Call the `partition` method first!')
            return zip()
        cs = cs[cs >= -1]
        rs = rs[rs >= -1]
        if comm_flags:
            f = np.full_like(cs, False, dtype=bool)
            f[self.cid:] = True
            return zip(cs, rs, f)
        return zip(cs, rs)

    def countMSGTransfers(self, b=8, transferOverMSG=False):
        """
        if transferOverMSG is True the function returns
        number of times each communicator collects
        data; otherwise, number of messages received
        by each communicator
        """
        out = decompose(self.n_packs, self.n_comms, b=b)
        if transferOverMSG: return out
        return out * self.dimC
        
    def partition(self, n_batches):
        """
        Distributes `n_tasks` among `size` processes in the way
        that one can gather calculated results on the major cores
        later with a minimum number of communications*.

        Returns: number of required communicators and communications
                 or None if there are no communications

        (*under the condition of minimum execution time)
        ------------------------------------------------
        In the original problem setting, `n_tasks` is always
        a multiple of `n_colors`, though one can define 
        the methods' behavior for incomplete batches/tares
        (of size less than `n_colors`)
        """
        n_tasks = n_batches*self.dimC
        self.cid, progress = 0, 0
        self._initRankMap()
        while n_tasks > 0:
            k = 1
            while k < self.dimC:
                if k*self._size - n_tasks >= 0:
                    return self._helper1(k, n_tasks, progress)
                k += 1
            self._helper0(n_tasks, progress) 
            n_tasks -= self.size*self.dimC
            progress += self.size
            self.cid += self.dimC

    def _helper1(self, n_rows, n_tasks, progress):
        """
        Partitions `n_tasks` among `n_rows`.
        Calculates the number of required communicators
        `n_comms` and the number of communications `n_packs`
        """
        self.n_packs = 0
        for i in range(n_rows):
            fill_size = min(self._size, n_tasks)
            if i == 0: self.n_comms = fill_size // self.dimC

            C = np.full(self.size, -1)
            V = np.arange(fill_size) % self.dimC
            C[:len(V)] = V

            R = np.full(self.size, -1)
            V = np.arange(fill_size) // self.dimC
            V += progress + i*self._ncom
            R[:len(V)] = V

            self.map['c'] = np.c_[self.map['c'], C]
            self.map['r'] = np.c_[self.map['r'], R]

            n_tasks -= self._size
            self.n_packs += fill_size // self.dimC
        return self.n_comms, self.n_packs

    def _helper0(self, n_tasks, progress):
        """
        Deals with non-communicative distribution
        """
        fill_size = min(self.size, n_tasks//self.dimC)

        C = np.full((self.size, self.dimC), -1)
        V = np.tile(np.arange(self.dimC), (fill_size, 1))
        C[:fill_size] = V

        R = np.full((self.size, self.dimC), -1)
        V = np.tile(np.arange(fill_size)[:,None], (1, self.dimC))
        V += progress
        R[:fill_size] = V
        
        self.map['c'] = np.hstack((self.map['c'], C))
        self.map['r'] = np.hstack((self.map['r'], R))

    def _initRankMap(self):
        self.map['c'] = np.array([], 'i8').reshape(self.size, 0)
        self.map['r'] = np.array([], 'i8').reshape(self.size, 0)


class Redistribution:
    pass
