# run with
# `mpirun -n n_proc python3 multi_send-recv.py int_flag`
# int_flag > 0 - use non-blocking recv
# int_flag = 0 - use blocking recv

import sys
import numpy as np
from mpi4py import MPI


def overlap_map(N, n):
    """
    Returns the structured array of id pairs and labels.
    Labels characterize the overlap position of two regions
    specified by an id pair
    """

    # neighbors below
    I = np.arange(N-n)
    pairs = np.column_stack((I, I+n))
    labels = np.full(N-n, 'bottom')

    # neighbors on the right
    I = np.arange(N).reshape(n, n)[:, :-1].flatten()
    new_pairs = np.column_stack((I, I+1))
    new_labels = np.full(N-n, 'right')
    pairs = np.row_stack((pairs, new_pairs))
    labels = np.r_[labels, new_labels]

    # neighbors on the right, below
    I = np.arange(N-n).reshape(n-1, n)[:, :-1].flatten()
    new_pairs = np.column_stack((I, I+n+1))
    new_labels = np.full(N-2*n+1, 'bot-right')
    pairs = np.row_stack((pairs, new_pairs))
    labels = np.r_[labels, new_labels]

    # neighbors on the left, below
    I = np.arange(N-n).reshape(n-1, n)[:, 1:].flatten()
    new_pairs = np.column_stack((I, I+n-1))
    new_labels = np.full(N-2*n+1, 'bot-left')
    pairs = np.row_stack((pairs, new_pairs))
    labels = np.r_[labels, new_labels]

    # U12 if 'bottom-right'
    overlaps = np.rec.fromarrays([pairs, labels],
            [('ids', 'u4', (2,)), ('pos', 'U9')])
    return overlaps 
## <imports~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


non_blocking_recv = (int(sys.argv[1]) > 0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size > 9:
    print('set no more than 9 processes!')
    sys.exit(1)

overlaps = overlap_map(9, 3)
size_mask = (overlaps.ids >= size).sum(1).astype(bool)
overlaps = overlaps[~size_mask]

mask_in = (overlaps.ids[:, 0] == rank)
mask_out = (overlaps.ids[:, 1] == rank)
send_to = overlaps[mask_out].ids[:, 0]
recv_from = overlaps[mask_in].ids[:, 1]

pack_size = 1000
data = np.full(pack_size, rank, 'f8')

print(f'info++++>{rank}')
print('send to:', send_to)
print('sent from:', recv_from)
print(f'{rank}<++++info')

t =- MPI.Wtime()
# ----------- Measuring exec time from here --------------
for dest in send_to: 
    comm.Isend([data, MPI.DOUBLE], dest)

recvbuf = np.empty((len(recv_from), pack_size))
reqs = [None] * len(recv_from)

if non_blocking_recv:
    method = getattr(comm, 'Irecv')
    if rank == 0: print('\n\nNON-BLOCKING')
else:
    method = getattr(comm, 'Recv')
    if rank == 0: print('\n\nBLOCKING')

for i, src in enumerate(recv_from): 
    reqs[i] = method(recvbuf[i], src)
    
## >>> This segment is only necessary for Irecv procedure
if non_blocking_recv:
    for req in reqs:
        req.wait()
## <<<

# ----------- Measuring exec time to here --------------
t += MPI.Wtime()
res = comm.reduce(t, op=MPI.MAX, root=0)
if rank == 0: print(f'EXEC TIME: {res}\n\n')

print()
# The next line prints execution time of every process calls it
# print(f'finished in {t:.3} s')
print(f'res---->{rank}')
print(recvbuf)
print(f'{rank}<----res')

MPI.Finalize()
