#!/bin/bash
python3 pre-build.py # TODO: pass arguments
mpirun -n 9 python3 cluster.py # TODO pass arguments
