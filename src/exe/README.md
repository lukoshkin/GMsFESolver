# FEniCS-2019's Limitations

1. There is no map between a function space and and its subspace if the former
is not defined on a mesh vertices.

1. MeshView and Submesh instances do not inherit the communicator
from the original mesh

1. SubMesh can appear in a parallel program only in the latter's serial parts.
Even in this case, it can cause errors because of 2.

1. CompiledSubDomain, Expression, SubMesh, MeshView (and possibly some other classes)
lead to deadlocks in a parallel code if not treated properly (the first two have
kwarg `mpi_comm` that should be specified)

    e.g. this deadlocks:
    ```python
    from dolfin import *
    if MPI.rank(mpi_comm_world())==0:
        CompiledSubDomain('x[0]<.5')
    ```
    (the example is taken from [here](
    https://bitbucket.org/fenics-project/dolfin/issues/304/compiledsubdomain-and-expression))

---

# Some Notes

1. One can build a singularity image from a docker container:
    1. `singularity build imagename.simg docker-daemon://imagename:tag`
    1.
        ```bash
         # Definition file
         Bootstrap: docker-daemon
         From: imagename:tag
        ```
    1.
        ```bash
         docker save imagename -o imagename.tar
         singularity build imagename.sif docker-archive://imagename.tar
        ```
