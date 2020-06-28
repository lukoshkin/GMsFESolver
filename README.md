# GMsFEM Implementation

# Pros

- easy-to-catch concept of GMsFEM shown in the NumPy implementation
- modular structure that allows targeted assembling
- convinient API for FEniCS-2019 docker containers
- parallel pipeline (not fully completed)

# Cons

- it includes all drawbacks connected with FEniCS-2019.1, like bugs
or unimplemented features. Major releases (e.g. dolfinx) coming after
make this implementation a bit outdated

- the code is not fully-optimized. For example, it can be rewritten in C++
which will half at least the the runtime of some of its sections. A few 
algorithmic features may be improved as well

# ToDo

* Write a C++ implementation
* Use DOLFINX (when it is released)
* Substitute triangular coarse elements for square ones
---
* Finish README
* Add MIT license
