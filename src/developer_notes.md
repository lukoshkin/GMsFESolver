# Notes

The following improvements (bullet marks) are only valid for
`FunctionSpace`s with dofs exclusively on vertices:

```python
def zero_extrapolate(f, V):
    f_ext = Function(V)
    LagrangeInterpolator.interpolate(f_ext, f)
    return f_ext
```

* Substitution (\*-->\*) of `zero_extrapolate` with `fill_extrapolate`
gives 40-300x speed-up
* `project(K_big, V_small)` --> `extract_subdomain(K_big, V_small, i, j)`
gives 3-10x speed-up
* `.compute_vertex_values()` --> `.vector()[v2d]`
(with precomputed `vertex_to_dof_map`) **[not in use]**
gives 3-20x speed-up


* `project(a*b, a.function_space())` --> 
`a.vector().get_local()*b.vector().get_local()` - they give slightly different
results. The use of the latter may lead to undesired behaviour, thus it is
**[not in use]**


Failed to construct vector elements map from a subdomain to the whole domain
Tried it as follows:
    * define monotonous function on a subdomain
    * zero extrapolate to the whole domain with `LagrangeInterpolator`
    * extract non zero dofs and sort them, indices that sort the array
    are the desired map (in fact, it is not)
The extrapolation used on monotonous functions (I used `np.arange`) does not
change the order of the vector elements. So, the first thing comes to
mind is that, maybe, sorting is redundant. Which of coarse is not the case.
The second is that for all functions the map will be different.
For linearly increasing ones, it is the indentity map. For others it is not
known, and this it the problem.

To examine the last hypothesis, I sampled random floats, initialized a subdomain
vector entries with them, extrapolated to the whole domain. Then, I made two scatter
plots on the one figure: one is vector elements (and their positions) from the subdomain,
the other - non-zero entries of the domain vector (and their positions). The point
clouds of two different colors overlaped not completely

