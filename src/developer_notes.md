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
