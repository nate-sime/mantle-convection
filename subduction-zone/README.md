# Subduction zone modelling with DOLFINx

Provided is a DOLFINx implementation of the subduction zone model presented
in van Keken et al. (2010) *A community benchmark for subduction zone modeling*
(https://doi.org/10.1016/j.pepi.2008.04.015). This implementation may be
used to reproduce the cases exhibited, or enable new subduction models with
prescribed geometries. 

These examples are extensible for custom models with more sophisticated
material models, boundary conditions and geometries.

### Note

These examples are not intended to be instructive for geophysics modelling or
DOLFINx use. Refer to, for example, van Keken et al (2010) and
[the DOLFINx tutorial](https://jsdokken.com/dolfinx-tutorial/), respectively.


### Dependencies

* The components of the FEniCSx project: https://github.com/FEniCS
* The components of PETSc: https://gitlab.com/petsc/petsc
* NURBS-Python (`geomdl`): https://github.com/orbingol/NURBS-Python
* gmsh (including OpenCASCADE): https://gmsh.info/
* NumPy: https://numpy.org/
* SciPy: https://scipy.org/


# Community subduction benchmark reproduction

The standard workflow is as follows:

1. Generate a 2D mesh. The default parameters are those required by the benchmark.

```bash
python3 mesh_generator.py
```

2. Run the subduction zone model

```bash
python3 subduction_zone.py
```

In order to reproduce cases 1c, 2a and 2b, isoviscous, diffusion creep and
dislocation creep viscosity models are provided  in `model.py`.


# Custom subduction zone geometry

`mesh_generator.py` is easily extensible for custom slab geometries defined
in the (x, y) plane. The
slab surface is approximated by a B-spline interpolation of a
sequence of cartesian points monotonically increasing the `x` direction.

A 3D mesh generator example is given in `mesh_generator3d.py` where the given
example is an approximation of the Mariana Trench.

### Notes on 3D modelling:

Mesh generation is currently limited to serial processing only.

Parallel computation of the subduction zone model is easily facilitated by MPI,
e.g.,

```bash
mpirun -np 2 python3 subduction_zone.py
```

Generating 3D geometries is *difficult*. One must ensure:

* an appropriate approximations of the slab surface,
* the B-spline interpolating those points is smooth,
* the curation of the geometry definitions, volume lables and face labels for 
  the solvers' interpretation,
* a high quality mesh with well conditioned cells,
* appropriate resolution of the mesh resolving material coefficients.

It is recommended to use an iterative solver to solve the 3D linear Stokes
system. A simple implementation is provided; however, it is encouraged that
the user implement their problem specific solver depending on the complexity
of their model.