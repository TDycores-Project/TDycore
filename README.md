[![Build Status](https://github.com/TDycores-Project/TDycore/workflows/auto_pr_test/badge.svg)](https://github.com/TDycores-Project/TDycore/actions)
[![Code Coverage](https://codecov.io/gh/TDycores-Project/TDycore/branch/master/graph/badge.svg)](https://codecov.io/gh/TDycores-Project/TDycore)
[![BSD-2-Clause](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

# The Terrestrial Dynamical Core (TDycore)

This project consists of a set of dynamical cores for simulating
three-dimensional land processes in order to address questions about the
hydrological cycle at global scale. TDycore applies higher-order,
spatially-adaptive algorithms on geometries that capture features of interest
to study the subsurface transport of water and energy.

TDycores provides two dynamical cores:

1. a finite element (FE) version based on an _H-div_ conforming space
2. a finite volume (FV) version based on the multi-point flux approximation
   (MPFA)

Both dycores can use meshes with hexahedral or triangular prismatic cells
with planar faces. The prismatic cells are aligned logically along a z axis.

## Dependencies

`TDycore` uses [PETSc](https://petsc.org/release/), tracking the
[main branch](https://gitlab.com/petsc/petsc) fairly closely. Currently,
`TDycore` is tested against revision `32a6fd01` (release v3.17.2).

## More information

Check out the [Wiki](https://github.com/TDycores-Project/TDycore/wiki) for
documentation, including instructions for
[building and installing TDycore](https://github.com/TDycores-Project/TDycore/wiki/Building-and-Installing-TDycore).
