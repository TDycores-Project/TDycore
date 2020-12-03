Introduction
========================================

TDycore is a set of dynamical cores for simulating three-dimensional land
processes in order to address questions about the hydrological cycle at a global
scale. TDycore applies higher-order, spatially-adaptive algorithms on geometries
that capture features of interest to study the subsurface transport of water and
energy.

TDycores provides two dynamical cores:

* a finite element (FE) version based on an *H-div* conforming space
* a finite volume (FV) version based on the multi-point flux approximation
  (MPFA)

Both dycores can use meshes with hexahedral or triangular prismatic cells with
planar faces. The prismatic cells are aligned logically along a z axis.

