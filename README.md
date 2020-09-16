[![Build Status](https://travis-ci.org/TDycores-Project/TDycore.svg?branch=master)](https://travis-ci.org/TDycores-Project/TDycore)
[![Code Coverage](https://codecov.io/gh/TDycores-Project/TDycore/branch/master/graph/badge.svg)](https://codecov.io/gh/TDycores-Project/TDycore)

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

## Installation

To build TDycores, you need

* [CMake](https://cmake.org/) version 3.10+
* GNU Make
* Decent C, C++, and Fortran compilers
* [PETSc](https://www.mcs.anl.gov/petsc/), git revision `1a6d72e33c`
  installed as described below, with `PETSC_DIR` and `PETSC_ARCH` environment
  variables set accordingly.

For the purposes of illustration, let's assume you're using the `gcc`, `g++`,
and `gfortran` compilers.

### Installing PETSc and Dependencies

TDycore relies on PETSc to provide several required libraries. Here's how you
can properly configure and install PETSc on your system.

1. Clone the PETSc repo somewhere in your workspace.
   ```
   git clone https://gitlab.com/petsc/petsc.git
   ```
2. Set the repo to the correct revision.
   ```
   cd petsc
   git checkout 1a6d72e33c
   ```
3. Set `PETSC_DIR` and `PETSC_ARCH`, and then run PETSc's `configure` script.
   The value of `PETSC_ARCH` doesn't matter--it's just a name for your build
   configuration. It might be good to name it `debug` or `opt`, for example,
   depending on how you're configuring it.
   ```
   export PETSC_DIR=$PWD    # (e.g. if you're using bash)
   export PETSC_DIR=debug   # (for debug config)
   ./configure \
     --with-cc=mpicc \
     --with-cxx=mpicxx \
     --with-fc=mpif90 \
     --CFLAGS='-g -O0' --CXXFLAGS='-g -O0' --FFLAGS='-g -O0 -Wno-unused-function' \
     --with-clanguage=c \
     --with-debug=1  \
     --with-shared-libraries=0 \
     --download-hdf5 \
     --download-metis \
     --download-parmetis \
     --download-exodusii \
     --download-netcdf \
     --download-pnetcdf \
     --download-zlib \
     --download-fblaslapack
   ```
4. Build PETSc.
   ```
   make all
   ```
5. Run PETSc's tests to make sure it's properly built.
   ```
   make test
   ```

We're building PETSc in a debug configuration above, for use in development. To
build a "production" version of PETSc for performant simulations, run
`configure` with `--with-debug=0` instead.

#### Troubleshooting

If you get errors about undefined symbols during the configuration/build
process, you can try to fix it using the `--LIBS` flag to pass arguments to
the linker. For example, you might need `--LIBS='-ldl -lz'` to get some of the
I/O libraries to install properly.

On some systems, it may be easier to use shared libraries instead of static
libraries (`--with-shared-libraries=1`).

[Look here](https://www.mcs.anl.gov/petsc/documentation/installation.html)
for more information on getting PETSc to build successfully in various
configurations.

### Building TDycore

When you've successully installed PETSc, you can build TDycore with `make` from
the top-level source directory:

```
make -j V=1
```

Here, `V=1` isn't strictly needed--it just provides verbose reporting of
compiler and linker activity, for those of us who like to know that things are
happening.

### Running the Regression Tests

You can run TDycore's regression test suite with `make` from the top-level
source directory:

```
make test
```
