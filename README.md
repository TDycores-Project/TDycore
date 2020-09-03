[![Build Status](https://travis-ci.org/TDycores-Project/TDycore.svg?branch=master)](https://travis-ci.org/TDycores-Project/TDycore)
[![Code Coverage](https://codecov.io/gh/TDycores-Project/TDycore/branch/master/graph/badge.svg)](https://codecov.io/gh/TDycores-Project/TDycore)

# The Terrestrial Dynamical Core (TDycore)

This project consists of a set of dynamical cores for simulating
three-dimensional land processes in order to address questions about the
hydrological cycle at global scale. TDycore applies higher-order,
spatially-adaptive algorithms on geometries that capture features of interest,
in an attempt to study problems including

* lateral water and energy transport in high-latitude systems
* transport of water and energy in the soil-plant continuum
* urban flooding
* surface water dynamics

TDycores provides two dynamical cores: one based on a finite element (FE)
discretization, and one based on the finite volume multi-point flux
approximation.

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
3. Set `PETSC_DIR` and run PETSc's `configure` script.
   ```
   export PETSC_DIR=$PWD    # (e.g. if you're using bash)
   ./configure PETSC_ARCH=petsc-arch \
     --with-cc=gcc \
     --with-cxx=g++ \
     --with-fc=gfortran \
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
     --download-fblaslapack \
     --download-mpich=http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
   ```
4. Build PETSc.
   ```
   make all
   ```
5. Run PETSc's tests to make sure it's properly built.
   ```
   make test
   ```

Notice we're building PETSc in a debug configuration above, for development. To
build a "production" version of PETSc for performant simulations, run
`configure` with `--with-debug=0` instead.

### Building TDycore

When you've successully installed PETSc, you can build TDycore with `make`:

```
make V=1
```

Here, `V=1` isn't strictly needed--it just provides verbose reporting of
compiler and linker activity, for those of us who like to know that things are
happening.

