#include <finclude/tdycore.h>
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdm.h>

      type tTDy
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTDy

!
!  Types of TDy methods
!
      PetscEnum MPFA_O
      PetscEnum MPFA_O_DAE
      PetscEnum MPFA_O_TRANSIENT_VAR
      PetscEnum BDM
      PetscEnum WY

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (MPFA_O=0,MPFA_O_DAE=1,MPFA_O_TRANSIENT_VAR=2)
      parameter (BDM=3,WY=4)

!
!  Types of TDy modes
!
      PetscEnum RICHARDS
      PetscEnum TH

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (RICHARDS=0,TH=1)

!
!  Types of TDy flow boundary conditions
!
      PetscEnum UNDEFINED_FLOW_BC
      PetscEnum PRESSURE_BC
      PetscEnum VELOCITY_BC
      PetscEnum NOFLOW_BC
      PetscEnum SEEPAGE_BC

!
!  Types of TDy thermal boundary conditions
!
      PetscEnum UNDEFINED_THERMAL_BC
      PetscEnum TEMPERATURE_BC
      PetscEnum HEAT_FLUX_BC

!
!  Types of TDy salinity boundary conditions
!
      PetscEnum UNDEFINED_SALINITY_BC
      PetscEnum SALINE_CONC_BC

!
!  Types of TDy water densities
!
      PetscEnum WATER_DENSITY_CONSTANT
      PetscEnum WATER_DENSITY_EXPONENTIAL

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (WATER_DENSITY_CONSTANT=0,WATER_DENSITY_EXPONENTIAL=1)

!
!  Types of TDy MPFAO GMatrix
!
      PetscEnum MPFAO_GMATRIX_DEFAULT
      PetscEnum MPFAO_GMATRIX_TPF

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (MPFAO_GMATRIX_DEFAULT=0,MPFAO_GMATRIX_TPF=1)
