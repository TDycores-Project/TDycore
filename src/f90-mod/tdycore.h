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
      PetscEnum TPF
      PetscEnum MPFA_O
      PetscEnum MPFA_O_DAE
      PetscEnum MPFA_O_TRANSIENT_VAR
      PetscEnum BDM
      PetscEnum Q2
      PetscEnum WY

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (TPF=0,MPFA_O=1,MPFA_O_DAE=2,MPFA_O_TRANSIENT_VAR=3)
      parameter (BDM=4,Q2=5,WY=6)

!
!  Types of TDy modes
!
      PetscEnum RICHARDS
      PetscEnum TH

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (RICHARDS=0,TH=1)

!
!  Types of TDy water densities
!
      PetscEnum WATER_DENSITY_CONSTANT
      PetscEnum WATER_DENSITY_EXPONENTIAL

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (WATER_DENSITY_CONSTANT=0,WATER_DENSITY_EXPONENTIAL=1)
