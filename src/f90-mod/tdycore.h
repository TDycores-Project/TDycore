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
      PetscEnum BDM
      PetscEnum WY

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (TPF=0,MPFA_O=1,MPFA_O_DAE=2)
      parameter (BDM=3,WY=4)

!
!  Types of TDy water densities
!
      PetscEnum WATER_DENSITY_CONSTANT
      PetscEnum WATER_DENSITY_EXPONENTIAL

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (WATER_DENSITY_CONSTANT=0,WATER_DENSITY_EXPONENTIAL=1)
