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
      PetscEnum BDM
      PetscEnum WY

      ! The parameters values need to match those defined in
      ! the C code (i.e. include/tdycore.h)
      parameter (TPF=0,MPFA_O=1)
      parameter (BDM=2,WY=3)
