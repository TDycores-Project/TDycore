module tdycoredefdummy
#include <../src/f90-mod/tdycore.h>
#include <petsc/finclude/petscts.h>
end module tdycoredefdummy

module tdycoredef
  use tdycoredefdummy
end module tdycoredef

module tdycore
  use tdycoredef

  interface
     subroutine TDyFinalize(ierr)
       integer ierr
     end subroutine TDyFinalize
  end interface
  interface
     subroutine TDyCreate(a,b,z)
       use petscdm
       use tdycoredef
       DM a
       TDy b
       integer z
     end subroutine TDyCreate
  end interface
  interface
     subroutine TDySetDiscretizationMethod(a,b,z)
       use tdycoredef
       TDy a
       PetscInt b
       integer z
     end subroutine TDySetDiscretizationMethod
  end interface
  interface
     subroutine TDySetupDiscretizationMethod(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDySetupDiscretizationMethod
  end interface
  interface
     subroutine TDySetup(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDySetup
  end interface
  interface
     subroutine TDyComputeSystem(a,b,c,z)
       use tdycoredef
       use petscvec
       use petscmat
       TDy a
       Mat b
       Vec c
       integer z
     end subroutine TDyComputeSystem
  end interface

  Interface TDySetBlockPermeabilityValuesLocal
     subroutine TDySetBlockPermeabilityValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetBlockPermeabilityValuesLocal0
     subroutine TDySetBlockPermeabilityValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetBlockPermeabilityValuesLocal11
  end interface TDySetBlockPermeabilityValuesLocal

  Interface TDySetPorosityValuesLocal
     subroutine TDySetPorosityValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetPorosityValuesLocal0
     subroutine TDySetPorosityValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetPorosityValuesLocal11
  end interface TDySetPorosityValuesLocal

  Interface TDySetResidualSaturationValuesLocal
     subroutine TDySetResidualSaturationValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetResidualSaturationValuesLocal0
     subroutine TDySetResidualSaturationValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetResidualSaturationValuesLocal11
  end interface TDySetResidualSaturationValuesLocal

  Interface TDySetMaterialPropertyMValuesLocal
     subroutine TDySetMaterialPropertyMValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetMaterialPropertyMValuesLocal0
     subroutine TDySetMaterialPropertyMValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetMaterialPropertyMValuesLocal11
  end interface TDySetMaterialPropertyMValuesLocal

  Interface TDySetMaterialPropertyNValuesLocal
     subroutine TDySetMaterialPropertyNValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetMaterialPropertyNValuesLocal0
     subroutine TDySetMaterialPropertyNValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetMaterialPropertyNValuesLocal11
  end interface TDySetMaterialPropertyNValuesLocal

  Interface TDySetMaterialPropertyAlphaValuesLocal
     subroutine TDySetMaterialPropertyAlphaValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetMaterialPropertyAlphaValuesLocal0
     subroutine TDySetMaterialPropertyAlphaValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetMaterialPropertyAlphaValuesLocal11
  end interface TDySetMaterialPropertyAlphaValuesLocal

  Interface TDySetSourceSinkValuesLocal
     subroutine TDySetSourceSinkValuesLocal0(a,b,c,d,z)
       use tdycoredef
       TDy a ! Vec
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       PetscScalar d (*) ! PetscScalar
       integer z
     end subroutine TDySetSourceSinkValuesLocal0
     subroutine TDySetSourceSinkValuesLocal11(a,b,c,d,e,z)
       use tdycoredef
       TDy a ! TDy
       PetscInt b ! PetscInt
       PetscInt c ! PetscInt
       PetscScalar d ! PetscScalar
       integer z
     end subroutine TDySetSourceSinkValuesLocal11
  end interface TDySetSourceSinkValuesLocal

  Interface TDyGetSaturationValuesLocal
     subroutine TDyGetSaturationValuesLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscScalar c (*) ! PetscScalar
       integer z
     end subroutine TDyGetSaturationValuesLocal
  end interface TDyGetSaturationValuesLocal

  Interface TDyGetLiquidMassValuesLocal
     subroutine TDyGetLiquidMassValuesLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscScalar c (*) ! PetscScalar
       integer z
     end subroutine TDyGetLiquidMassValuesLocal
  end interface TDyGetLiquidMassValuesLocal

  Interface TDyGetMaterialPropertyMValuesLocal
     subroutine TDyGetMaterialPropertyMValuesLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscScalar c (*) ! PetscScalar
       integer z
     end subroutine TDyGetMaterialPropertyMValuesLocal
  end interface TDyGetMaterialPropertyMValuesLocal

  Interface TDyGetMaterialPropertyAlphaValuesLocal
     subroutine TDyGetMaterialPropertyAlphaValuesLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscScalar c (*) ! PetscScalar
       integer z
     end subroutine TDyGetMaterialPropertyAlphaValuesLocal
  end interface TDyGetMaterialPropertyAlphaValuesLocal

  Interface TDyGetPorosityValuesLocal
     subroutine TDyGetPorosityValuesLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscScalar c (*) ! PetscScalar
       integer z
     end subroutine TDyGetPorosityValuesLocal
  end interface TDyGetPorosityValuesLocal

  Interface TDyGetBlockPermeabilityValuesLocal
     subroutine TDyGetBlockPermeabilityValuesLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscScalar c (*) ! PetscScalar
       integer z
     end subroutine TDyGetBlockPermeabilityValuesLocal
  end interface TDyGetBlockPermeabilityValuesLocal

  Interface TDyGetNumCellsLocal
     subroutine TDyGetNumCellsLocal(a,b,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       integer z
     end subroutine TDyGetNumCellsLocal
  end interface TDyGetNumCellsLocal

  Interface TDyGetCellNaturalIDsLocal
     subroutine TDyGetCellNaturalIDsLocal(a,b,c,z)
       use tdycoredef
       TDy a ! tdy
       PetscInt b ! PetscInt
       PetscInt c (*) ! PetscInt
       integer z
     end subroutine TDyGetCellNaturalIDsLocal
  end interface TDyGetCellNaturalIDsLocal

  Interface TDyGetCellIsLocal
     subroutine TDyGetCellIsLocal(a,b,c,z)
       use tdycoredef
       TDy a
       PetscInt b
       PetscInt c (*)
       integer z
     end subroutine TDyGetCellIsLocal
  end interface TDyGetCellIsLocal

  Interface TDyUpdateState
     subroutine TDyUpdateState(a,b,z)
       use tdycoredef
       TDy a ! tdy
       PetscScalar b (*) ! PetscScalar
       integer z
     end subroutine TDyUpdateState
  end interface TDyUpdateState

  interface
     subroutine TDySetIFunction(a,b,z)
       use tdycoredef
       use petscts
       TS a
       TDy b
       integer z
     end subroutine TDySetIFunction
  end interface

  interface
     subroutine TDySetIJacobian(a,b,z)
       use tdycoredef
       use petscts
       TS a
       TDy b
       integer z
     end subroutine TDySetIJacobian
  end interface

  interface
     subroutine TDySetSNESFunction(a,b,z)
       use tdycoredef
       use petscsnes
       SNES a
       TDy b
       integer z
     end subroutine TDySetSNESFunction
  end interface

  interface
     subroutine TDySetSNESJacobian(a,b,z)
       use tdycoredef
       use petscsnes
       SNES a
       TDy b
       integer z
     end subroutine TDySetSNESJacobian
  end interface

  interface
     subroutine TDyCreateVectors(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyCreateVectors
  end interface

  interface
     subroutine TDyCreateJacobian(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyCreateJacobian
  end interface

  interface
     subroutine TDySetDtimeForSNESSolver(a,b,z)
       use tdycoredef
       TDy a
       PetscReal b
       integer z
     end subroutine TDySetDtimeForSNESSolver
  end interface

  interface
     subroutine TDySetPreviousSolutionForSNESSolver(a,b,z)
       use tdycoredef
       use petscvec
       TDy a
       Vec b
       integer z
     end subroutine TDySetPreviousSolutionForSNESSolver
  end interface

  interface
     subroutine TDyPreSolveSNESSolver(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyPreSolveSNESSolver
  end interface

  interface
     subroutine TDyPostSolveSNESSolver(a,b,z)
       use tdycoredef
       use petscvec
       TDy a
       Vec b
       integer z
     end subroutine TDyPostSolveSNESSolver
  end interface

  contains

  subroutine TDyInit(ierr)
#include <petsc/finclude/petscvec.h>
     use petscvec
     implicit none
     integer ierr

     call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
     CHKERRQ(ierr)
     call TDyInitNoArguments(ierr)
  end subroutine TDyInit

end module tdycore
