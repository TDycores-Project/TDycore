#if !defined(TDYMEMORY_H)
#define TDYMEMORY_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode Initialize_IntegerArray_1D(PetscInt*,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_1D(PetscReal*,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_2D(PetscReal**,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_3D(PetscReal***,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_1D(PetscReal**,PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_2D(PetscReal***,PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_3D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_4D(PetscReal*****,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Deallocate_RealArray_2D(PetscReal**,PetscInt);
PETSC_EXTERN PetscErrorCode Deallocate_RealArray_3D(PetscReal***,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Deallocate_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt);

#endif

