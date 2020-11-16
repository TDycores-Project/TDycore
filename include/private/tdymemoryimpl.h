#if !defined(TDYMEMORYIMPL_H)
#define TDYMEMORYIMPL_H

#include <petsc.h>
#include <private/tdymeshimpl.h>

PETSC_EXTERN PetscErrorCode TDyInitialize_IntegerArray_1D(PetscInt*,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyInitialize_IntegerArray_2D(PetscInt**,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyInitialize_RealArray_1D(PetscReal*,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode TDyInitialize_RealArray_2D(PetscReal**,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode TDyInitialize_RealArray_3D(PetscReal***,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode TDyInitialize_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode TDyAllocate_IntegerArray_1D(PetscInt**,PetscInt);
PETSC_EXTERN PetscErrorCode TDyAllocate_IntegerArray_2D(PetscInt***,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyAllocate_RealArray_1D(PetscReal**,PetscInt);
PETSC_EXTERN PetscErrorCode TDyAllocate_RealArray_2D(PetscReal***,PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TDyAllocate_RealArray_3D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyAllocate_RealArray_4D(PetscReal*****,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyDeallocate_IntegerArray_2D(PetscInt**,PetscInt);
PETSC_EXTERN PetscErrorCode TDyDeallocate_RealArray_2D(PetscReal**,PetscInt);
PETSC_EXTERN PetscErrorCode TDyDeallocate_RealArray_3D(PetscReal***,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyDeallocate_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDyCell_1D(PetscInt,TDyCell**);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDyVertex_1D(PetscInt,TDyVertex**);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDyEdge_1D(PetscInt,TDyEdge**);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDyFace_1D(PetscInt,TDyFace**);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDySubcell_1D(PetscInt,TDySubcell**);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDyVector_1D(PetscInt,TDyVector**);
PETSC_EXTERN PetscErrorCode TDyAllocate_TDyCoordinate_1D(PetscInt,TDy_coordinate**);

#endif

