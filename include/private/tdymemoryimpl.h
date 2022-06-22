#if !defined(TDYMEMORYIMPL_H)
#define TDYMEMORYIMPL_H

#include <petsc.h>
#include <private/tdymeshimpl.h>

PETSC_INTERN PetscErrorCode TDyInitialize_IntegerArray_1D(PetscInt*,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyInitialize_IntegerArray_2D(PetscInt**,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_1D(PetscReal*,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_2D(PetscReal**,PetscInt,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_3D(PetscReal***,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyAllocate_IntegerArray_1D(PetscInt**,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_IntegerArray_2D(PetscInt***,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_1D(PetscReal**,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_2D(PetscReal***,PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_3D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_4D(PetscReal*****,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_1D(PetscReal*);
PETSC_INTERN PetscErrorCode TDyDeallocate_IntegerArray_1D(PetscInt*);
PETSC_INTERN PetscErrorCode TDyDeallocate_IntegerArray_2D(PetscInt**,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_2D(PetscReal**,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_3D(PetscReal***,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyCell_1D(PetscInt,TDyCell**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyVertex_1D(PetscInt,TDyVertex**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyEdge_1D(PetscInt,TDyEdge**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyFace_1D(PetscInt,TDyFace**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDySubcell_1D(PetscInt,TDySubcell**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyVector_1D(PetscInt,TDyVector**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyCoordinate_1D(PetscInt,TDyCoordinate**);

#endif

