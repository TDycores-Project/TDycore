#if !defined(TDYMEMORYIMPL_H)
#define TDYMEMORYIMPL_H

#include <petsc.h>
#include <private/tdymeshimpl.h>

PETSC_EXTERN PetscErrorCode Initialize_IntegerArray_1D(PetscInt*,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_1D(PetscReal*,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_2D(PetscReal**,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_3D(PetscReal***,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Initialize_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode Allocate_IntegerArray_1D(PetscInt**,PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_1D(PetscReal**,PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_2D(PetscReal***,PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_3D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_RealArray_4D(PetscReal*****,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Deallocate_RealArray_2D(PetscReal**,PetscInt);
PETSC_EXTERN PetscErrorCode Deallocate_RealArray_3D(PetscReal***,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Deallocate_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode Allocate_TDyCell_1D(PetscInt,TDy_cell**);
PETSC_EXTERN PetscErrorCode Allocate_TDyVertex_1D(PetscInt,TDy_vertex**);
PETSC_EXTERN PetscErrorCode Allocate_TDyEdge_1D(PetscInt,TDy_edge**);
PETSC_EXTERN PetscErrorCode Allocate_TDyFace_1D(PetscInt,TDy_face**);
PETSC_EXTERN PetscErrorCode Allocate_TDySubcell_1D(PetscInt,TDy_subcell**);
PETSC_EXTERN PetscErrorCode Allocate_TDyVector_1D(PetscInt,TDy_vector**);
PETSC_EXTERN PetscErrorCode Allocate_TDyCoordinate_1D(PetscInt,TDy_coordinate**);

#endif

