#if !defined(TDYMPFAO3DUTILSIMPL_H)
#define TDYMPFAO3DUTILSIMPL_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyUpdateBoundaryState(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDy,Vec);
PETSC_EXTERN PetscErrorCode ComputeGtimesZ(PetscReal*,PetscReal*,PetscInt,PetscReal*);

#endif
