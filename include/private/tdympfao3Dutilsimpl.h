#if !defined(TDYMPFAO3DUTILSIMPL_H)
#define TDYMPFAO3DUTILSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyUpdateBoundaryState(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDy,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDy,Vec);
PETSC_INTERN PetscErrorCode ComputeGtimesZ(PetscReal*,PetscReal*,PetscInt,PetscReal*);

#endif
