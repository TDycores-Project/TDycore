#if !defined(TDYMPFAOUTILSIMPL_H)
#define TDYMPFAOUTILSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode ExtractSubGmatrix(TDy,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode ExtractTempSubGmatrix(TDy,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode TDyUpdateBoundaryState(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDy,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDy,Vec);
PETSC_INTERN PetscErrorCode ComputeGtimesZ(PetscReal*,PetscReal*,PetscInt,PetscReal*);

#endif
