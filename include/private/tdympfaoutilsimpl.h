#if !defined(TDYMPFAOUTILSIMPL_H)
#define TDYMPFAOUTILSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode ExtractSubGmatrix(TDyMPFAO*,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode ExtractTempSubGmatrix(TDyMPFAO*,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode TDyUpdateBoundaryState(TDyMPFAO*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices(TDyMPFAO*,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices(TDyMPFAO*,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices(TDyMPFAO*,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDyMPFAO*,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDyMPFAO*,Vec);
PETSC_INTERN PetscErrorCode ComputeGtimesZ(PetscReal*,PetscReal*,PetscInt,PetscReal*);

#endif
