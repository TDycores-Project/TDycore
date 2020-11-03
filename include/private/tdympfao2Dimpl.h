#if !defined(TDYMPFAO2DIMPL_H)
#define TDYMPFAO2DIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyComputeGMatrixFor2DMesh(TDy);
PETSC_INTERN PetscErrorCode TDyComputeTransmissibilityMatrix2DMesh(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_2DMesh(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_2DMesh(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_2DMesh(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_2DMesh(TDy,Vec);
PETSC_INTERN PetscReal TDyMPFAOVelocityNorm_2DMesh(TDy);

#endif
