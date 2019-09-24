#if !defined(TDYMPFAO2DIMPL_H)
#define TDYMPFAO2DIMPL_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyComputeGMatrixFor2DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyComputeTransmissibilityMatrix2DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_2DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_2DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_2DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_2DMesh(TDy,Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm_2DMesh(TDy);

#endif
