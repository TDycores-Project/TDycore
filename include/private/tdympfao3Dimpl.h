#if !defined(TDYMPFAO3DIMPL_H)
#define TDYMPFAO3DIMPL_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyComputeGMatrixFor3DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyComputeTransmissibilityMatrix3DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_3DMesh(TDy,Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm_3DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy,Vec,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIFunction_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIJacobian_3DMesh(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOSetFromOptions(TDy);
#endif

