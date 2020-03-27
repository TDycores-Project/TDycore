#if !defined(TDYMPFAO3DCOREIMPL_H)
#define TDYMPFAO3DCOREIMPL_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyComputeGMatrixFor3DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyComputeTransmissibilityMatrix3DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity_3DMesh(TDy,Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm_3DMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOIFunction_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIJacobian_3DMesh(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIFunction_DAE_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOTransientVariable_3DMesh(TS,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIFunction_TransientVariable_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOSNESFunction_3DMesh(SNES,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOSNESJacobian_3DMesh(SNES,Vec,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOSetFromOptions(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOSNESPreSolve_3DMesh(TDy);

#endif
