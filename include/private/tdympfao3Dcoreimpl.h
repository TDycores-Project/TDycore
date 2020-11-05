#if !defined(TDYMPFAO3DCOREIMPL_H)
#define TDYMPFAO3DCOREIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyComputeGMatrixFor3DMesh(TDy);
PETSC_INTERN PetscErrorCode TDyUpdateTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeTransmissibilityMatrix3DMesh(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_3DMesh(TDy,Vec);
PETSC_INTERN PetscReal TDyMPFAOVelocityNorm_3DMesh(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_3DMesh_TH(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_3DMesh_TH(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_3DMesh(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_DAE_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOTransientVariable_3DMesh(TS,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_TransientVariable_3DMesh(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESFunction_3DMesh(SNES,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESJacobian_3DMesh(SNES,Vec,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSetFromOptions(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESPreSolve_3DMesh(TDy);

#endif
