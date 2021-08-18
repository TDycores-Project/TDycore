#if !defined(TDYMPFAOIMPL_H)
#define TDYMPFAOIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyMPFAOInitialize(TDy);
PETSC_INTERN PetscErrorCode TDyComputeGMatrixMPFAO(TDy);
PETSC_INTERN PetscErrorCode TDyComputeGMatrixTPF(TDy);
PETSC_INTERN PetscErrorCode TDyUpdateTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeGravityDiscretization(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_TH(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_TH(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_DAE(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOTransientVariable(TS,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_TransientVariable(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESFunction(SNES,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESJacobian(SNES,Vec,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSetFromOptions(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESPreSolve(TDy);

#endif
