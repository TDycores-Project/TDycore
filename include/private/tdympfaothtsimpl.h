#if !defined(TDYMPFAOTHTSIMPL_H)
#define TDYMPFAOTHTSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_InternalVertices_TH(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_TH(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_TH(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_InternalVertices_TH(Vec, Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_TH(Vec, Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_TH(Vec, Mat,void*);
#endif


