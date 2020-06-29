#if !defined(TDYMPFAO3DTSIMPL_H)
#define TDYMPFAO3DTSIMPL_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyMPFAOIFunction_InternalVertices_3DMesh(Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_3DMesh(Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIJacobian_InternalVertices_3DMesh(Vec, Mat,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh(Vec, Mat,void*);
PETSC_EXTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Vec, Mat,void*);
#endif


