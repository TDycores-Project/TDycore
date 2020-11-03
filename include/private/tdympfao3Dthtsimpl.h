#if !defined(TDYMPFAO3DTHTSIMPL_H)
#define TDYMPFAO3DTHTSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_InternalVertices_3DMesh_TH(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh_TH(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh_TH(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_InternalVertices_3DMesh_TH(Vec, Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh_TH(Vec, Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh_TH(Vec, Mat,void*);
#endif


