#if !defined(TDYMPFAO3DTSIMPL_H)
#define TDYMPFAO3DTSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_Vertices_3DMesh(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_Vertices_3DMesh(Vec, Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Vec, Mat,void*);
#endif


