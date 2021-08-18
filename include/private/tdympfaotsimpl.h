#if !defined(TDYMPFAOTSIMPL_H)
#define TDYMPFAOTSIMPL_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_Vertices(Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_Vertices(Vec, Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices(Vec, Mat,void*);
#endif


