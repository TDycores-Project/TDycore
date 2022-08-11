#if !defined(TDYMPFAO_SALINITY_TS_IMPL_H)
#define TDYMPFAO_SALINITY_TS_IMPL_H

#include <petsc.h>

 PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_InternalVertices_Salinity(Vec,Vec,void*);
 PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_Salinity(Vec,Vec,void*);
 PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_Salinity(Vec,Vec,void*);
 PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_InternalVertices_Salinity(Vec, Mat,void*);
 PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_Salinity(Vec, Mat,void*);
 PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_Salinity(Vec, Mat,void*);

#endif

