#if !defined(TDYMESHCUSTOMIMPL_H)
#define TDYMESHCUSTOMIMPL_H

#include <petsc.h>
#include <tdycore.h>

PETSC_INTERN PetscErrorCode TDyMeshCreateFromDiscretization(TDyDiscretizationType*,TDyMesh**);

#endif