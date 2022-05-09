#if !defined(TDYDISCRETIZATION_H)
#define TDYDISCRETIZATION_H

#include <petsc.h>
#include <private/tdydmimpl.h>

PETSC_INTERN PetscErrorCode TDyCreateGlobalVector(TDyDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateLocalVector(TDyDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateNaturalVector(TDyDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateJacobianMatrix(TDyDM*,Mat*);
PETSC_INTERN PetscErrorCode TDyGlobalToNatural(TDyDM*,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyGlobalToLocal(TDyDM*,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyNaturalToGlobal(TDyDM*,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyNaturaltoLocal(TDyDM*,Vec,Vec*);

#endif

