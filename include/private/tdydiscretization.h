#if !defined(TDYDISCRETIZATION_H)
#define TDYDISCRETIZATION_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyCreateGlobalVector(TDy,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateLocalVector(TDy,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateNaturalVector(TDy,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateJacobianMatrix(TDy,Mat*);
PETSC_INTERN PetscErrorCode TDyGlobalToNatural(TDy,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyGlobalToLocal(TDy,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyNaturalToGlobal(TDy,Vec,Vec);

#endif

