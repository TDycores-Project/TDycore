#if !defined(TDYCONDITIONS_H)
#define TDYCONDITIONS_H

#include <petsc.h>

PETSC_INTERN PetscErrorCode TDyConstantBoundaryPressureFn(TDy,PetscReal*,PetscReal*,void*);
PETSC_INTERN PetscErrorCode TDyConstantBoundaryVelocityFn(TDy,PetscReal*,PetscReal*,void*);

#endif

