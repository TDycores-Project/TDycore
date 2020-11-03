#if !defined(TDYROCKPROPERTIES_H)
#define TDYROCKPROPERTIES_H

#include <petsc.h>

PETSC_INTERN void TDyRockDensityFunctionDefault(PetscReal*,PetscReal*);
PETSC_INTERN void TDySpecificHeatCapacityFunctionDefault(PetscReal*,PetscReal*);

#endif
