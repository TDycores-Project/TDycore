#if !defined(TDYROCKPROPERTIES_H)
#define TDYROCKPROPERTIES_H

#include <petsc.h>

PETSC_EXTERN void TDyRockDensityFunctionDefault(PetscReal*,PetscReal*);
PETSC_EXTERN void TDySpecificHeatCapacityFunctionDefault(PetscReal*,PetscReal*);

#endif
