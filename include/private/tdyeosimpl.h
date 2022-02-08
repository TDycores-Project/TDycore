#if !defined(TDYEOSIMPL_H)
#define TDYEOSIMPL_H

#include <petsc.h>



typedef enum {
  WATER_ENTHALPY_CONSTANT=0,
} TDyWaterEnthalpyType;

PETSC_EXTERN PetscErrorCode ComputeWaterDensity(PetscReal,PetscReal,PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ComputeWaterViscosity(PetscReal,PetscReal,PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ComputeWaterEnthalpy(PetscReal,PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ComputeSalinityFraction(PetscReal,PetscReal,PetscReal,PetscReal*);

#endif

