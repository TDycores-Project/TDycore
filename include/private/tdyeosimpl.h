#if !defined(TDYEOSIMPL_H)
#define TDYEOSIMPL_H

#include <petsc.h>

typedef enum {
  WATER_VISCOSITY_CONSTANT=0,
} TDyWaterViscosityType;

typedef enum {
  WATER_ENTHALPY_CONSTANT=0,
} TDyWaterEnthalpyType;

PETSC_EXTERN PetscErrorCode ComputeWaterDensity(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ComputeWaterViscosity(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ComputeWaterEnthalpy(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);

#endif

