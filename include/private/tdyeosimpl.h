#if !defined(TDYEOSIMPL_H)
#define TDYEOSIMPL_H

#include <petsc.h>

typedef enum {
  WATER_DENSITY_CONSTANT=0,
  Water_DENSITY_EXPONENTIAL=1
} TDyWaterDensityType;

typedef enum {
  WATER_VISCOSITY_CONSTANT=0,
} TDyWaterViscosityType;

PETSC_EXTERN PetscErrorCode ComputeWaterDensity(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ComputeWaterViscosity(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscReal*);

#endif

