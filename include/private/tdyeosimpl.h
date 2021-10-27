#if !defined(TDYEOSIMPL_H)
#define TDYEOSIMPL_H

#include <petsc.h>

typedef enum {
  WATER_VISCOSITY_CONSTANT=0,
} TDyWaterViscosityType;

typedef enum {
  WATER_ENTHALPY_CONSTANT=0,
} TDyWaterEnthalpyType;

typedef struct TDyEOS {
  TDyWaterDensityType density_type;
  PetscInt viscosity_type;
  PetscInt enthalpy_type;
} TDyEOS;

PETSC_INTERN PetscErrorCode TDyEOSComputeWaterDensity(TDyEOS*,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TDyEOSComputeWaterViscosity(TDyEOS*,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TDyEOSComputeWaterEnthalpy(TDyEOS*,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

#endif

