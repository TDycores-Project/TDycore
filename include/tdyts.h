#if !defined(TIMESTEPPER_H)
#define TIMESTEPPER_H

#include <petsc.h>

typedef struct TDyTimestepper *TDyTimestepper;

struct TDyTimestepper {
  SNES snes;
  PetscScalar dt_init;
  PetscScalar dt_max;
  PetscScalar dt_reduction_factor;
  PetscScalar dt_growth_factor;
  PetscScalar dt;
  PetscScalar time;
  PetscScalar final_time;
  PetscInt istep;
};

PETSC_EXTERN PetscErrorCode TDyTimestepperCreate(TDyTimestepper*);
PETSC_EXTERN PetscErrorCode TDyTimestepperUpdateDT(TDyTimestepper,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimestepperRunToTime(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimestepperDestroy(TDyTimestepper*);

#endif
