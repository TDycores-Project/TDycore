#if !defined(TIMESTEPPER_H)
#define TIMESTEPPER_H

#include <petsc.h>

typedef struct Timestepper *Timestepper;

struct Timestepper {
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

PETSC_EXTERN PetscErrorCode TDyTimestepperCreate(Timestepper*);
PETSC_EXTERN PetscErrorCode TDyTimestepperUpdateDT(Timestepper,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimestepperRunToTime(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimestepperDestroy(Timestepper*);

#endif
