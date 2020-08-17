#if !defined(TIMESTEPPER_H)
#define TIMESTEPPER_H

#include <petsc.h>

typedef struct Timestepper *Timestepper;

struct Timestepper {
  SNES snes;
  PetscBool io_process;
  PetscBool print_intermediate;
  PetscScalar dt_init;
  PetscScalar dt;
  PetscScalar time;
  PetscScalar final_time;
  PetscInt istep;
};

PETSC_EXTERN PetscErrorCode TimestepperCreate(Timestepper*);
PETSC_EXTERN PetscErrorCode TimestepperUpdateDT(Timestepper);
PETSC_EXTERN PetscErrorCode TimestepperRunToTime(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TimestepperDestroy(Timestepper*);

#endif
