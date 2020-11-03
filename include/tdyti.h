#if !defined(TIMEINTEGRATOR_H)
#define TIMEINTEGRATOR_H

#include <petsc.h>

typedef struct TDyTimeIntegrator *TDyTimeIntegrator;

struct TDyTimeIntegrator {
  SNES snes;
  TS ts;
  TDyTimeIntegrationMethod time_integration_method;
  PetscScalar dt_init;
  PetscScalar dt_max;
  PetscScalar dt_reduction_factor;
  PetscScalar dt_growth_factor;
  PetscScalar dt;
  PetscScalar dt_save;
  PetscScalar time;
  PetscScalar final_time;
  PetscInt istep;
};

PETSC_INTERN PetscErrorCode TDyTimeIntegratorCreate(TDyTimeIntegrator*);
PETSC_INTERN PetscErrorCode TDyTimeIntegratorUpdateDT(TDyTimeIntegrator,
                                                      PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorRunToTime(TDy,PetscReal);
PETSC_INTERN PetscErrorCode TDyTimeIntegratorDestroy(TDyTimeIntegrator*);

#endif
