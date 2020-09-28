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

PETSC_EXTERN PetscErrorCode TDyTimeIntegratorCreate(TDyTimeIntegrator*);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorUpdateDT(TDyTimeIntegrator,
                                                      PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorRunToTime(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorDestroy(TDyTimeIntegrator*);

#endif
