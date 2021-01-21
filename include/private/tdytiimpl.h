#if !defined(TIMEINTEGRATORIMPL_H)
#define TIMEINTEGRATORIMPL_H

#include <petsc.h>

typedef struct _p_TDyTimeIntegrator *TDyTimeIntegrator;

struct _p_TDyTimeIntegrator {
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
PETSC_INTERN PetscErrorCode TDyTimeIntegratorUpdateDT(TDyTimeIntegrator,PetscReal);
PETSC_INTERN PetscErrorCode TDyTimeIntegratorDestroy(TDyTimeIntegrator*);

#endif
