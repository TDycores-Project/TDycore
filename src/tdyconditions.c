#include <private/tdycoreimpl.h>

/*
  Boundary and source-sink conditions are set by PETSc operations
*/

PetscErrorCode TDySetForcingFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeforcing = f;
  if (ctx) tdy->forcingctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetEnergyForcingFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeenergyforcing = f;
  if (ctx) tdy->energyforcingctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBoundaryTemperatureFn(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->compute_boundary_temperature = f;
  if (ctx) tdy->boundary_temperature_ctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBoundaryPressureFn(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->compute_boundary_pressure = f;
  if (ctx) tdy->boundary_pressure_ctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBoundaryVelocityFn(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->compute_boundary_velocity = f;
  if (ctx) tdy->boundary_velocity_ctx = ctx;
  PetscFunctionReturn(0);
}

/*
  Boundary and source-sink conditions are cell-by-cell
*/

PetscErrorCode TDySetSourceSinkValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  for(i=0; i<ni; i++) {
    tdy->source_sink[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetEnergySourceSinkValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  for(i=0; i<ni; i++) {
    tdy->energy_source_sink[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

