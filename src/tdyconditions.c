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

PetscErrorCode TDySetTemperatureDirichletValueFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computetemperaturedirichletvalue = f;
  if (ctx) tdy->temperaturedirichletvaluectx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDirichletValueFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computedirichletvalue = f;
  if (ctx) tdy->dirichletvaluectx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDirichletFluxFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computedirichletflux = f;
  if (ctx) tdy->dirichletfluxctx = ctx;
  PetscFunctionReturn(0);
}

/*
  Boundary and source-sink conditions are cell-by-cell
*/

PetscErrorCode TDySetSourceSinkValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  for(i=0; i<ni; i++) {
    tdy->source_sink[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetEnergySourceSinkValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  for(i=0; i<ni; i++) {
    tdy->energy_source_sink[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

