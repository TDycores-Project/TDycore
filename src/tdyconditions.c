#include <private/tdycoreimpl.h>

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


