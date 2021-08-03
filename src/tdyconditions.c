#include <private/tdycoreimpl.h>
#include <petsc/private/khash/khash.h>

/*
  Boundary and source-sink conditions are set by PETSc operations
*/

// Here's a registry of pressure, temperature, and velocity functions.
typedef PetscErrorCode(*BCFunction)(TDy, PetscReal*, PetscReal*, void*);
KHASH_MAP_INIT_STR(TDY_BCFUNC_MAP, BCFunction)
static khash_t(TDY_BCFUNC_MAP)* pressure_funcs_ = NULL;
static khash_t(TDY_BCFUNC_MAP)* temperature_funcs_ = NULL;
static khash_t(TDY_BCFUNC_MAP)* velocity_funcs_ = NULL;

PetscErrorCode TDyRegisterPressureFn(const char* name, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*)) {
  PetscFunctionBegin;
  if (pressure_funcs_ == NULL) {
    pressure_funcs_ = kh_init(TDY_BCFUNC_MAP);
  }

  int retval;
  khiter_t iter = kh_put(TDY_BCFUNC_MAP, pressure_funcs_, name, &retval);
  kh_val(pressure_funcs_, iter) = f;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegisterTemperatureFn(const char* name, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*)) {
  PetscFunctionBegin;
  if (temperature_funcs_ == NULL) {
    temperature_funcs_ = kh_init(TDY_BCFUNC_MAP);
  }

  int retval;
  khiter_t iter = kh_put(TDY_BCFUNC_MAP, temperature_funcs_, name, &retval);
  kh_val(temperature_funcs_, iter) = f;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegisterVelocityFn(const char* name, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*)) {
  PetscFunctionBegin;
  if (velocity_funcs_ == NULL) {
    velocity_funcs_ = kh_init(TDY_BCFUNC_MAP);
  }

  int retval;
  khiter_t iter = kh_put(TDY_BCFUNC_MAP, velocity_funcs_, name, &retval);
  kh_val(velocity_funcs_, iter) = f;
  PetscFunctionReturn(0);
}

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

PetscErrorCode TDySelectBoundaryPressureFn(TDy tdy, const char* name, void* ctx) {
  PetscFunctionBegin;
  int ierr;

  khiter_t iter = kh_get(TDY_BCFUNC_MAP, pressure_funcs_, name);
  if (iter != kh_end(pressure_funcs_)) { // found it!
    BCFunction f = kh_val(pressure_funcs_, iter);
    ierr = TDySetBoundaryPressureFn(tdy, f, ctx); CHKERRQ(ierr);
  } else {
    printf("Uh oh!\n"); // TODO: handle error condition
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryTemperatureFn(TDy tdy, const char* name, void* ctx) {
  PetscFunctionBegin;
  int ierr;

  khiter_t iter = kh_get(TDY_BCFUNC_MAP, temperature_funcs_, name);
  if (iter != kh_end(temperature_funcs_)) { // found it!
    BCFunction f = kh_val(temperature_funcs_, iter);
    ierr = TDySetBoundaryTemperatureFn(tdy, f, ctx); CHKERRQ(ierr);
  } else {
    printf("Uh oh!\n"); // TODO: handle error condition
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryVelocityFn(TDy tdy, const char* name, void* ctx) {
  PetscFunctionBegin;
  int ierr;

  khiter_t iter = kh_get(TDY_BCFUNC_MAP, velocity_funcs_, name);
  if (iter != kh_end(velocity_funcs_)) { // found it!
    BCFunction f = kh_val(velocity_funcs_, iter);
    ierr = TDySetBoundaryVelocityFn(tdy, f, ctx); CHKERRQ(ierr);
  } else {
    printf("Uh oh!\n"); // TODO: handle error condition
  }
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

