#include <private/tdycoreimpl.h>
#include <petsc/private/khash/khash.h>

/*
  Boundary and source-sink conditions are set by PETSc operations
*/

// Here's a registry of functions that can be used for boundary conditions and
// forcing terms.
typedef PetscErrorCode(*Function)(TDy, PetscReal*, PetscReal*, void*);
KHASH_MAP_INIT_STR(TDY_FUNC_MAP, Function)
static khash_t(TDY_FUNC_MAP)* funcs_ = NULL;

// This function is called on finalization to destroy the function registry.
static void DestroyFunctionRegistry() {
  kh_destroy(TDY_FUNC_MAP, funcs_);
}

PetscErrorCode TDyRegisterFunction(const char* name, Function f) {
  PetscFunctionBegin;
  if (funcs_ == NULL) {
    funcs_ = kh_init(TDY_FUNC_MAP);
    TDyOnFinalize(DestroyFunctionRegistry);
  }

  int retval;
  khiter_t iter = kh_put(TDY_FUNC_MAP, funcs_, name, &retval);
  kh_val(funcs_, iter) = f;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetFunction(const char* name, Function* f) {
  PetscFunctionBegin;
  int ierr;

  if (funcs_ != NULL) {
    khiter_t iter = kh_get(TDY_FUNC_MAP, funcs_, name);
    if (iter != kh_end(funcs_)) { // found it!
      *f = kh_val(funcs_, iter);
    } else {
      ierr = -1;
      SETERRQ(MPI_COMM_WORLD, ierr, "Function not found!");
      return ierr;
    }
  } else {
    ierr = -1;
    SETERRQ(MPI_COMM_WORLD, ierr, "No functions have been registered!");
    return ierr;
  }
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
  Function f;
  ierr = TDyGetFunction(name, &f); CHKERRQ(ierr);
  ierr = TDySetBoundaryPressureFn(tdy, f, ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryTemperatureFn(TDy tdy, const char* name, void* ctx) {
  PetscFunctionBegin;
  int ierr;
  Function f;
  ierr = TDyGetFunction(name, &f); CHKERRQ(ierr);
  ierr = TDySetBoundaryTemperatureFn(tdy, f, ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryVelocityFn(TDy tdy, const char* name, void* ctx) {
  PetscFunctionBegin;
  int ierr;
  Function f;
  ierr = TDyGetFunction(name, &f); CHKERRQ(ierr);
  ierr = TDySetBoundaryVelocityFn(tdy, f, ctx); CHKERRQ(ierr);
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

