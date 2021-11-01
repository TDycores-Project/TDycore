#include <private/tdycoreimpl.h>
#include <petsc/private/khash/khash.h>

/// Initializes a new Conditions instance.
/// @param [out] conditions a new instance
PetscErrorCode ConditionsCreate(Conditions** conditions) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscCalloc(sizeof(TDyConditions), conditions); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Frees the resources associated with the given Conditions instance.
PetscErrorCode ConditionsDestroy(Conditions* conditions) {
  PetscFunctionBegin;
  ConditionsSetForcing(conditions, NULL, NULL);
  ConditionsSetEnergyForcing(conditions, NULL, NULL);
  ConditionsSetBoundaryPressure(conditions, NULL, NULL);
  ConditionsSetBoundaryTemperature(conditions, NULL, NULL);
  ConditionsSetBoundaryVelocity(conditions, NULL, NULL);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute forcing.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetForcing(Conditions *conditions, void *context,
                                    PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                    void (*dtor)(void*)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (conditions->forcing_context && conditions->forcing_dtor)
    conditions->forcing_dtor(conditions->forcing_context);
  conditions->forcing_context = context;
  conditions->compute_forcing = f;
  conditions->forcing_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute energy forcing.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes energy forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetEnergyForcing(Conditions *condition, void *context,
                                          PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                          void (*dtor)(void*)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (conditions->energy_forcing_context && conditions->energy_forcing_dtor)
    conditions->energy_forcing_dtor(conditions->energy_forcing_context);
  conditions->energy_forcing_context = context;
  conditions->compute_energy_forcing = f;
  conditions->energy_forcing_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the boundary pressure.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the boundary pressure forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetBoundaryPressure(Conditions *conditions, void *context,
                                             PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                             void (*dtor)(void*)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (conditions->boundary_pressure_context && conditions->boundary_pressure_dtor)
    conditions->boundary_pressure_dtor(conditions->boundary_pressure_context);
  conditions->boundary_pressure_context = context;
  conditions->compute_boundary_pressure = f;
  conditions->boundary_pressure_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the boundary temperature.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the boundary temperature forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetBoundaryTemperature(Conditions *conditions,
                                                void *context,
                                                PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                                void (*dtor)(void*)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (conditions->boundary_temperature_context && conditions->boundary_temperature_dtor)
    conditions->boundary_temperature_dtor(conditions->boundary_temperature_context);
  conditions->boundary_temperature_context = context;
  conditions->compute_boundary_temperature = f;
  conditions->boundary_temperature_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the (normal) boundary velocity.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the boundary temperature forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetBoundaryVelocity(Conditions *conditions,
                                             void *context,
                                             PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                             void (*dtor)(void*)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (conditions->boundary_velocity_context && conditions->boundary_velocity_dtor)
    conditions->boundary_velocity_dtor(conditions->boundary_velocity_context);
  conditions->boundary_velocity_context = context;
  conditions->compute_boundary_velocity = f;
  conditions->boundary_velocity_dtor = dtor;
  PetscFunctionReturn(0);
}

// Here's a registry of functions that can be used for boundary conditions and
// forcing terms.
typedef PetscErrorCode(*Function)(PetscInt, PetscReal*, PetscReal*);
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
      SETERRQ(PETSC_COMM_WORLD, ierr, "Function not found!");
      return ierr;
    }
  } else {
    ierr = -1;
    SETERRQ(PETSC_COMM_WORLD, ierr, "No functions have been registered!");
    return ierr;
  }
  PetscFunctionReturn(0);
}

// This struct is stored in a context and used to call a Function with a NULL
// context.
typedef struct WrapperStruct {
  Function func;
} WrapperStruct;

// This function calls an underlying Function with a NULL context.
PetscErrorCode WrapperFunction(void *context, PetscInt n, PetscReal *x, PetscReal *v) {
  WrapperStruct *wrapper = context;
  return wrapper->func(n, x, v);
}

PetscErrorCode ConditionsSelectBoundaryPressure(Conditions *conditions,
                                                const char* name) {
  PetscFunctionBegin;
  int ierr;
  Function f;
  ierr = TDyGetFunction(name, &f); CHKERRQ(ierr);
  FunctionWrapper *wrapper = malloc(sizeof(FunctionWrapper));
  wrapper->func = f;
  ierr = ConditionsSetBoundaryPressure(conditions, wrapper, f, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSelectBoundaryTemperature(Conditions *conditions,
                                                   const char* name) {
  PetscFunctionBegin;
  int ierr;
  Function f;
  ierr = TDyGetFunction(name, &f); CHKERRQ(ierr);
  FunctionWrapper *wrapper = malloc(sizeof(FunctionWrapper));
  wrapper->func = f;
  ierr = ConditionsSetBoundaryTemperature(conditions, wrapper, f, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSelectBoundaryVelocity(Conditions *conditions,
                                                void *context,
                                                const char* name) {
  PetscFunctionBegin;
  int ierr;
  Function f;
  ierr = TDyGetFunction(name, &f); CHKERRQ(ierr);
  FunctionWrapper *wrapper = malloc(sizeof(FunctionWrapper));
  wrapper->func = f;
  ierr = ConditionsSetBoundaryVelocity(conditions, wrapper, f, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ConstantBoundaryFn(void *context,
                                         PetscInt n, PetscReal *x,
                                         PetscReal *v) {
  PetscFunctionBegin;
  PetscReal v0 = *((PetscReal*)context);
  for (PetscInt i = 0; i < n; ++i) {
    v[i] = v0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryPressure(Conditions *conditions,
                                                     PetscReal p0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = p0;
  ierr = ConditionsSetBoundaryPressure(conditions, val, ConѕtantBoundaryFn,
                                       free);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryTemperature(Conditions *conditions,
                                                        PetscReal T0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = T0;
  ierr = ConditionsSetBoundaryTemperature(conditions, val, ConѕtantBoundaryFn,
                                          free);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryVelocity(Conditions *conditions,
                                                     PetscReal v0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = v0;
  ierr = ConditionsSetBoundaryVelocity(conditions, val, ConѕtantBoundaryFn,
                                       free);
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

