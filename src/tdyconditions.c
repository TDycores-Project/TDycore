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

PetscBool ConditionsHasForcing(Conditions *conditions) {
  return (conditions->compute_forcing != NULL);
}

PetscBool ConditionsHasEnergyForcing(Conditions *conditions) {
  return (conditions->compute_energy_forcing != NULL);
}

PetscBool ConditionsHasBoundaryPressure(Conditions *conditions) {
  return (conditions->compute_boundary_pressure != NULL);
}

PetscBool ConditionsHasBoundaryTemperature(Conditions *conditions) {
  return (conditions->compute_boundary_temperature != NULL);
}

PetscBool ConditionsHasBoundaryVelocity(Conditions *conditions) {
  return (conditions->compute_boundary_velocity != NULL);
}

PetscErrorCode ConditionsComputeForcing(Conditions *conditions, PetscInt n,
                                        PetscReal *x, PetscReal *F) {
  return conditions->compute_forcing(conditions->forcing_context, n, x, F);
}

PetscErrorCode ConditionsComputeEnergyForcing(Conditions *conditions,
                                              PetscInt n, PetscReal *x,
                                              PetscReal *E) {
  return conditions->compute_energy_forcing(conditions->energy_forcing_context,
                                            n, x, E);
}

PetscErrorCode ConditionsComputeBoundaryPressure(Conditions *conditions,
                                                 PetscInt n, PetscReal *x,
                                                 PetscReal *p) {
  return conditions->compute_boundary_pressure(conditions->boundary_pressure_context,
                                               n, x, p);
}

PetscErrorCode ConditionsComputeBoundaryTemperature(Conditions *conditions,
                                                    PetscInt n, PetscReal *x,
                                                    PetscReal *T) {
  return conditions->compute_boundary_temperature(conditions->boundary_temperature_context,
                                                  n, x, T);
}

PetscErrorCode ConditionsComputeBoundaryVelocity(Conditions *conditions,
                                                 PetscInt n, PetscReal *x,
                                                 PetscReal *v) {
  return conditions->compute_boundary_velocity(conditions->boundary_velocity_context,
                                                  n, x, v);
}

// This struct is stored in a context and used to call a Function with a NULL
// context.
typedef struct WrapperStruct {
  TDySpatialFunction func;
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

