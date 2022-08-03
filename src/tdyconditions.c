#include <private/tdycoreimpl.h>
#include <private/tdymemoryimpl.h>
#include <petsc/private/khash/khash.h>

/// Initializes a new Conditions instance.
/// @param [out] conditions a new instance
PetscErrorCode ConditionsCreate(Conditions** conditions) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = TDyAlloc(sizeof(Conditions), conditions); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Frees the resources associated with the given Conditions instance.
PetscErrorCode ConditionsDestroy(Conditions* conditions) {
  PetscFunctionBegin;
  ConditionsSetForcing(conditions, NULL, NULL, NULL);
  ConditionsSetEnergyForcing(conditions, NULL, NULL, NULL);
  ConditionsSetBoundaryPressure(conditions, NULL, NULL, NULL);
  ConditionsSetBoundaryVelocity(conditions, NULL, NULL, NULL);
  ConditionsSetBoundaryTemperature(conditions, NULL, NULL, NULL);
  ConditionsSetBoundarySalinity(conditions, NULL, NULL, NULL);
  TDyFree(conditions);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute forcing.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetForcing(Conditions *conditions, void *context,
                                    PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                    PetscErrorCode (*dtor)(void*)) {
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
PetscErrorCode ConditionsSetEnergyForcing(Conditions *conditions, void *context,
                                          PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                          PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->energy_forcing_context && conditions->energy_forcing_dtor)
    conditions->energy_forcing_dtor(conditions->energy_forcing_context);
  conditions->energy_forcing_context = context;
  conditions->compute_energy_forcing = f;
  conditions->energy_forcing_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute salinity sources
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes salinity sources at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetSalinitySource(Conditions *conditions, void *context,
                                           PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                           PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->salinity_source_context && conditions->salinity_source_dtor)
    conditions->salinity_source_dtor(conditions->salinity_source_context);
  conditions->salinity_source_context = context;
  conditions->compute_salinity_source = f;
  conditions->salinity_source_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the boundary pressure.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the boundary pressure forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetBoundaryPressure(Conditions *conditions, void *context,
                                             PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                             PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->boundary_pressure_context && conditions->boundary_pressure_dtor)
    conditions->boundary_pressure_dtor(conditions->boundary_pressure_context);
  conditions->boundary_pressure_context = context;
  conditions->compute_boundary_pressure = f;
  conditions->boundary_pressure_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to set the boundary pressure type
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that sets the boundary pressure type
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetBoundaryPressureType(Conditions *conditions, void *context,
                                                 PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscInt*),
                                                 PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->boundary_pressure_type_context && conditions->boundary_pressure_type_dtor)
    conditions->boundary_pressure_type_dtor(conditions->boundary_pressure_type_context);
  conditions->boundary_pressure_type_context = context;
  conditions->assign_boundary_pressure_type = f;
  conditions->boundary_pressure_type_dtor = dtor;
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
                                             PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->boundary_velocity_context && conditions->boundary_velocity_dtor)
    conditions->boundary_velocity_dtor(conditions->boundary_velocity_context);
  conditions->boundary_velocity_context = context;
  conditions->compute_boundary_velocity = f;
  conditions->boundary_velocity_dtor = dtor;
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
                                                PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->boundary_temperature_context && conditions->boundary_temperature_dtor)
    conditions->boundary_temperature_dtor(conditions->boundary_temperature_context);
  conditions->boundary_temperature_context = context;
  conditions->compute_boundary_temperature = f;
  conditions->boundary_temperature_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the boundary saline concentration.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the boundary salinity forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetBoundarySalinity(Conditions *conditions,
                                             void *context,
                                             PetscErrorCode (*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                             PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->boundary_salinity_context && conditions->boundary_salinity_dtor)
    conditions->boundary_salinity_dtor(conditions->boundary_salinity_context);
  conditions->boundary_salinity_context = context;
  conditions->compute_boundary_salinity = f;
  conditions->boundary_salinity_dtor = dtor;
  PetscFunctionReturn(0);
}

PetscBool ConditionsHasForcing(Conditions *conditions) {
  return (conditions->compute_forcing != NULL);
}

PetscBool ConditionsHasEnergyForcing(Conditions *conditions) {
  return (conditions->compute_energy_forcing != NULL);
}

PetscBool ConditionsHasSalinitySource(Conditions *conditions) {
  return (conditions->compute_salinity_source != NULL);
}

PetscBool ConditionsHasBoundaryPressure(Conditions *conditions) {
  return (conditions->compute_boundary_pressure != NULL);
}

PetscBool ConditionsHasBoundaryPressureType(Conditions *conditions) {
  return (conditions->assign_boundary_pressure_type != NULL);
}

PetscBool ConditionsHasBoundaryVelocity(Conditions *conditions) {
  return (conditions->compute_boundary_velocity != NULL);
}

PetscBool ConditionsHasBoundaryTemperature(Conditions *conditions) {
  return (conditions->compute_boundary_temperature != NULL);
}

PetscBool ConditionsHasBoundarySalinity(Conditions *conditions) {
  return (conditions->compute_boundary_salinity != NULL);
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

PetscErrorCode ConditionsComputeSalinitySource(Conditions *conditions,
                                               PetscInt n, PetscReal *x,
                                               PetscReal *E) {
  return conditions->compute_salinity_source(conditions->salinity_source_context,
                                             n, x, E);
}

PetscErrorCode ConditionsComputeBoundaryPressure(Conditions *conditions,
                                                 PetscInt n, PetscReal *x,
                                                 PetscReal *p) {
  return conditions->compute_boundary_pressure(conditions->boundary_pressure_context,
                                               n, x, p);
}
PetscErrorCode ConditionsAssignBoundaryPressureType(Conditions *conditions,
                                                    PetscInt n, PetscReal *x,
                                                    PetscInt *btype) {
  return conditions->assign_boundary_pressure_type(conditions->boundary_pressure_type_context,
                                                   n, x, btype);
}

PetscErrorCode ConditionsComputeBoundaryVelocity(Conditions *conditions,
                                                 PetscInt n, PetscReal *x,
                                                 PetscReal *v) {
  return conditions->compute_boundary_velocity(conditions->boundary_velocity_context,
                                               n, x, v);
}

PetscErrorCode ConditionsComputeBoundaryTemperature(Conditions *conditions,
                                                    PetscInt n, PetscReal *x,
                                                    PetscReal *T) {
  return conditions->compute_boundary_temperature(conditions->boundary_temperature_context,
                                                  n, x, T);
}

PetscErrorCode ConditionsComputeBoundarySalinity(Conditions *conditions,
                                                 PetscInt n, PetscReal *x,
                                                 PetscReal *S) {
  return conditions->compute_boundary_salinity(conditions->boundary_salinity_context,
                                               n, x, S);
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
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = p0;
  ierr = ConditionsSetBoundaryPressure(conditions, val, ConstantBoundaryFn,
                                       TDyFree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryVelocity(Conditions *conditions,
                                                     PetscReal v0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = v0;
  ierr = ConditionsSetBoundaryVelocity(conditions, val, ConstantBoundaryFn,
                                       TDyFree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryTemperature(Conditions *conditions,
                                                        PetscReal T0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = T0;
  ierr = ConditionsSetBoundaryTemperature(conditions, val, ConstantBoundaryFn,
                                          TDyFree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundarySalinity(Conditions *conditions,
                                                     PetscReal S0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = S0;
  ierr = ConditionsSetBoundarySalinity(conditions, val, ConstantBoundaryFn,
                                       TDyFree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

