#include <private/tdycoreimpl.h>
#include <petsc/private/khash/khash.h>

/// Initializes a new Conditions instance.
/// @param [out] conditions a new instance
PetscErrorCode ConditionsCreate(Conditions** conditions) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscCalloc(sizeof(Conditions), conditions); CHKERRQ(ierr);
  (*conditions)->bcs = kh_init(TDY_BC);
  PetscFunctionReturn(0);
}

/// Frees the resources associated with the given Conditions instance.
PetscErrorCode ConditionsDestroy(Conditions* conditions) {
  PetscFunctionBegin;
  ConditionsSetForcing(conditions, NULL, NULL, NULL);
  ConditionsSetEnergyForcing(conditions, NULL, NULL, NULL);

  // Clean up boundary conditions.
  for (khiter_t iter = kh_begin(conditions->bcs);
                iter != kh_end(conditions->bc);
              ++iter) {
    if (!kh_exist(conditions->bcs, iter)) continue;
    BoundaryCondition bc = kh_value(conditions->bcs, iter);
    if (bc.context && bc.dtor) {
      bc.dtor(bc.context);
    }
  }
  kh_destroy(TDY_BC, conditions->bcs);

  PetscFree(conditions);
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
                                          void (*dtor)(void*)) {
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
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [in] bc A BoundaryCondition struct that can compute the boundary
///                pressure
PetscErrorCode ConditionsSetBoundaryPressure(Conditions *conditions,
                                             PetscInt face_set,
                                             BoundaryCondition bc) {
  PetscFunctionBegin;
  khiter_t iter = kh_get(conditions->bcs, face_set);
  if (iter != kh_end(conditions->bcs)) { // we've already assigned a BC
    // Destroy the previous BC.
    BoundaryCondition prev_bc = kh_value(conditions->bcs, iter);
    if (prev_bc.context && prev_bc.dtor) {
      prev_bc.dtor(prev_bc.context);
    }
  }
  kh_value(conditions->bcs, iter) = bc;
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
  ierr = ConditionsSetBoundaryPressure(conditions, val, ConstantBoundaryFn,
                                       free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryTemperature(Conditions *conditions,
                                                        PetscReal T0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = T0;
  ierr = ConditionsSetBoundaryTemperature(conditions, val, ConstantBoundaryFn,
                                          free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConditionsSetConstantBoundaryVelocity(Conditions *conditions,
                                                     PetscReal v0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = v0;
  ierr = ConditionsSetBoundaryVelocity(conditions, val, ConstantBoundaryFn,
                                       free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
