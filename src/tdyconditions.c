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
                iter != kh_end(conditions->bcs);
              ++iter) {
    if (!kh_exist(conditions->bcs, iter)) continue;
    BoundaryConditions bcs = kh_value(conditions->bcs, iter);
    if (bcs.mechanical_bc.context && bcs.mechanical_bc.dtor) {
      bcs.mechanical_bc.dtor(bcs.mechanical_bc.context);
    }
    if (bcs.thermal_bc.context && bcs.thermal_bc.dtor) {
      bcs.thermal_bc.dtor(bcs.thermal_bc.context);
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

PetscBool ConditionsHasForcing(Conditions *conditions) {
  return (conditions->compute_forcing != NULL);
}

PetscBool ConditionsHasEnergyForcing(Conditions *conditions) {
  return (conditions->compute_energy_forcing != NULL);
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

/// Sets mechanical and thermal boundary conditions on the face set with the
/// given index.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [in] bcs A BoundaryConditions struct holding the desired mechanical
///                 and thermal boundary conditions
PetscErrorCode ConditionsSetBCs(Conditions *conditions,
                                PetscInt face_set,
                                BoundaryConditions bcs) {
  PetscFunctionBegin;
  khiter_t iter = kh_get(TDY_BC, conditions->bcs, face_set);
  if (iter != kh_end(conditions->bcs)) { // we've already assigned a BC
    // Destroy the previous BC.
    BoundaryConditions prev_bcs = kh_value(conditions->bcs, iter);
    if (prev_bcs.mechanical_bc.context && prev_bcs.mechanical_bc.dtor) {
      prev_bcs.mechanical_bc.dtor(prev_bcs.mechanical_bc.context);
    }
    if (prev_bcs.thermal_bc.context && prev_bcs.thermal_bc.dtor) {
      prev_bcs.thermal_bc.dtor(prev_bcs.thermal_bc.context);
    }
  } else {
    int ret;
    iter = kh_put(TDY_BC, conditions->bcs, face_set, &ret);
  }
  kh_value(conditions->bcs, iter) = bcs;
  PetscFunctionReturn(0);
}

/// Retrieves mechanical and thermal boundary conditions on the face set with
/// the given index.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [in] bcs Storage for the retrieved boundary conditions. If no
///                 boundary conditions exist on the given face set, *bcs is
///                 zero-initialized (indicating no boundary conditions).
PetscErrorCode ConditionsGetBCs(Conditions *conditions,
                                PetscInt face_set,
                                BoundaryConditions *bcs) {
  PetscFunctionBegin;
  *bcs = (BoundaryConditions){0};
  khiter_t iter = kh_get(TDY_BC, conditions->bcs, face_set);
  if (iter != kh_end(conditions->bcs)) {
    *bcs = kh_val(conditions->bcs, iter);
  }
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

PetscErrorCode CreateConstantPressureBC(MechanicalBC *bc, PetscReal p0) {
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = p0;
  bc->type = TDY_PRESSURE_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = free;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantVelocityBC(MechanicalBC *bc, PetscReal v0) {
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = v0;
  bc->type = TDY_VELOCITY_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = free;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSeepageBC(MechanicalBC *bc) {
  PetscFunctionBegin;
  *bc = (MechanicalBC){0};
  bc->type = TDY_SEEPAGE_BC;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantTemperatureBC(ThermalBC *bc, PetscReal T0) {
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = T0;
  bc->type = TDY_TEMPERATURE_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = free;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantHeatFluxBC(ThermalBC *bc, PetscReal Q0) {
  PetscFunctionBegin;
  PetscReal *val = malloc(sizeof(PetscReal));
  *val = Q0;
  bc->type = TDY_HEAT_FLUX_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = free;
  PetscFunctionReturn(0);
}
