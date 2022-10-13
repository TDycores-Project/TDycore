#include <private/tdycoreimpl.h>
#include <private/tdymemoryimpl.h>
#include <petsc/private/khash/khash.h>

/// Populates a BoundaryFaces instance with face indices corresponding to
/// the given face set in the given DM.
PetscErrorCode BoundaryFacesCreate(DM dm,
                                   PetscInt face_set,
                                   BoundaryFaces *bfaces) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  DMLabel face_label;
  ierr = DMGetLabelByNum(dm, face_set, &face_label); CHKERRQ(ierr);
  IS face_is;
  ierr = DMLabelGetValueIS(face_label, &face_is); CHKERRQ(ierr);
  ierr = ISGetSize(face_is, &bfaces->num_faces); CHKERRQ(ierr);
  const PetscInt *faces;
  ierr = ISGetIndices(face_is, &faces); CHKERRQ(ierr);
  ierr = TDyAlloc(bfaces->num_faces*sizeof(PetscInt), &(bfaces->faces)); CHKERRQ(ierr);
  memcpy(bfaces->faces, faces, bfaces->num_faces*sizeof(PetscInt));
  ierr = ISRestoreIndices(face_is, &faces); CHKERRQ(ierr);
  ierr = ISDestroy(&face_is); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Destroys the given BoundaryFaces instance.
PetscErrorCode BoundaryFacesDestroy(BoundaryFaces bfaces) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (bfaces.faces) {
    ierr = TDyFree(bfaces.faces); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Initializes a new Conditions instance.
/// @param [out] conditions a new instance
PetscErrorCode ConditionsCreate(Conditions** conditions) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = TDyAlloc(sizeof(Conditions), conditions); CHKERRQ(ierr);
  (*conditions)->bcs = kh_init(TDY_BC);
  (*conditions)->bfaces = kh_init(TDY_BFACE);
  PetscFunctionReturn(0);
}

/// Frees the resources associated with the given Conditions instance.
PetscErrorCode ConditionsDestroy(Conditions* conditions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ConditionsSetForcing(conditions, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = ConditionsSetEnergyForcing(conditions, NULL, NULL, NULL); CHKERRQ(ierr);

  // Clean up boundary conditions.
  for (khiter_t iter = kh_begin(conditions->bcs);
                iter != kh_end(conditions->bcs);
              ++iter) {
    if (!kh_exist(conditions->bcs, iter)) continue;
    BoundaryConditions bcs = kh_value(conditions->bcs, iter);
    if (bcs.flow_bc.context && bcs.flow_bc.dtor) {
      ierr = bcs.flow_bc.dtor(bcs.flow_bc.context); CHKERRQ(ierr);
    }
    if (bcs.thermal_bc.context && bcs.thermal_bc.dtor) {
      ierr = bcs.thermal_bc.dtor(bcs.thermal_bc.context); CHKERRQ(ierr);
    }
  }
  kh_destroy(TDY_BC, conditions->bcs);

  // Clean up boundary faces.
  for (khiter_t iter = kh_begin(conditions->bfaces);
                iter != kh_end(conditions->bfaces);
              ++iter) {
    if (!kh_exist(conditions->bfaces, iter)) continue;
    BoundaryFaces bfaces = kh_value(conditions->bfaces, iter);
    ierr = BoundaryFacesDestroy(bfaces); CHKERRQ(ierr);
  }
  kh_destroy(TDY_BFACE, conditions->bfaces);

  ierr = TDyFree(conditions); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute forcing.
/// @param [in] conditions A Conditions instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes forcing at a given number of points
/// @param [in] dtor A function that destroys the context when conditions is destroyed (can be NULL).
PetscErrorCode ConditionsSetForcing(Conditions *conditions, void *context,
                                    PetscErrorCode (*f)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*),
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
                                          PetscErrorCode (*f)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*),
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
                                           PetscErrorCode (*f)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*),
                                           PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (conditions->salinity_source_context && conditions->salinity_source_dtor)
    conditions->salinity_source_dtor(conditions->salinity_source_context);
  conditions->salinity_source_context = context;
  conditions->compute_salinity_source = f;
  conditions->salinity_source_dtor = dtor;
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

PetscErrorCode ConditionsComputeForcing(Conditions *conditions, PetscReal t, PetscInt n,
                                        PetscReal *x, PetscReal *F) {
  return conditions->compute_forcing(conditions->forcing_context, t, n, x, F);
}

PetscErrorCode ConditionsComputeEnergyForcing(Conditions *conditions, PetscReal t,
                                              PetscInt n, PetscReal *x,
                                              PetscReal *E) {
  return conditions->compute_energy_forcing(conditions->energy_forcing_context,
                                            t, n, x, E);
}

PetscErrorCode ConditionsComputeSalinitySource(Conditions *conditions, PetscReal t,
                                               PetscInt n, PetscReal *x,
                                               PetscReal *E) {
  return conditions->compute_salinity_source(conditions->salinity_source_context,
                                             t, n, x, E);
}

/// Returns the number of face sets associated with boundary conditions in this
/// Conditions instance.
/// @param [in] conditions A Conditions instance
PetscInt ConditionsNumFaceSets(Conditions* conditions) {
  PetscFunctionBegin;
  PetscFunctionReturn((PetscInt)kh_size(conditions->bcs));
}

/// Returns true if the given face set is associated with a set of boundary
/// conditions, false otherwise.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [out] have_bcs stores the result of the query
PetscBool ConditionsHaveBCs(Conditions *conditions, PetscInt face_set) {
  PetscFunctionBegin;
  khiter_t iter = kh_get(TDY_BC, conditions->bcs, face_set);
  PetscFunctionReturn(iter != kh_end(conditions->bcs));
}

/// Sets flow, thermal, and salinity boundary conditions on the face set with
/// the given index.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [in] bcs A BoundaryConditions struct holding the desired flow,
///                 thermal, and salinity boundary conditions
PetscErrorCode ConditionsSetBCs(Conditions *conditions,
                                PetscInt face_set,
                                BoundaryConditions bcs) {
  PetscFunctionBegin;
  khiter_t iter = kh_get(TDY_BC, conditions->bcs, face_set);
  if (iter != kh_end(conditions->bcs)) { // we've already assigned a BC
    // Destroy the previous BC.
    BoundaryConditions prev_bcs = kh_value(conditions->bcs, iter);
    if (prev_bcs.flow_bc.context && prev_bcs.flow_bc.dtor) {
      prev_bcs.flow_bc.dtor(prev_bcs.flow_bc.context);
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

/// Retrieves flow, thermal, and salinity boundary conditions on the face set
/// with the given index.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [out] bcs Storage for the retrieved boundary conditions. If no
///                  boundary conditions exist on the given face set, *bcs is
///                  zero-initialized (indicating no boundary conditions).
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

/// Retrieves all flow, thermal, and salinity boundary conditions on all face
/// set associated with this Conditions instance
/// @param [in] conditions A Conditions instance
/// @param [in] face_sets An array sufficiently large to store the indices of
///                       all face sets associated with this Conditions. If
///                       NULL, face sets are not retrieved.
/// @param [out] bcs An array suffiently large to store all boundary conditions.
///                  If NULL, boundary conditions are not retrieved.
PetscErrorCode ConditionsGetAllBCs(Conditions *conditions,
                                   PetscInt *face_sets,
                                   BoundaryConditions *bcs) {
  PetscFunctionBegin;
  PetscInt i = 0;
	for (khiter_t k = kh_begin(conditions->bcs);
       k != kh_end(conditions->bcs); ++k) {
    if (kh_exist(conditions->bcs, k)) {
      if (face_sets) {
        face_sets[i] = kh_key(conditions->bcs, k);
      }
      if (bcs) {
        bcs[i] = kh_val(conditions->bcs, k);
      }
      ++i;
    }
  }
  PetscFunctionReturn(0);
}

/// Sets conditions on the face set with
/// the given index.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [in] bfaces A BoundaryFaces struct holding the indices of faces
///                    that belong to the given face set. The Conditions
///                    instance assumes responsibility for bfaces.
PetscErrorCode ConditionsSetBoundaryFaces(Conditions *conditions,
                                          PetscInt face_set,
                                          BoundaryFaces bfaces) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  khiter_t iter = kh_get(TDY_BFACE, conditions->bfaces, face_set);
  if (iter != kh_end(conditions->bfaces)) { // we've already assigned a BC
    // Destroy the previous boundary faces.
    BoundaryFaces prev_bfaces = kh_value(conditions->bfaces, iter);
    ierr = BoundaryFacesDestroy(prev_bfaces); CHKERRQ(ierr);
  } else {
    int ret;
    iter = kh_put(TDY_BFACE, conditions->bfaces, face_set, &ret);
  }
  kh_value(conditions->bfaces, iter) = bfaces;
  PetscFunctionReturn(0);
}

/// Retrieves boundary faces for the given face set.
/// @param [in] conditions A Conditions instance
/// @param [in] face_set The index of a face set identifying the surface on
///                      which to specify the boundary pressure
/// @param [out] bfaces Storage for the retrieved boundary faces. If no
///                     boundary faces exist on the given face set, *bfaces is
///                     zero-initialized (indicating no boundary faces).
PetscErrorCode ConditionsGetBoundaryFaces(Conditions *conditions,
                                          PetscInt face_set,
                                          BoundaryFaces *bfaces) {
  PetscFunctionBegin;
  *bfaces = (BoundaryFaces){0};
  khiter_t iter = kh_get(TDY_BFACE, conditions->bfaces, face_set);
  if (iter != kh_end(conditions->bfaces)) {
    *bfaces = kh_val(conditions->bfaces, iter);
  }
  PetscFunctionReturn(0);
}

/// Retrieves all flow, thermal, and salinity boundary conditions on all face
/// set associated with this Conditions instance
/// @param [in] conditions A Conditions instance
/// @param [in] face_sets An array sufficiently large to store the indices of
///                       all face sets associated with this Conditions. If
///                       NULL, face sets are not retrieved.
/// @param [in] bfaces An array suffiently large to store all boundary faces.
///                    If NULL, boundary conditions are not retrieved.
PetscErrorCode ConditionsGetAllBoundaryFaces(Conditions *conditions,
                                             PetscInt *face_sets,
                                             BoundaryFaces *bfaces) {
  PetscFunctionBegin;
  PetscInt i = 0;
	for (khiter_t k = kh_begin(conditions->bfaces);
       k != kh_end(conditions->bfaces); ++k) {
    if (kh_exist(conditions->bfaces, k)) {
      if (face_sets) {
        face_sets[i] = kh_key(conditions->bfaces, k);
      }
      if (bfaces) {
        bfaces[i] = kh_val(conditions->bfaces, k);
      }
      ++i;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ConstantBoundaryFn(void *context, PetscReal t,
                                         PetscInt n, PetscReal *x,
                                         PetscReal *v) {
  PetscFunctionBegin;
  PetscReal v0 = *((PetscReal*)context);
  for (PetscInt i = 0; i < n; ++i) {
    v[i] = v0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantPressureBC(FlowBC *bc, PetscReal p0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = p0;
  bc->type = TDY_PRESSURE_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = TDyFree;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantVelocityBC(FlowBC *bc, PetscReal v0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = v0;
  bc->type = TDY_VELOCITY_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = TDyFree;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSeepageBC(FlowBC *bc) {
  PetscFunctionBegin;
  *bc = (FlowBC){0};
  bc->type = TDY_SEEPAGE_BC;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantTemperatureBC(ThermalBC *bc, PetscReal T0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = T0;
  bc->type = TDY_TEMPERATURE_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = TDyFree;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantHeatFluxBC(ThermalBC *bc, PetscReal T0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = T0;
  bc->type = TDY_HEAT_FLUX_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = TDyFree;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateConstantSalinityBC(SalinityBC *bc, PetscReal S0) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *val;
  ierr = TDyAlloc(sizeof(PetscReal), &val); CHKERRQ(ierr);
  *val = S0;
  bc->type = TDY_SALINE_CONC_BC;
  bc->context = val;
  bc->compute = ConstantBoundaryFn;
  bc->dtor = TDyFree;
  PetscFunctionReturn(0);
}

PetscErrorCode EnforceFlowBC(FlowBC* bc, PetscReal t,
                             PetscInt n, PetscReal *points,
                             PetscReal *values) {
  PetscFunctionBegin;
  bc->compute(bc->context, t, n, points, values);
  PetscFunctionReturn(0);
}

PetscErrorCode EnforceThermalBC(ThermalBC* bc, PetscReal t,
                                PetscInt n, PetscReal* points,
                                PetscReal *values) {
  PetscFunctionBegin;
  bc->compute(bc->context, t, n, points, values);
  PetscFunctionReturn(0);
}

PetscErrorCode EnforceSalinityBC(SalinityBC* bc, PetscReal t,
                                 PetscInt n, PetscReal* points,
                                 PetscReal *values) {
  PetscFunctionBegin;
  bc->compute(bc->context, t, n, points, values);
  PetscFunctionReturn(0);
}
