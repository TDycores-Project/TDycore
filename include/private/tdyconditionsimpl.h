#if !defined(TDYCONDITIONS_H)
#define TDYCONDITIONS_H

#include <private/tdyoptions.h>

/// This type gathers settings related to boundary and source/sink conditions.
typedef struct Conditions {

  /// Contexts provided for condition-related functions.
  void* boundary_pressure_context;
  void* forcing_context;
  void* energy_forcing_context;
  void* salinity_source_context;
  void* boundary_pressure_type_context;
  void* boundary_velocity_context;
  void* boundary_temperature_context;
  void* boundary_saline_conc_context;

  /// Compute momentum source/sink contributions at a set of given points.
  PetscErrorCode (*compute_forcing)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Compute energy source/sink contributions at a set of given points.
  PetscErrorCode (*compute_energy_forcing)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Compute salinity source/sink contributions at a set of given points.
  PetscErrorCode (*compute_salinity_source)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Compute the pressure at a set of given points on the boundary.
  PetscErrorCode (*compute_boundary_pressure)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Set the pressure at a set of given points on the boundary.
  PetscErrorCode (*assign_boundary_pressure_type)(void*,PetscInt,PetscReal*,PetscInt*);

  /// Compute the components of the component of the velocity normal to a given
  /// set of points on the boundary.
  PetscErrorCode (*compute_boundary_velocity)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Compute the temperature at a set of given points on the boundary.
  PetscErrorCode (*compute_boundary_temperature)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Compute the saline concentration at a set of given points on the boundary.
  PetscErrorCode (*compute_boundary_saline_conc)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Context destructors.
  PetscErrorCode (*forcing_dtor)(void*);
  PetscErrorCode (*energy_forcing_dtor)(void*);
  PetscErrorCode (*salinity_source_dtor)(void*);
  PetscErrorCode (*boundary_velocity_dtor)(void*);
  PetscErrorCode (*boundary_pressure_dtor)(void*);
  PetscErrorCode (*boundary_pressure_type_dtor)(void*);
  PetscErrorCode (*boundary_temperature_dtor)(void*);
  PetscErrorCode (*boundary_saline_conc_dtor)(void*);
} Conditions;

// conditions creation/destruction
PETSC_INTERN PetscErrorCode ConditionsCreate(Conditions**);
PETSC_INTERN PetscErrorCode ConditionsDestroy(Conditions*);

// conditions setup functions
PETSC_INTERN PetscErrorCode ConditionsSetForcing(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetEnergyForcing(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetSalinitySource(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryPressure(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryPressureType(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscInt*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryVelocity(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryTemperature(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*) , PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundarySalineConcentration(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));

// conditions query functions
PETSC_INTERN PetscBool ConditionsHasForcing(Conditions*);
PETSC_INTERN PetscBool ConditionsHasEnergyForcing(Conditions*);
PETSC_INTERN PetscBool ConditionsHasSalinitySource(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundaryPressure(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundaryPressureType(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundaryVelocity(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundaryTemperature(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundarySalineConcentration(Conditions*);

// conditions computation
// TODO: Change to PETSC_INTERN when we fix demo/steady/steady.c
PETSC_EXTERN PetscErrorCode ConditionsComputeForcing(Conditions*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeEnergyForcing(Conditions*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeSalinitySource(Conditions*,PetscInt,PetscReal*,PetscReal*);
// TODO: Change to PETSC_INTERN when we fix demo/steady/steady.c
PETSC_EXTERN PetscErrorCode ConditionsComputeBoundaryPressure(Conditions*,PetscInt,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode ConditionsAssignBoundaryPressureType(Conditions*,PetscInt,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode ConditionsComputeBoundaryVelocity(Conditions*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeBoundaryTemperature(Conditions*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeBoundarySalineConcentration(Conditions*,PetscInt,PetscReal*,PetscReal*);

// convenience functions
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundaryPressure(Conditions*,PetscReal);
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundaryVelocity(Conditions*,PetscReal);
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundaryTemperature(Conditions*,PetscReal);
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundarySalineConcentration(Conditions*,PetscReal);

#endif

