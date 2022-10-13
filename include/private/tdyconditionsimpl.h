#if !defined(TDYCONDITIONS_H)
#define TDYCONDITIONS_H

#include <private/tdyoptions.h>

#include <petsc/private/khash/khash.h>

/// This type represents a flow (pressure/velocity/head) boundary condition
/// defined by a function, a context, and a destructor.
typedef struct FlowBC {
  /// type of boundary condition
  TDyFlowBCType type;
  /// context pointer
  void *context;
  /// vectorized function for computing boundary values
  PetscErrorCode (*compute)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);
  /// destructor
  PetscErrorCode (*dtor)(void*);
} FlowBC;

/// This type represents a temperature boundary condition defined by a function,
/// a context, and a destructor.
typedef struct ThermalBC {
  /// type of boundary condition
  TDyThermalBCType type;
  /// context pointer
  void *context;
  /// vectorized function for computing boundary values
  PetscErrorCode (*compute)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);
  /// destructor
  PetscErrorCode (*dtor)(void*);
} ThermalBC;

/// This type represents a saline concentration boundary condition defined by a
/// function, a context, and a destructor.
typedef struct SalinityBC {
  /// type of boundary condition
  TDySalinityBCType type;
  /// context pointer
  void *context;
  /// vectorized function for computing boundary values
  PetscErrorCode (*compute)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);
  /// destructor
  PetscErrorCode (*dtor)(void*);
} SalinityBC;

/// This type holds flow and thermal boundary conditions that can be
/// associated with a face set via the hash map below.
typedef struct BoundaryConditions {
  FlowBC     flow_bc;
  ThermalBC  thermal_bc;
  SalinityBC salinity_bc;
} BoundaryConditions;

// This type is a convenient mechanism for retrieving face indices from a
// face set.
typedef struct BoundaryFaces {
  /// number of boundary faces
  PetscInt  num_faces;
  /// indices of boundary faces
  PetscInt *faces;
} BoundaryFaces;

PETSC_INTERN PetscErrorCode BoundaryFacesCreate(DM, PetscInt, BoundaryFaces*);
PETSC_INTERN PetscErrorCode BoundaryFacesDestroy(BoundaryFaces);

// This defines as hash map that maps a sparse set of face set indices to
// flow, thermal, and saline concentration boundary conditions.
KHASH_MAP_INIT_INT(TDY_BC, BoundaryConditions)

// This defines as hash map that maps a sparse set of face set indices to
// a set of boundary faces.
KHASH_MAP_INIT_INT(TDY_BFACE, BoundaryFaces)

/// This type gathers settings related to boundary and source/sink conditions.
typedef struct Conditions {

  /// Contexts provided for sources and sinks.
  void* forcing_context;
  void* energy_forcing_context;
  void* salinity_source_context;

  /// Compute momentum source/sink contributions at a set of given points.
  PetscErrorCode (*compute_forcing)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);

  /// Compute energy source/sink contributions at a set of given points.
  PetscErrorCode (*compute_energy_forcing)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);

  /// Compute salinity source/sink contributions at a set of given points.
  PetscErrorCode (*compute_salinity_source)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);

  /// Context destructors.
  PetscErrorCode (*forcing_dtor)(void*);
  PetscErrorCode (*energy_forcing_dtor)(void*);
  PetscErrorCode (*salinity_source_dtor)(void*);

  /// Mapping of face set indices to boundary conditions.
  khash_t(TDY_BC) *bcs;

  /// Mapping of face set indices to boundary faces.
  khash_t(TDY_BFACE) *bfaces;
} Conditions;

// conditions creation/destruction
PETSC_INTERN PetscErrorCode ConditionsCreate(Conditions**);
PETSC_INTERN PetscErrorCode ConditionsDestroy(Conditions*);

// source/sink setup functions
PETSC_INTERN PetscErrorCode ConditionsSetForcing(Conditions*, void*, PetscErrorCode(*)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetEnergyForcing(Conditions*, void*, PetscErrorCode(*)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetSalinitySource(Conditions*, void*, PetscErrorCode(*)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*), PetscErrorCode (*)(void*));

// source/sink query functions
PETSC_INTERN PetscBool ConditionsHasForcing(Conditions*);
PETSC_INTERN PetscBool ConditionsHasEnergyForcing(Conditions*);
PETSC_INTERN PetscBool ConditionsHasSalinitySource(Conditions*);

// conditions computation
// TODO: Change to PETSC_INTERN when we fix demo/steady/steady.c
PETSC_EXTERN PetscErrorCode ConditionsComputeForcing(Conditions*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeEnergyForcing(Conditions*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeSalinitySource(Conditions*,PetscReal,PetscInt,PetscReal*,PetscReal*);

// boundary condition setup and query functions
PETSC_INTERN PetscInt ConditionsNumFaceSets(Conditions*);
PETSC_INTERN PetscBool ConditionsHaveBCs(Conditions*, PetscInt);
PETSC_INTERN PetscErrorCode ConditionsSetBCs(Conditions*, PetscInt, BoundaryConditions);
PETSC_INTERN PetscErrorCode ConditionsGetBCs(Conditions*, PetscInt, BoundaryConditions*);
PETSC_INTERN PetscErrorCode ConditionsGetAllBCs(Conditions*, PetscInt*, BoundaryConditions*);

// boundary face setup and query functions
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryFaces(Conditions*, PetscInt, BoundaryFaces);
PETSC_INTERN PetscErrorCode ConditionsGetBoundaryFaces(Conditions*, PetscInt, BoundaryFaces*);
PETSC_INTERN PetscErrorCode ConditionsGetAllBoundaryFaces(Conditions*, PetscInt*, BoundaryFaces*);

// boundary condition convenience constructor functions
PETSC_INTERN PetscErrorCode CreateConstantPressureBC(FlowBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateConstantVelocityBC(FlowBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateSeepageBC(FlowBC*);
PETSC_INTERN PetscErrorCode CreateConstantTemperatureBC(ThermalBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateConstantHeatFluxBC(ThermalBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateConstantSalinityBC(SalinityBC*,PetscReal);

// vectorized functions that enforce boundary conditions at a given time
PETSC_INTERN PetscErrorCode EnforceFlowBC(FlowBC*, PetscReal, PetscInt, PetscReal*, PetscReal*);
PETSC_INTERN PetscErrorCode EnforceThermalBC(ThermalBC*, PetscReal, PetscInt, PetscReal*, PetscReal*);
PETSC_INTERN PetscErrorCode EnforceSalinityBC(SalinityBC*, PetscReal, PetscInt, PetscReal*, PetscReal*);

#endif

