#if !defined(TDYCONDITIONS_H)
#define TDYCONDITIONS_H

#include <private/tdyoptions.h>

#include <petsc/private/khash/khash.h>

/// Types of mechanical boundary conditions
typedef enum {
  TDY_UNDEFINED_FLOW_BC = 0,
  TDY_PRESSURE_BC,
  TDY_VELOCITY_BC,
  TDY_SEEPAGE_BC
} FlowBCType;

/// This type represents a flow (pressure/velocity/head) boundary condition
/// defined by a function, a context, and a destructor.
typedef struct FlowBC {
  /// type of boundary condition
  FlowBCType type;
  /// context pointer
  void *context;
  /// vectorized function for computing boundary values
  PetscErrorCode (*compute)(void*,PetscInt,PetscReal*,PetscReal*);
  /// destructor
  void (*dtor)(void*);
} FlowBC;

/// Types of thermal boundary conditions
typedef enum {
  TDY_UNDEFINED_THERMAL_BC = 0,
  TDY_TEMPERATURE_BC,
  TDY_HEAT_FLUX_BC,
} ThermalBCType;

/// This type represents a thermal boundary condition defined by a function, a
/// context, and a destructor.
typedef struct ThermalBC {
  /// type of boundary condition
  ThermalBCType type;
  /// context pointer
  void *context;
  /// vectorized function for computing boundary values
  PetscErrorCode (*compute)(void*,PetscInt,PetscReal*,PetscReal*);
  /// destructor
  void (*dtor)(void*);
} ThermalBC;

/// This type holds flow and thermal boundary conditions that can be
/// associated with a face set via the hash map below.
typedef struct BoundaryConditions {
  FlowBC    flow_bc;
  ThermalBC thermal_bc;
} BoundaryConditions;

// This maps a sparse set of face set indices to flow and thermal boundary
// conditions.
KHASH_MAP_INIT_INT(TDY_BC, BoundaryConditions)

/// This type gathers settings related to boundary and source/sink conditions.
typedef struct Conditions {

  /// Contexts provided for sources and sinks.
  void* forcing_context;
  void* energy_forcing_context;

  /// Compute momentum source contributions at a set of given points.
  PetscErrorCode (*compute_forcing)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Compute energy source contributions at a set of given points.
  PetscErrorCode (*compute_energy_forcing)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Context destructors.
  void (*forcing_dtor)(void*);
  void (*energy_forcing_dtor)(void*);

  /// Mapping of face set indices to boundary conditions.
  khash_t(TDY_BC) *bcs;

} Conditions;

// conditions creation/destruction
PETSC_INTERN PetscErrorCode ConditionsCreate(Conditions**);
PETSC_INTERN PetscErrorCode ConditionsDestroy(Conditions*);

// source/sink setup functions
PETSC_INTERN PetscErrorCode ConditionsSetForcing(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), void (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetEnergyForcing(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), void (*)(void*));

// source/sink query functions
PETSC_INTERN PetscBool ConditionsHasForcing(Conditions*);
PETSC_INTERN PetscBool ConditionsHasEnergyForcing(Conditions*);

// source/sink computation functions
// TODO: Change to PETSC_INTERN when we fix demo/steady steady.c
PETSC_EXTERN PetscErrorCode ConditionsComputeForcing(Conditions*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeEnergyForcing(Conditions*,PetscInt,PetscReal*,PetscReal*);

// boundary condition setup and query functions
PETSC_INTERN PetscErrorCode ConditionsSetBCs(Conditions*, PetscInt, BoundaryConditions);
PETSC_INTERN PetscErrorCode ConditionsGetBCs(Conditions*, PetscInt, BoundaryConditions*);

// boundary condition convenience functions
PETSC_INTERN PetscErrorCode CreateConstantPressureBC(FlowBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateConstantVelocityBC(FlowBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateSeepageBC(FlowBC*);
PETSC_INTERN PetscErrorCode CreateConstantTemperatureBC(ThermalBC*,PetscReal);
PETSC_INTERN PetscErrorCode CreateConstantHeatFluxBC(ThermalBC*,PetscReal);

// vectorized functions that enforce boundary conditions
PETSC_INTERN PetscErrorCode FlowBCEnforce(FlowBC*, PetscInt, PetscReal*, PetscReal*);
PETSC_INTERN PetscErrorCode ThermalBCEnforce(ThermalBC*, PetscInt, PetscReal*, PetscReal*);

#endif

