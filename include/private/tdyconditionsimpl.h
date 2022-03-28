#if !defined(TDYCONDITIONS_H)
#define TDYCONDITIONS_H

#include <private/tdyoptions.h>

#include <petsc/private/khash/khash.h>

/// This type represents a boundary condition implemented by a function, a
/// context, and a destructor.
typedef struct BoundaryCondition {
  /// context pointer
  void *context;
  /// vectorized function for computing boundary values
  PetscErrorCode (*compute)(void*,PetscInt,PetscReal*,PetscReal*);
  /// destructor
  void (*dtor)(void*);
} BoundaryCondition;

// This maps a sparse set of face set indices to boundary conditions.
KHASH_MAP_INIT_INT(TDY_BC, BoundaryCondition)

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

// boundary condition setup functions
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryPressure(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), void (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryTemperature(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*) , void (*)(void*));
PETSC_INTERN PetscErrorCode ConditionsSetBoundaryVelocity(Conditions*, void*, PetscErrorCode(*)(void*,PetscInt,PetscReal*,PetscReal*), void (*)(void*));

// boundary condition query functions
PETSC_INTERN PetscBool ConditionsHasBoundaryPressure(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundaryTemperature(Conditions*);
PETSC_INTERN PetscBool ConditionsHasBoundaryVelocity(Conditions*);

// boundary condition computation functions
// TODO: Change to PETSC_INTERN when we fix demo/steady steady.c
PETSC_EXTERN PetscErrorCode ConditionsComputeBoundaryPressure(Conditions*,PetscInt,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeBoundaryTemperature(Conditions*,PetscInt,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode ConditionsComputeBoundaryVelocity(Conditions*,PetscInt,PetscInt,PetscReal*,PetscReal*);

// boundary condition convenience functions
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundaryPressure(Conditions*,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundaryTemperature(Conditions*,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode ConditionsSetConstantBoundaryVelocity(Conditions*,PetscInt,PetscReal);

#endif

