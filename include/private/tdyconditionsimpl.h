#if !defined(TDYCONDITIONS_H)
#define TDYCONDITIONS_H

#include <private/tdyoptions.h>

/// This type gathers settings related to boundary and source/sink conditions.
typedef struct TDyConditions {

  /// A context provided to each condition-related function.
  void* context;

  /// Compute momentum source contributions at a given point.
  PetscErrorCode (*compute_forcing)(TDy,PetscReal*,PetscReal*,void*);

  /// Compute energy source contributions at a given point.
  PetscErrorCode (*compute_energy_forcing)(TDy,PetscReal*,PetscReal*,void*);

  /// Compute the pressure at a given point on the boundary.
  PetscErrorCode (*compute_boundary_pressure)(TDy,PetscReal*,PetscReal*,void*);

  /// Compute the temperature at a given point on the boundary.
  PetscErrorCode (*compute_boundary_temperature)(TDy,PetscReal*,PetscReal*,void*);

  /// Compute the components of the component of the velocity normal to a given
  /// point on the boundary.
  PetscErrorCode (*compute_boundary_velocity)(TDy,PetscReal*,PetscReal*,void*);

} TDyConditions;

PETSC_INTERN PetscErrorCode TDyConstantBoundaryPressureFn(TDy,PetscReal*,PetscReal*,void*);
PETSC_INTERN PetscErrorCode TDyConstantBoundaryTemperatureFn(TDy,PetscReal*,PetscReal*,void*);
PETSC_INTERN PetscErrorCode TDyConstantBoundaryVelocityFn(TDy,PetscReal*,PetscReal*,void*);

#endif

