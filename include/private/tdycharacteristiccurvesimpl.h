#if !defined(TDYCHARACTERISTICCURVESIMPL_H)
#define TDYCHARACTERISTICCURVESIMPL_H

#include <petsc.h>

/// This type enumerates the different parameterizations for saturation.
typedef enum {
  SAT_FUNC_GARDNER=0,
  SAT_FUNC_VAN_GENUCHTEN=1
} SaturationType;

/// A Saturation model associates a saturation function and its parameters with
/// each point in a given domain.
typedef struct Saturation {
  /// Number of points in a domain for each saturation model is defined.
  PetscInt num_points[2];
  /// Arrays of indices of points that use a given saturation function type.
  PetscInt *points[2];
  /// Saturation parameters for points of each saturation function type.
  /// The ordering and number of the parameters depends on the model:
  /// 1. The Gardner model associates 3 parameters (n, m, alpha) with each point.
  /// 2. The Van Genuchten model associates 2 parameters (m, alpha) with each point.
  PetscReal *parameters[2];

  /// Mappings from point indices to their offsets in each model type. This
  /// allows updates of specific sets of points using SaturationComputeOnPoints.
  void *point_maps[2];
} Saturation;

/// This type enumerates the different parameterizations for relative
/// permeability.
typedef enum {
  REL_PERM_FUNC_IRMAY=0,
  REL_PERM_FUNC_MUALEM=1
} RelativePermeabilityType;

/// A RelativePermeability model associates a function and its parameters with
/// each point in a given domain.
typedef struct RelativePermeability {
  /// Number of points in a domain for which a relative permeability is defined.
  PetscInt num_points[2];
  /// Arrays of indices of points that use a given saturation function type.
  PetscInt *points[2];
  /// Relative permeability parameters for points of each saturation function type.
  /// The ordering and number of the parameters depends on the model:
  /// 1. The Irmay model associates a single parameter m with each point.
  /// 2. The Mualem model associates 6 parameters with each point: the model
  ///    parameter m; the value of the effective saturation above which the
  ///    model employs a cubic interpolation polynomial; and the 4 coefficients
  ///    of the cubic polynomial.
  PetscReal *parameters[2];

  /// Mappings from point indices to their offsets in each model type. This
  /// allows updates of specific sets of points using
  /// RelativePermeabilityComputeOnPoints.
  void *point_maps[2];
} RelativePermeability;

/// This type collects parameterized functions that describe the saturation
/// and relative permeability in a domain.
typedef struct {
  /// saturation model
  Saturation* saturation;
  /// relative permeability model
  RelativePermeability* rel_perm;

} CharacteristicCurves;

PETSC_INTERN PetscErrorCode CharacteristicCurvesCreate(CharacteristicCurves**);
PETSC_INTERN PetscErrorCode CharacteristicCurvesDestroy(CharacteristicCurves*);

PETSC_INTERN PetscErrorCode SaturationCreate(Saturation**);
PETSC_INTERN PetscErrorCode SaturationDestroy(Saturation*);
PETSC_INTERN PetscErrorCode SaturationSetType(Saturation*,SaturationType,PetscInt,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode SaturationCompute(Saturation*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode SaturationComputeOnPoints(Saturation*,PetscInt,PetscInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

PETSC_INTERN PetscErrorCode RelativePermeabilityCreate(RelativePermeability**);
PETSC_INTERN PetscErrorCode RelativePermeabilityDestroy(RelativePermeability*);
PETSC_INTERN PetscErrorCode RelativePermeabilitySetType(RelativePermeability*,RelativePermeabilityType,PetscInt,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode RelativePermeabilityCompute(RelativePermeability*,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode RelativePermeabilityComputeOnPoints(RelativePermeability*,PetscInt,PetscInt*,PetscReal*,PetscReal*,PetscReal*);

PETSC_INTERN PetscErrorCode RelativePermeability_Mualem_GetSmoothingCoeffs(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*);
PETSC_INTERN void RelativePermeability_Mualem(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal,PetscReal*,PetscReal*);
PETSC_INTERN void RelativePermeability_Irmay(PetscReal,PetscReal,PetscReal*,PetscReal*);

PETSC_INTERN void PressureSaturation_VanGenuchten(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN void PressureSaturation_Gardner(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

#endif
