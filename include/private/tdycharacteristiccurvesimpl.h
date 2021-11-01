#if !defined(TDYCHARACTERISTICCURVESIMPL_H)
#define TDYCHARACTERISTICCURVESIMPL_H

#include <petsc.h>

/// Question: do we allow saturation and relative permeability parameters to
/// vary point by point?

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
  PetscInt num_points;
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
} RelativePermeability;

/// This type collects parameterized functions that describe the saturation
/// and relative permeability in a domain.
typedef struct {
  /// saturation model
  Saturation* saturation;
  /// relative permeability model
  RelativePermeability* rel_perm;

#if 0
  PetscInt *SatFuncType;         /* type of saturation function */
  PetscInt *RelPermFuncType;     /* type of relative permeability */
  PetscReal *sr;                 /* residual saturation (min) [1] */
  PetscReal *gardner_m;          /* parameter used in Gardner saturation [-] */
  PetscReal *vg_m;               /* parameter used in VanGenuchten saturation function [-] */
  PetscReal *irmay_m;            /* parameter used in Irmay relative permeability function [-] */
  PetscReal *mualem_m;           /* parameter used in Mualem relative permeability function [-] */
  PetscReal *gardner_n;          /* parameter used in Gardner saturation function [-] */
  PetscReal *vg_alpha;           /* parameter used in VanGenuchten saturation function [-] */
  PetscReal *S,                  /* saturation for each cell */
            *dS_dP,              /* derivative of saturation wrt pressure for each cell [Pa^-1] */
            *d2S_dP2,            /* second derivative of saturation wrt pressure for each cell [Pa^-2] */
            *dS_dT;              /* derivate of saturation wrt to temperature for each cell [K^-1] */
  PetscReal *Kr, *dKr_dS;        /* relative permeability for each cell [1] */

  PetscReal *mualem_poly_low;    /* value of effecitive saturation above which cubic interpolation is used */
  PetscReal **mualem_poly_coeffs;/* coefficients for cubic polynomial */
#endif

} CharacteristicCurves;

PETSC_INTERN PetscErrorCode CharacteristicCurvesCreate(CharacteristicCurves**);
PETSC_INTERN PetscErrorCode CharacteristicCurvesDestroy(CharacteristicCurves*);

PETSC_INTERN PetscErrorCode SaturationCreate(Saturation**);
PETSC_INTERN PetscErrorCode SaturationDestroy(Saturation*);
PETSC_INTERN PetscErrorCode SaturationSetType(Saturation*,SaturationType,PetscInt,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode SaturationCompute(Saturation*,SaturationType,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

PETSC_INTERN PetscErrorCode RelativePermeabilityCreate(RelativePermeability**);
PETSC_INTERN PetscErrorCode RelativePermeabilityDestroy(RelativePermeability*);
PETSC_INTERN PetscErrorCode RelativePermeabilitySetType(RelativePermeability*,RelativePermeabilityType,PetscInt,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode RelativePermeabilityCompute(RelativePermeability*,RelativePermeabilityType,PetscReal*,PetscReal*,PetscReal*);

//PETSC_INTERN PetscErrorCode RelativePermeability_Mualem_SetupSmooth(CharacteristicCurves*,PetscInt);

PETSC_INTERN void RelativePermeability_Mualem(PetscReal,PetscReal,PetscReal*,PetscReal,PetscReal*,PetscReal*);
PETSC_INTERN void RelativePermeability_Irmay(PetscReal,PetscReal,PetscReal*,PetscReal*);

PETSC_INTERN void PressureSaturation_VanGenuchten(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN void PressureSaturation_Gardner(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

#endif
