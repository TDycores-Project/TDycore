#if !defined(TDYCHARACTERISTICCURVESIMPL_H)
#define TDYCHARACTERISTICCURVESIMPL_H

#include <petsc.h>

typedef enum {
  REL_PERM_FUNC_IRMAY=0,
  REL_PERM_FUNC_MUALEM=1
} TDyRelPermFuncType;

typedef enum {
  SAT_FUNC_GARDNER=0,
  SAT_FUNC_VAN_GENUCHTEN=1
} TDySatFuncType;


typedef struct {
  PetscInt *SatFuncType;         /* type of saturation function */
  PetscInt *RelPermFuncType;     /* type of relative permeability */
  PetscReal *sr;                 /* residual saturation (min) [1] */
  PetscReal *gardner_m;          /* parameter used in Garnder saturation [-] */
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

} CharacteristicCurve;

PETSC_INTERN PetscErrorCode CharacteristicCurveCreate(PetscInt,CharacteristicCurve**);
PETSC_INTERN PetscErrorCode CharacteristicCurveDestroy(CharacteristicCurve*);
PETSC_INTERN PetscErrorCode RelativePermeability_Mualem_SetupSmooth(CharacteristicCurve*,PetscInt);


PETSC_INTERN void RelativePermeability_Mualem(PetscReal,PetscReal,PetscReal*,PetscReal,PetscReal*,PetscReal*);
PETSC_INTERN void RelativePermeability_Irmay(PetscReal,PetscReal,PetscReal*,PetscReal*);

PETSC_INTERN void PressureSaturation_VanGenuchten(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN void PressureSaturation_Gardner(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

#endif
