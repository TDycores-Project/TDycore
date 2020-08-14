#include <private/tdycoreimpl.h>
#include <private/tdyeosimpl.h>

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterDensity_Constant(PetscReal p, PetscReal *den, PetscReal *dden_dP, PetscReal *d2den_dP2) {

  PetscFunctionBegin;

  *den = 998.0;
  *dden_dP = 0.0;
  *d2den_dP2 = 0.0;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterDensity_Exponential(PetscReal p, PetscReal *den, PetscReal *dden_dP, PetscReal *d2den_dP2) {

  PetscReal ReferenceDensity = 997.16e0;
  PetscReal ReferencePressure = 101325.e0;
  PetscReal WaterCompressibility = 1.e-8;

  PetscFunctionBegin;

  if (p < ReferencePressure) p = ReferencePressure;
  *den = ReferenceDensity*exp(WaterCompressibility*(p - ReferencePressure));
  *dden_dP = WaterCompressibility * (*den);
  *d2den_dP2 = WaterCompressibility * (*dden_dP);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterDensity(PetscReal p, PetscInt density_type, PetscReal *den, PetscReal *dden_dP, PetscReal *d2den_dP2) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (density_type) {
  case WATER_DENSITY_CONSTANT :
    ierr = ComputeWaterDensity_Constant(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
    break;
  case WATER_DENSITY_EXPONENTIAL :
    ierr = ComputeWaterDensity_Exponential(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown water density function");
    break;
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterViscosity_Constant(PetscReal p, PetscReal *vis, PetscReal *dvis_dP, PetscReal *d2vis_dP2) {

  PetscFunctionBegin;

  *vis = 9.94e-4;
  *dvis_dP = 0.0;
  *d2vis_dP2 = 0.0;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterViscosity(PetscReal p, PetscInt density_type, PetscReal *vis, PetscReal *dvis_dP, PetscReal *d2vis_dP2) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (density_type) {
  case WATER_VISCOSITY_CONSTANT :
    ierr = ComputeWaterViscosity_Constant(p,vis,dvis_dP,d2vis_dP2); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown water viscosity function");
    break;
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterEnthalpy_Constant(PetscReal t, PetscReal p, PetscReal *hw, PetscReal *dhw_dP, PetscReal *dhw_dT) {

  PetscFunctionBegin;

  *hw = 1.8890; // J/mol
  *dhw_dP = 0.0;
  *dhw_dT = 0.0;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterEnthalpy(PetscReal t, PetscReal p, PetscInt enthalpy_type, PetscReal *hw, PetscReal *dhw_dP, PetscReal *dhw_dT) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (enthalpy_type) {
  case WATER_ENTHALPY_CONSTANT :
    ierr = ComputeWaterEnthalpy_Constant(t,p,hw,dhw_dP,dhw_dT); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown water enthalpy function");
    break;
  }

  PetscFunctionReturn(0);
}
