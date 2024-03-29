#include <private/tdycoreimpl.h>
#include <private/tdyeosimpl.h>

/* ---------------------------------------------------------------- */
static PetscErrorCode ComputeWaterDensity_Constant(PetscReal p, PetscReal *den, PetscReal *dden_dP, PetscReal *d2den_dP2) {

  PetscFunctionBegin;

  *den = 998.0;
  *dden_dP = 0.0;
  *d2den_dP2 = 0.0;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode ComputeWaterDensity_Exponential(PetscReal p, PetscReal *den, PetscReal *dden_dP, PetscReal *d2den_dP2) {

  PetscReal ReferenceDensity = 997.16e0;
  PetscReal ReferencePressure = 101325.e0;
  PetscReal WaterCompressibility = 1.e-8;

  PetscFunctionBegin;

  //if (p < ReferencePressure) p = ReferencePressure;
  if (p < ReferencePressure) {
    *den = ReferenceDensity;
    *dden_dP = 0.0;
    *d2den_dP2 = 0.0;
  } else {
    *den = ReferenceDensity*exp(WaterCompressibility*(p - ReferencePressure));
    *dden_dP = WaterCompressibility * (*den);
    *d2den_dP2 = WaterCompressibility * (*dden_dP);
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode ComputeWaterDensity_BatzleWang(
  PetscReal P, PetscReal T, PetscReal S, PetscReal *den) {

  PetscFunctionBegin;

  PetscReal pw;
  PetscReal Pa_to_MPa = 1.e-6;

  pw = 1 + 1.e-6*(-80.*T - 3.3 * pow(T,2) +0.00175*pow(T,3)+489. * P*Pa_to_MPa - 2.*T *P*Pa_to_MPa + 0.016 *P*Pa_to_MPa *pow(T,2) -
		  1.3e-5*pow(T,3)*P*Pa_to_MPa - 0.333 * pow(P*Pa_to_MPa,2) - 0.002*T *pow(P*Pa_to_MPa,2));
  pw *= 1000.;

  *den = pw + S*(0.668 + 0.44*S + 1e-6 * (300. *P*Pa_to_MPa - 2400.*P*Pa_to_MPa*S +T*(80.*3.*T-3300.*S -
 						       13.*P*Pa_to_MPa + 47.*P*Pa_to_MPa*S)))*1000.;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
// FIXME: S here seems to mean "saline concentration", not "saturation".
PetscErrorCode EOSComputeWaterDensity(EOS *eos,
  PetscReal p, PetscReal T, PetscReal S,
  PetscReal *den, PetscReal *dden_dP,
  PetscReal *d2den_dP2) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (eos->density_type) {
    case WATER_DENSITY_CONSTANT :
      ierr = ComputeWaterDensity_Constant(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
      break;
    case WATER_DENSITY_EXPONENTIAL :
      ierr = ComputeWaterDensity_Exponential(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
      break;
    case WATER_DENSITY_BATZLE_AND_WANG :
      ierr = ComputeWaterDensity_BatzleWang(p,T,S,den); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown water density function");
      break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeWaterViscosity_Constant(PetscReal p,
                                                     PetscReal *vis,
                                                     PetscReal *dvis_dP,
                                                     PetscReal *d2vis_dP2) {

  PetscFunctionBegin;

  *vis = 9.94e-4;
  *dvis_dP = 0.0;
  *d2vis_dP2 = 0.0;

  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeWaterViscosity_BatzleWang(
  PetscReal p, PetscReal T, PetscReal S, PetscReal *vis) {

  PetscFunctionBegin;

  *vis = 0.1 +
         0.333 * S +
         (1.65 + 91.9 *pow(S,3))*exp((-0.42*pow((pow(S,0.8)-0.17),2)+0.045)*pow(T,0.8));
  *vis *= 1e-3;

  PetscFunctionReturn(0);
}

// FIXME: S here seems to mean "saline concentration", not "saturation".
PetscErrorCode EOSComputeWaterViscosity(EOS *eos,
  PetscReal p, PetscReal T, PetscReal S,
  PetscReal *vis, PetscReal *dvis_dP, PetscReal *d2vis_dP2) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (eos->viscosity_type) {
    case WATER_VISCOSITY_CONSTANT :
      ierr = ComputeWaterViscosity_Constant(p,vis,dvis_dP,d2vis_dP2); CHKERRQ(ierr);
      break;
    case WATER_VISCOSITY_BATZLE_AND_WANG :
      ierr = ComputeWaterViscosity_BatzleWang(p,T,S,vis); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown water viscosity function");
      break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeWaterEnthalpy_Constant(PetscReal t, PetscReal p,
                                                    PetscReal *hw,
                                                    PetscReal *dhw_dP,
                                                    PetscReal *dhw_dT) {

  PetscFunctionBegin;

  *hw = 1.8890; // J/mol
  *dhw_dP = 0.0;
  *dhw_dT = 0.0;

  PetscFunctionReturn(0);
}

PetscErrorCode EOSComputeWaterEnthalpy(EOS *eos, PetscReal t, PetscReal p,
                                       PetscReal *hw, PetscReal *dhw_dP,
                                       PetscReal *dhw_dT) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (eos->enthalpy_type) {
    case WATER_ENTHALPY_CONSTANT :
      ierr = ComputeWaterEnthalpy_Constant(t,p,hw,dhw_dP,dhw_dT); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown water enthalpy function");
      break;
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode EOSComputeSalinityFraction(EOS* eos,
  PetscReal Psi, PetscReal mw, PetscReal den, PetscReal *m_nacl) {

  PetscFunctionBegin;

  *m_nacl = (Psi * mw) / ((Psi*mw) + 999.);

  PetscFunctionReturn(0);
}
