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
PetscErrorCode ComputeWaterDensity_BatzleWang(PetscReal P, PetscReal T, PetscReal S, PetscReal dS, PetscReal d2S, PetscReal *den, PetscReal *dden_dP, PetscReal *dden_dPsi, PetscReal *d2den_dP2) {
  
  PetscFunctionBegin;

  PetscReal pw;
  PetscReal dpw_drho;
  PetscReal Pa_to_MPa = 1.e-6;

  pw = 1 + 1.e-6*(-80.*T - 3.3 * pow(T,2) +0.00175*pow(T,3)+489. * P*Pa_to_MPa - 2.*T *P*Pa_to_MPa + 0.016 *P*Pa_to_MPa *pow(T,2) -
		  1.3e-5*pow(T,3)*P*Pa_to_MPa - 0.333 * pow(P*Pa_to_MPa,2) - 0.002*T *pow(P*Pa_to_MPa,2));
  pw = pw * 1000.;
  *den = pw + S*(0.668 + 0.44*S + 1e-6 * (300. *P*Pa_to_MPa - 2400.*P*Pa_to_MPa*S +T*(80.*3.*T-3300.*S -
							       13.*P*Pa_to_MPa + 47.*P*Pa_to_MPa*S)))*1000.;
  // *den = 998.;

  dpw_drho = 1.6e-6 * (489. - 2.*T + 0.016 * pow(T,2) - 1.3e-5 * pow(T,3) - 2. *0.333 * P * Pa_to_MPa - 0.0002 * 2 *T * P);
  dpw_drho = dpw_drho * 1000.;
  
  *dden_dP = dpw_drho + 1000. * S * 1e-6 * (300. - 2400. * S + T * 13. + T * 47. * S);
  
  *dden_dPsi = (0.668 + 0.44 * d2S + 1e-6 * 300 * P - 1e-6 * 2400. * P * d2S + 1e-6 * pow(T,2) * 80. * 3. - d2S * 1e-6 * T * 3300. - 1e-6 * T * 13. * P + d2S * 1e-6 * 47 * T * P) * 1000.;
  *d2den_dP2 = 0.0;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterDensity(PetscReal p, PetscReal T, PetscReal S, PetscReal dS, PetscReal d2S, PetscInt density_type, PetscReal *den, PetscReal *dden_dP, PetscReal *d2den_dP2, PetscReal *dden_dPsi) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (density_type) {
  case WATER_DENSITY_CONSTANT :
    ierr = ComputeWaterDensity_Constant(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
    break;
  case WATER_DENSITY_EXPONENTIAL :
    ierr = ComputeWaterDensity_Exponential(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
    break;
  case WATER_DENSITY_BATZLE_AND_WANG :
    ierr = ComputeWaterDensity_Constant(p,den,dden_dP,d2den_dP2); CHKERRQ(ierr);
    *dden_dPsi = 0.0;
    //  ierr = ComputeWaterDensity_BatzleWang(p,T,S,dS,d2S,den,dden_dP,dden_dPsi,d2den_dP2); CHKERRQ(ierr);
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
PetscErrorCode ComputeWaterViscosity_BatzleWang(PetscReal p, PetscReal T, PetscReal S, PetscReal *vis, PetscReal *dvis_dP, PetscReal *dvis_dPsi) {

  PetscFunctionBegin;

  *vis = 0.1 + 0.333 * S + (1.65 + 91.9 *pow(S,3))*exp((-0.42*pow((pow(S,0.8)-0.17),2)+0.45)*pow(T,0.8));
  *vis = *vis * 1e-3;
  *dvis_dP = 0.0;
  *dvis_dPsi = 0.0;


  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeWaterViscosity(PetscReal p, PetscReal T, PetscReal S,PetscInt density_type, PetscReal *vis, PetscReal *dvis_dP, PetscReal *d2vis_dP2,PetscReal *dvis_dPsi) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  switch (density_type) {
  case WATER_VISCOSITY_CONSTANT :
    ierr = ComputeWaterViscosity_Constant(p,vis,dvis_dP,d2vis_dP2); CHKERRQ(ierr);
    break;
  case WATER_VISCOSITY_BATZLE_AND_WANG :
    ierr = ComputeWaterViscosity_Constant(p,vis,dvis_dP,d2vis_dP2); CHKERRQ(ierr);
    *dvis_dPsi = 0.0;
    //  ierr = ComputeWaterViscosity_BatzleWang(p,T,S,vis,dvis_dP,dvis_dPsi); CHKERRQ(ierr);
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

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeSalinityFraction(PetscReal Psi, PetscReal mw, PetscReal den, PetscReal *m_nacl, PetscReal *dm_nacl, PetscReal *d2m_nacl) {

  PetscFunctionBegin;

  *m_nacl = (Psi * mw) / ((Psi*mw) + 999.);
  *dm_nacl = (999. * mw) / pow(((Psi*mw) + 999.),2);
  *d2m_nacl = (1998. * Psi * pow(mw,2)) / pow(((Psi*mw) + 999.),3);

  PetscFunctionReturn(0);
}
