#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>

PetscErrorCode CharacteristicCurveCreate(PetscInt ncells, CharacteristicCurve **_cc){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  *_cc = (CharacteristicCurve *)malloc(sizeof(CharacteristicCurve));

  ierr = PetscMalloc(ncells*sizeof(PetscInt),&((*_cc)->SatFuncType)); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscInt),&(*_cc)->RelPermFuncType); CHKERRQ(ierr);

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->Kr); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->dKr_dS); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->S); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->dS_dP); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->d2S_dP2); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->dS_dT); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->sr); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->gardner_m); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->vg_m); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->mualem_m); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->irmay_m); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->gardner_n); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->vg_alpha); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->mualem_poly_low); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&(*_cc)->mualem_poly_coeffs,ncells,4); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicCurveDestroy(CharacteristicCurve *cc){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  if (cc->SatFuncType    ) { ierr = PetscFree(cc->SatFuncType    ); CHKERRQ(ierr); }
  if (cc->RelPermFuncType) { ierr = PetscFree(cc->RelPermFuncType); CHKERRQ(ierr); }
  if (cc->Kr             ) { ierr = PetscFree(cc->Kr             ); CHKERRQ(ierr); }
  if (cc->dKr_dS         ) { ierr = PetscFree(cc->dKr_dS         ); CHKERRQ(ierr); }
  if (cc->S              ) { ierr = PetscFree(cc->S              ); CHKERRQ(ierr); }
  if (cc->dS_dP          ) { ierr = PetscFree(cc->dS_dP          ); CHKERRQ(ierr); }
  if (cc->d2S_dP2        ) { ierr = PetscFree(cc->d2S_dP2        ); CHKERRQ(ierr); }
  if (cc->dS_dT          ) { ierr = PetscFree(cc->dS_dT          ); CHKERRQ(ierr); }
  if (cc->sr             ) { ierr = PetscFree(cc->sr             ); CHKERRQ(ierr); }
  if (cc->gardner_m      ) { ierr = PetscFree(cc->gardner_m      ); CHKERRQ(ierr); }
  if (cc->vg_m           ) { ierr = PetscFree(cc->vg_m           ); CHKERRQ(ierr); }
  if (cc->irmay_m        ) { ierr = PetscFree(cc->irmay_m        ); CHKERRQ(ierr); }
  if (cc->mualem_m       ) { ierr = PetscFree(cc->mualem_m       ); CHKERRQ(ierr); }
  if (cc->gardner_n      ) { ierr = PetscFree(cc->gardner_n      ); CHKERRQ(ierr); }
  if (cc->vg_alpha       ) { ierr = PetscFree(cc->vg_alpha       ); CHKERRQ(ierr); }
  if (cc->mualem_poly_low    ) { ierr = PetscFree(cc->mualem_poly_low    ); CHKERRQ(ierr); }
  
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetResidualSaturationValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;

  for(i=0; i<ni; i++) {
    cc->sr[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCharacteristicCurveMualemValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->mualem_m[ix[i]] = y[i];
  }

  ierr = RelativePermeability_Mualem_SetupSmooth(cc, ni); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCharacteristicCurveNValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->gardner_n[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCharacteristicCurveVanGenuchtenValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[], const PetscScalar z[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->vg_m[ix[i]] = y[i];
    cc->vg_alpha[ix[i]] = z[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetSaturationValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd;
  PetscInt junkInt, gref;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  CharacteristicCurve *cc = tdy->cc;
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = cc->S[c-cStart];
      *ni += 1;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetLiquidMassValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd;
  PetscInt junkInt, gref;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  CharacteristicCurve *cc = tdy->cc;
  MaterialProp *matprop = tdy->matprop;
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = tdy->rho[c-cStart]*matprop->porosity[c-cStart]*cc->S[c-cStart]*tdy->V[c];
      *ni += 1;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetCharacteristicCurveMValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd;
  PetscInt junkInt, gref;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  CharacteristicCurve *cc = tdy->cc;
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = cc->mualem_m[c-cStart];
      *ni += 1;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetCharacteristicCurveAlphaValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd;
  PetscInt junkInt, gref;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  CharacteristicCurve *cc = tdy->cc;
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = cc->vg_alpha[c-cStart];
      *ni += 1;
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Compute value and derivate of relative permeability using Irmay function
///
/// @param [in] m            parameter for Irmay function
/// @param [in] Se           effective saturation
/// @param [inout] *Kr       value of relative permeability
/// @param [inout] *dKr_dSe  derivate of relative permeability w.r.t. Se
///
/// kr = Se^m  if Se < 1.0
///    = 1     otherwise
///
/// dkr/dSe = m * Se^{m-1}  if Se < 1.0
///         = 0             otherwise
///
void RelativePermeability_Irmay(PetscReal m,PetscReal Se,PetscReal *Kr,
                                PetscReal *dKr_dSe) {
  *Kr = 1.0;
  if (dKr_dSe) *dKr_dSe = 0.0;

  if (Se>=1.0) return;

  *Kr = PetscPowReal(Se,m);
  if(dKr_dSe) *dKr_dSe = PetscPowReal(Se,m-1)*m;
}

/* -------------------------------------------------------------------------- */
/// Compute value and derivate of relative permeability using Mualem function
///
/// @param [in] m            parameter for Mualem function
/// @param [in] Se           effective saturation
/// @param [inout] *Kr       value of relative permeability
/// @param [inout] *dKr_dSe  derivate of relative permeability w.r.t. Se
///
/// kr = Se^{0.5} * [ 1 - (1 - Se^{1/m})^m ]^2     if P < P_ref or Se < 1.0
///    = 1                                         otherwise
///
/// dkr/dSe = 0.5 Se^{-0.5} [ 1 - (1 - Se^{1/m})^m ] +
///           Se^{0.5}      2 * Se^{1/m - 1/} * (1 - Se^{1/m})^{m - 1} * (1 - (1 - Se^{1/m})^m)  if Se < 1.0
///         = 0                                                                                  otherwise
///
void RelativePermeability_Mualem_NonSmooth(PetscReal m,PetscReal Se,PetscReal *Kr,
				 PetscReal *dKr_dSe) {
  PetscReal Se_one_over_m,tmp;

  *Kr = 1.0;
  if(dKr_dSe) *dKr_dSe = 0.0;

  if (Se>=1.0) return;

  Se_one_over_m = PetscPowReal(Se,1/m);
  tmp = PetscPowReal(1-Se_one_over_m,m);
  (*Kr)  = PetscSqrtReal(Se);
  (*Kr) *= PetscSqr(1-tmp);
  if(dKr_dSe){
    (*dKr_dSe)  = 0.5*(*Kr)/Se;
    (*dKr_dSe) += 2*PetscPowReal(Se,1/m-0.5) * PetscPowReal(1-Se_one_over_m,m-1) * (1-PetscPowReal(1-Se_one_over_m,m));
  }
}

/* -------------------------------------------------------------------------- */
///
/// Sets up a cubic polynomial interpolation for relative permeability following
/// PFLOTRAN's approach of smoothing relative permeability functions
///
/// @param [in] x1           low value of x
/// @param [in] x2           high value of x
/// @param [inout] *rhs      rhs vector (input) and coefficients (output)
///
///  f(x) = a0 + a1 * x + a2 * x^2 + a2 * x^3
///  df_dx = a1 + 2 * a2 * x + 3 * a3 * x^2
///
/// Constraints:
/// f(low)      = rel_perm_fn(x)      = f1
/// f(high)     = 1.0                 = f2
/// df_dx(low)  = drv_rel_perm_fn(x)  = df1_dx
/// df_dy(high) = 0.0                 = df2_dx
///
/// Linear system:
/// a0 + a1 * x1 + a2 * x1^2   + a2 * x1^3     = f1
/// a0 + a1 * x2 + a2 * x1^2   + a2 * x2^3     = f2
///      a1      + 2 * a2 * x1 + 3 * a3 * x1^2 = df1_dx
///      a1      + 2 * a2 * x2 + 3 * a3 * x2^2 = df2_dx
///
PetscErrorCode CubicPolynomialSetup(PetscReal x1, PetscReal x2, PetscReal *rhs) {

  PetscInt n = 4, nrhs = 1;

  PetscReal A[16] = {
     1.0,                  1.0,                  0.0,                      0.0,
     x2,                   x1,                   1.0,                      1.0,
     PetscPowReal(x2,2.0), PetscPowReal(x1,2.0), 2.0*x2,                   2.0*x1,
     PetscPowReal(x2,3.0), PetscPowReal(x1,3.0), 3.0*PetscPowReal(x2,2.0), 3.0*PetscPowReal(x1,2.0)
  };

   PetscInt lda=4, ldb=4;
   PetscInt info; // success/failure from LAPACK
   PetscInt ipiv[n]; // pivot indices

   dgesv_( &n, &nrhs, A, &lda, ipiv, rhs, &ldb, &info );

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
///
/// Computes the value and derivative of the cubic polynomial
///
/// @param [in] *coeffs  coefficients of a cubic polynomals
/// @param [in] x        value
/// @param [out] *f      function evaluated at value
/// @param [out] *df_dx  derivative of the function evaluated at value = x
///
///  f(x) = a0 + a1 * x + a2 * x^2 + a2 * x^3
///  df_dx = a1 + 2 * a2 * x + 3 * a3 * x^2
///
PetscErrorCode CubicPolynomialEvaluate(PetscReal *coeffs, PetscReal x, PetscReal *f, PetscReal *df_dx) {

  PetscFunctionBegin;

  *f = coeffs[0] + coeffs[1]*x + coeffs[2]*PetscPowReal(x,2.0) + coeffs[3]*PetscPowReal(x,3.0);
  *df_dx = coeffs[1] + 2.0*coeffs[2]*x + 3.0*coeffs[3]*PetscPowReal(x,2.0);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
///
/// Sets up cubic polynomial smoothing for Mualem relative permeability function
///
/// @param [inout] *cc  Charcteristic curve
/// @param [in] ncells  Number of cells
///
PetscErrorCode RelativePermeability_Mualem_SetupSmooth(CharacteristicCurve *cc, PetscInt ncells){

  PetscFunctionBegin;

  for (PetscInt c=0; c<ncells; c++){

    PetscReal m = cc->vg_m[c];
    PetscReal poly_low = cc->mualem_poly_low[c];
    PetscReal poly_high = 1.0;

    PetscReal Kr, dKr_dSe;
    PetscErrorCode ierr;
    RelativePermeability_Mualem_NonSmooth(m, poly_low, &Kr, &dKr_dSe);

    cc->mualem_poly_coeffs[c][0] = 1.0;
    cc->mualem_poly_coeffs[c][1] = Kr;
    cc->mualem_poly_coeffs[c][2] = 0.0;
    cc->mualem_poly_coeffs[c][3] = dKr_dSe;

    ierr = CubicPolynomialSetup(poly_low, poly_high, cc->mualem_poly_coeffs[c]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
///
/// Computes relative permeability using Mualem function
///
/// @param [in] m         Mualem parameter
/// @param [in] poly_low  Value of Se above which polynomial smoothing should be done
/// @param [in] *coeffs   Coefficents of cubic polynomial
/// @param [in] Se        Effective saturation
/// @param [out] *Kr      Relative permeability
/// @param [out] *dKr_dSe Derivative of relative permeability
///
void RelativePermeability_Mualem(PetscReal m, PetscReal poly_low, PetscReal *coeffs, PetscReal Se,PetscReal *Kr,PetscReal *dKr_dSe) {

  if (Se > poly_low) {
    CubicPolynomialEvaluate(coeffs, Se, Kr, dKr_dSe);
  } else {
    RelativePermeability_Mualem_NonSmooth(m, Se, Kr, dKr_dSe);
  }
}

/* -------------------------------------------------------------------------- */
/// Compute value and derivates of saturation using Gardner function
///
/// @param [in] n            parameter for Gardner function
/// @param [in] m            parameter for Gardner function
/// @param [in] alpha        parameter for Gardner function
/// @param [in] Sr           residual saturation
/// @param [in] Pc           capillary pressure
/// @param [inout] S         value of saturation
/// @param [inout] *dS_dP    first derivate of saturation w.r.t. pressure
/// @param [inout] *d2S_dP2  second derivate of saturation w.r.t. pressure
///
/// Se = exp(-alpha/m*Pc) if Pc < 0.0
///    = 1                otherwise
///
/// S = (1 - Sr)*Se + Sr
///
/// dSe/dPc = -alpha/m*exp(-alpha/m*Pc-1) if Pc < 0.0
///         = 0                            otherwise
///
/// dS/dP  = dS/dPc * dPc/dP
///        = dSe/dPc * dS/dSe * dPc/dP
///
/// and dS/dSe = 1 - Sr; dPc/dP = -1
///
void PressureSaturation_Gardner(PetscReal n,PetscReal m,PetscReal alpha, PetscReal Sr,
                                PetscReal Pc,PetscReal *S,PetscReal *dS_dP,PetscReal *d2S_dP2) {
  if(Pc < 0) { /* if Pc < 0 then P > Pref and Se = 1 */
    *S = 1;
    if(dS_dP) *dS_dP = 0;
    if(d2S_dP2) *d2S_dP2 =0.0;
  }else{
    PetscReal Se, dSe_dPc;
    Se = PetscExpReal(-alpha*Pc/m);
    *S = (1.0 - Sr)*Se + Sr;
    if(dS_dP) {
      dSe_dPc = -alpha/m*PetscExpReal(-alpha*Pc/m);
      *dS_dP = -dSe_dPc*(1.0 - Sr);
      if (d2S_dP2) {
        PetscReal d2Se_dPc2;
        d2Se_dPc2 = -alpha/m*dSe_dPc;
        *d2S_dP2 = (1.0-Sr)*d2Se_dPc2;
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/// Compute value and derivates of saturation using Van Genuchten function
///
/// @param [in] m            parameter for van Genuchten function
/// @param [in] alpha        parameter for van Genuchten function
/// @param [in] Sr           residual saturation
/// @param [in] Pc           capillary pressure
/// @param [inout] S         value of saturation
/// @param [inout] *dS_dP    first derivate of saturation w.r.t. pressure
/// @param [inout] *d2S_dP2  second derivate of saturation w.r.t. pressure
///
///  Se = [1 + (a * Pc)^(1/(1-m))]^{-m}   if  Pc < 0
///     = 1                               otherwise
///
///  Let n = 1/(1-m)
///
/// dSe/dPc = - [m * n * a * (a*Pc)^n] / denom
/// denom   = (a*Pc) * [ (a*Pc)^n + 1]^{m+1}
///
/// S = (1 - Sr)*Se + Sr
///
/// dSe/dPc = -alpha/m*exp(-alpha/m*Pc-1) if Pc < 0.0
///         = 0                            otherwise
///
/// dS/dP  = dS/dPc * dPc/dP
///        = dSe/dPc * dS/dSe * dPc/dP
///
/// and dS/dSe = 1 - Sr; dPc/dP = -1
///
void PressureSaturation_VanGenuchten(PetscReal m,PetscReal alpha,  PetscReal Sr,
				     PetscReal Pc,PetscReal *S,PetscReal *dS_dP,PetscReal *d2S_dP2) {
  PetscReal pc_alpha,pc_alpha_n,one_plus_pc_alpha_n,n;
  if(Pc <= 0) { 
    *S = 1;
    if(dS_dP) *dS_dP = 0;
    if(d2S_dP2) *d2S_dP2 =0.0;
  }else{
    PetscReal Se, dSe_dPc;
    n = 1/(1-m);
    pc_alpha = Pc*alpha;
    pc_alpha_n = PetscPowReal(pc_alpha,n);
    one_plus_pc_alpha_n = 1+pc_alpha_n;
    Se = PetscPowReal(one_plus_pc_alpha_n,-m);
    *S = (1.0 - Sr)*Se + Sr;
    if(dS_dP){
      dSe_dPc = -m*n*alpha*pc_alpha_n/(pc_alpha*PetscPowReal(one_plus_pc_alpha_n,m+1));
      *dS_dP = -dSe_dPc*(1.0 - Sr);
      if (d2S_dP2) {
        PetscReal d2Se_dPc2;
        d2Se_dPc2 = m*n*(pc_alpha_n)*PetscPowReal(one_plus_pc_alpha_n,-m-2.0)*((m*n+1.0)*pc_alpha_n-n+1)/PetscPowReal(Pc,2.0);
        *d2S_dP2 = (1.0-Sr)*d2Se_dPc2;
      }
    }
  }
}
