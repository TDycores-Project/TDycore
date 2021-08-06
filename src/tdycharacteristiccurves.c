#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdymemoryimpl.h>

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
  if (cc->gardner_n      ) { ierr = PetscFree(cc->gardner_n              ); CHKERRQ(ierr); }
  if (cc->vg_alpha       ) { ierr = PetscFree(cc->vg_alpha          ); CHKERRQ(ierr); }
  
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

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->mualem_m[ix[i]] = y[i];
    cc->vg_m[ix[i]] = y[i];
  }

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
void RelativePermeability_Mualem(PetscReal m,PetscReal Se,PetscReal *Kr,
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


