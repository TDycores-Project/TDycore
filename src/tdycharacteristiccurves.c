#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdymemoryimpl.h>

PetscErrorCode CharacteristicCurveCreate(PetscInt ncells, CharacteristicCurve **_cc){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  *_cc = (CharacteristicCurve *)malloc(sizeof(struct _CharacteristicCurve));

  ierr = PetscMalloc(ncells*sizeof(PetscInt),&(*_cc)->SatFuncType); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscInt),&(*_cc)->RelPermFuncType); CHKERRQ(ierr);

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->Kr); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->dKr_dS); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->S); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->dS_dP); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->d2S_dP2); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->dS_dT); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->sr); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->m); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->n); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_cc)->alpha); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetResidualSaturationValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;

  for(i=0; i<ni; i++) {
    cc->sr[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCharacteristicCurveMValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->m[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCharacteristicCurveNValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->n[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCharacteristicCurveAlphaValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  CharacteristicCurve *cc = tdy->cc;
  for(i=0; i<ni; i++) {
    cc->alpha[ix[i]] = y[i];
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
  MaterialProp matprop = tdy->matprop;
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
      y[*ni] = cc->m[c-cStart];
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
      y[*ni] = cc->alpha[c-cStart];
      *ni += 1;
    }
  }

  PetscFunctionReturn(0);
}

void RelativePermeability_Irmay(PetscReal m,PetscReal Se,PetscReal *Kr,
                                PetscReal *dKr_dSe) {
  *Kr = 1.0;
  if (dKr_dSe) *dKr_dSe = 0.0;

  if (Se>=1.0) return;

  *Kr = PetscPowReal(Se,m);
  if(dKr_dSe) *dKr_dSe = PetscPowReal(Se,m-1)*m;
}

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


