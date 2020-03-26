#include <private/tdycoreimpl.h>

PetscErrorCode TDySetPermeabilityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computepermeability = f;
  if (ctx) tdy->permeabilityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetThermalConductivityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computethermalconductivity = f;
  if (ctx) tdy->thermalconductivityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetResidualSaturationFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeresidualsaturation = f;
  if (ctx) tdy->residualsaturationctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetForcingFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeforcing = f;
  if (ctx) tdy->forcingctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetEnergyForcingFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeenergyforcing = f;
  if (ctx) tdy->energyforcingctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetTemperatureDirichletValueFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computetemperaturedirichletvalue = f;
  if (ctx) tdy->temperaturedirichletvaluectx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDirichletValueFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computedirichletvalue = f;
  if (ctx) tdy->dirichletvaluectx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDirichletFluxFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computedirichletflux = f;
  if (ctx) tdy->dirichletfluxctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityScalar(TDy tdy,SpatialFunction f) {
  PetscInt dim,dim2,i,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    f(&(tdy->X[dim*c]),&(tdy->K[dim2*(c-cStart)]));
    for(i=1; i<dim; i++) {
      tdy->K[dim2*(c-cStart)+i*dim+i] = tdy->K[dim2*(c-cStart)];
    }
  }
  ierr = PetscMemcpy(tdy->K0,tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityDiagonal(TDy tdy,SpatialFunction f) {
  PetscInt dim,dim2,i,c,cStart,cEnd;
  PetscReal val[3];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    f(&(tdy->X[dim*c]),&(val[0]));
    for(i=0; i<dim; i++) {
      tdy->K[dim2*(c-cStart)+i*dim+i] = val[i];
    }
  }
  ierr = PetscMemcpy(tdy->K0,tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBlockPermeabilityValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i,j;
  PetscInt dim,dim2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;

  for(i=0; i<ni; i++) {
    for(j=0; j<dim2; j++) tdy->K0[ix[i]*dim2 + j] = y[i*dim2+j];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityTensor(TDy tdy,SpatialFunction f) {
  PetscInt dim,dim2,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    f(&(tdy->X[dim*c]),&(tdy->K[dim2*(c-cStart)]));
  }
  ierr = PetscMemcpy(tdy->K0,tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCellPermeability(TDy tdy,PetscInt c,PetscReal *K) {
  PetscInt dim2,i,cStart;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&i); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim2); CHKERRQ(ierr);
  dim2 *= dim2;  
  for(i=0;i<dim2;i++) tdy->K[dim2*(c-cStart)+i] = K[i];
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

PetscErrorCode TDyGetBlockPermeabilityValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd,j;
  PetscInt junkInt, gref;
  PetscInt dim,dim2;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      for(j=0; j<dim2; j++) {
        y[*ni] = tdy->K0[(c-cStart)*dim2 + j];
        *ni += 1;
      }
    }
  }

  PetscFunctionReturn(0);
}
