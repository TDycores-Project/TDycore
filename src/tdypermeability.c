#include <private/tdycoreimpl.h>

PetscErrorCode TDySetPermeabilityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computepermeability = f;
  if (ctx) tdy->permeabilityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetForcingFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeforcing = f;
  if (ctx) tdy->forcingctx = ctx;
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

void RelativePermeability_Irmay(PetscReal m,PetscReal Se,PetscReal *Kr,
                                PetscReal *dKr_dSe) {
  *Kr = PetscPowReal(Se,m);
  if(dKr_dSe) *dKr_dSe = PetscPowReal(Se,m-1)*m;
}

