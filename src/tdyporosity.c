#include <private/tdycoreimpl.h>

PetscErrorCode TDySetPorosity(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(tdy->porosity,sizeof(PetscReal)*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(tdy->porosity[c-cStart]));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeporosity = f;
  if (ctx) tdy->porosityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosityValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  for(i=0; i<ni; i++) {
    tdy->porosity[ix[i]] = y[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetPorosityValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd;
  PetscInt junkInt, gref;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = tdy->porosity[c-cStart];
      *ni += 1;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyPorosityFunctionDefault(TDy tdy, double *x, double *por, void *ctx) {
  PetscErrorCode ierr;
  PetscInt dim;
  PetscFunctionBegin;
  TDy tdya;
  *por = 0.25;
  PetscFunctionReturn(0);
}
