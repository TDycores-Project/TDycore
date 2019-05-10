#include <private/tdycoreimpl.h>

PetscErrorCode TDySetPorosity(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(tdy->porosity,sizeof(PetscReal)*(cEnd-cStart));
  CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(tdy->K[c-cStart]));
  PetscFunctionReturn(0);
}


