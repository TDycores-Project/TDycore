#include "tdycore.h"

PetscErrorCode TDySetPorosity(DM dm,TDy tdy,SpatialFunction f){
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm ,1);
  PetscValidPointer(tdy,2);
  PetscValidPointer(f  ,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscMemzero(tdy->porosity,sizeof(PetscReal)*(cEnd-cStart));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++) f(&(tdy->X[dim*c]),&(tdy->K[c-cStart]));
  PetscFunctionReturn(0);
}


