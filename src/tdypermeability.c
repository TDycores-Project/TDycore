#include "tdycore.h"

PetscErrorCode TDySetPermeabilityScalar(DM dm,TDy tdy,SpatialFunction f){
  PetscInt dim,dim2,i,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm ,1);
  PetscValidPointer(tdy,2);
  PetscValidPointer(f  ,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    f(&(tdy->X[dim*c]),&(tdy->K[dim2*(c-cStart)]));
    for(i=1;i<dim;i++){
      tdy->K[dim2*(c-cStart)+i*dim+i] = tdy->K[dim2*(c-cStart)];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityDiagonal(DM dm,TDy tdy,SpatialFunction f){
  PetscInt dim,dim2,i,c,cStart,cEnd;
  PetscReal val[3];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm ,1);
  PetscValidPointer(tdy,2);
  PetscValidPointer(f  ,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    f(&(tdy->X[dim*c]),&(val[0]));
    for(i=0;i<dim;i++){
      tdy->K[dim2*(c-cStart)+i*dim+i] = val[i];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityTensor(DM dm,TDy tdy,SpatialFunction f){
  PetscInt dim,dim2,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm ,1);
  PetscValidPointer(tdy,2);
  PetscValidPointer(f  ,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(tdy->K,sizeof(PetscReal)*dim2*(cEnd-cStart));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    f(&(tdy->X[dim*c]),&(tdy->K[dim2*(c-cStart)]));
  }
  PetscFunctionReturn(0);
}
