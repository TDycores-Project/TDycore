#include "tdycore.h"

PetscErrorCode TDyCreate(DM dm,TDy *_tdy){
  TDy            tdy;
  PetscInt       d,dim,p,pStart,pEnd,vStart,vEnd,cStart,cEnd,offset;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_tdy,1);
  ierr = PetscCalloc1(1,&tdy);CHKERRQ(ierr);
  *_tdy = tdy;
  
  /* compute/store plex geometry */  
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(tdy->V));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(tdy->X));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(tdy->N));CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal (dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    if((p >= vStart) && (p < vEnd)){
      ierr = PetscSectionGetOffset(coordSection,p,&offset);CHKERRQ(ierr);
      for(d=0;d<dim;d++) tdy->X[p*dim+d] = coords[offset+d];
    }else{
      ierr = DMPlexComputeCellGeometryFVM(dm,p,&(tdy->V[p]),&(tdy->X[p*dim]),&(tdy->N[p*dim]));CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  
  /* allocate space for a full tensor perm for each cell */
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);  
  ierr = PetscMalloc(dim*dim*(cEnd-cStart)*sizeof(PetscReal),&(tdy->K));CHKERRQ(ierr);

  /* initialize method information to null */
  tdy->vmap = NULL; tdy->emap = NULL; tdy->Alocal = NULL; tdy->Flocal = NULL;
  
  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy(TDy *_tdy){
  TDy            tdy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_tdy,1);
  tdy = *_tdy; *_tdy = NULL;
  if (!tdy) PetscFunctionReturn(0);
  ierr = TDyResetDiscretizationMethod(tdy);CHKERRQ(ierr);
  ierr = PetscFree(tdy->V);CHKERRQ(ierr);
  ierr = PetscFree(tdy->X);CHKERRQ(ierr);
  ierr = PetscFree(tdy->N);CHKERRQ(ierr);
  ierr = PetscFree(tdy->K);CHKERRQ(ierr);  
  ierr = PetscFree(tdy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyResetDiscretizationMethod(TDy tdy){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  if (tdy->vmap  ) { ierr = PetscFree(tdy->vmap  );CHKERRQ(ierr); }
  if (tdy->emap  ) { ierr = PetscFree(tdy->emap  );CHKERRQ(ierr); }
  if (tdy->Alocal) { ierr = PetscFree(tdy->Alocal);CHKERRQ(ierr); }
  if (tdy->Flocal) { ierr = PetscFree(tdy->Flocal);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDiscretizationMethod(DM dm,TDy tdy,TDyMethod method){
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscValidPointer( dm,1);
  PetscValidPointer(tdy,2);
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  if (tdy->method != method) { ierr = TDyResetDiscretizationMethod(tdy);CHKERRQ(ierr); }
  tdy->method = method;
  switch (method) {
  case TWO_POINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"TWO_POINT_FLUX is not yet implemented");
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"MULTIPOINT_FLUX is not yet implemented");
  case MIXED_FINITE_ELEMENT:
    SETERRQ(comm,PETSC_ERR_SUP,"MIXED_FINITE_ELEMENT is not yet implemented");
  case WHEELER_YOTOV:
    ierr = TDyWYInitialize(dm,tdy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetForcingFunction(TDy tdy,SpatialFunction f){
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(  f,2);
  tdy->forcing = f;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDirichletFunction(TDy tdy,SpatialFunction f){
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(  f,2);
  tdy->dirichlet = f;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyComputeSystem(DM dm,TDy tdy,Mat K,Vec F){
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  switch (tdy->method) {
  case TWO_POINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"TWO_POINT_FLUX is not yet implemented");
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"MULTIPOINT_FLUX is not yet implemented");
  case MIXED_FINITE_ELEMENT:
    SETERRQ(comm,PETSC_ERR_SUP,"MIXED_FINITE_ELEMENT is not yet implemented");
  case WHEELER_YOTOV:
    ierr = TDyWYComputeSystem(dm,tdy,K,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
