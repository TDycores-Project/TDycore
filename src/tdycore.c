#include "tdycore.h"

PetscErrorCode TDyCreate(DM dm,TDy *_tdy){
  TDy            tdy;
  PetscInt       d,dim,p,pStart,pEnd,vStart,vEnd,cStart,cEnd,eStart,eEnd,offset,nc;
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
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd);CHKERRQ(ierr);  
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
      if((dim == 3) && (p >= eStart) && (p < eEnd)) continue;
      ierr = DMPlexComputeCellGeometryFVM(dm,p,&(tdy->V[p]),&(tdy->X[p*dim]),&(tdy->N[p*dim]));CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  
  /* allocate space for a full tensor perm for each cell */
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  nc   = cEnd-cStart;
  ierr = PetscMalloc(dim*dim*nc*sizeof(PetscReal),&(tdy->K0));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim*nc*sizeof(PetscReal),&(tdy->K ));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->porosity));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->Kr));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->S));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->dS_dP));CHKERRQ(ierr);

  /* problem constants FIX: add mutators */
  tdy->rho  = 998;
  tdy->mu   = 9.94e-4;
  tdy->Sr   = 0.15;
  tdy->Ss   = 1;
  tdy->Pref = 101325;
  tdy->gravity[0] = 0; tdy->gravity[1] = 0; tdy->gravity[2] = 0;
  tdy->gravity[dim-1] = -9.81;
  
  /* initialize method information to null */
  tdy->vmap = NULL; tdy->emap = NULL; tdy->Alocal = NULL; tdy->Flocal = NULL; tdy->quad = NULL;

  /* initialize function pointers */
  tdy->forcing = NULL ; tdy->dirichlet = NULL ; tdy->flux = NULL ;
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
  ierr = PetscFree(tdy->porosity);CHKERRQ(ierr);
  ierr = PetscFree(tdy->Kr);CHKERRQ(ierr);
  ierr = PetscFree(tdy->S);CHKERRQ(ierr);
  ierr = PetscFree(tdy->dS_dP);CHKERRQ(ierr);
  ierr = PetscFree(tdy->K);CHKERRQ(ierr);
  ierr = PetscFree(tdy->K0);CHKERRQ(ierr);
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
  if (tdy->vel   ) { ierr = PetscFree(tdy->vel   );CHKERRQ(ierr); } 
  if (tdy->fmap  ) { ierr = PetscFree(tdy->fmap  );CHKERRQ(ierr); } 
  if (tdy->quad  ) { ierr = PetscQuadratureDestroy(&(tdy->quad));CHKERRQ(ierr); }
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

PetscErrorCode TDySetIFunction(TS ts,TDy tdy){
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscValidPointer( ts,1);
  PetscValidPointer(tdy,2);
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  switch (tdy->method) {
  case TWO_POINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for TWO_POINT_FLUX");
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for MULTIPOINT_FLUX");
  case MIXED_FINITE_ELEMENT:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for MIXED_FINITE_ELEMENT");
  case WHEELER_YOTOV:
    ierr = TSSetIFunction(ts,NULL,TDyWYResidual,tdy);CHKERRQ(ierr);
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

PetscErrorCode TDySetDirichletFlux(TDy tdy,SpatialFunction f){
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(  f,2);
  tdy->flux = f;
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

PetscErrorCode TDyUpdateState(DM dm,TDy tdy,PetscReal *P){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt  dim,dim2,i,j,c,cStart,cEnd;
  PetscReal Se,dSe_dPc,n=1.0,m=1.0,alpha=1.6717e-5,Kr; /* FIX: generalize */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    i = c-cStart;
    PressureSaturation_Gardner(n,m,alpha,tdy->Pref-P[i],&Se,&dSe_dPc);
    RelativePermeability_Irmay(m,Se,&Kr,NULL);
    tdy->S[i] = (tdy->Ss-tdy->Sr)*Se+tdy->Sr;
    tdy->dS_dP[i] = -dSe_dPc/(tdy->Ss-tdy->Sr);
    for(j=0;j<dim2;j++) tdy->K[i*dim2+j] = tdy->K0[i*dim2+j] * Kr;
  }
  PetscFunctionReturn(0);
}
