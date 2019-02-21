#include "tdycore.h"

const char *const TDyMethods[] = {
  "TWO_POINT_FLUX",
  "MULTIPOINT_FLUX",
  "BDM",
  "WY",
  /* */
  "TDyMethod","TDY_METHOD_",NULL};

PetscClassId TDY_CLASSID = 0;

PETSC_EXTERN PetscBool TDyPackageInitialized;
PetscBool TDyPackageInitialized = PETSC_FALSE;

PetscErrorCode TDyFinalizePackage(void)
{
  PetscFunctionBegin;
  TDyPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TDyPackageInitialized) PetscFunctionReturn(0);
  TDyPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("TDy",&TDY_CLASSID);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("tdy",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(TDY_CLASSID);CHKERRQ(ierr);}
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("tdy",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventDeactivateClass(TDY_CLASSID);CHKERRQ(ierr);}
  }
  ierr = PetscRegisterFinalize(TDyFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreate(DM dm,TDy *_tdy){
  TDy            tdy;
  PetscInt       d,dim,p,pStart,pEnd,vStart,vEnd,cStart,cEnd,eStart,eEnd,offset,nc;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  PetscValidPointer(_tdy,1);
  ierr = TDyInitializePackage();CHKERRQ(ierr);
  *_tdy = NULL;
  ierr = PetscHeaderCreate(tdy,TDY_CLASSID,"TDy","TDy","TDy",comm,TDyDestroy,TDyView);CHKERRQ(ierr);
  *_tdy = tdy;
  
  /* compute/store plex geometry */
  tdy->dm = dm;
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
  tdy->faces = NULL; tdy->LtoG = NULL; tdy->orient = NULL;

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
  if (tdy->faces ) { ierr = PetscFree(tdy->faces );CHKERRQ(ierr); }
  if (tdy->LtoG  ) { ierr = PetscFree(tdy->LtoG  );CHKERRQ(ierr); }
  if (tdy->orient) { ierr = PetscFree(tdy->orient);CHKERRQ(ierr); }
  if (tdy->quad  ) { ierr = PetscQuadratureDestroy(&(tdy->quad));CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}


PetscErrorCode TDyView(TDy tdy,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)tdy)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tdy,1,viewer,2);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetFromOptions(TDy tdy)
{
  PetscErrorCode ierr;
  PetscBool flg;
  TDyMethod method = WY;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)tdy);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_method","Discretization method","TDySetDiscretizationMethod",TDyMethods,(PetscEnum)method,(PetscEnum *)&method,&flg);
  if (flg && (method != tdy->method)) { ierr = TDySetDiscretizationMethod(tdy,method); }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDiscretizationMethod(TDy tdy,TDyMethod method){
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscValidPointer(tdy,2);
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)(tdy->dm),&comm);CHKERRQ(ierr);
  if (tdy->method != method) { ierr = TDyResetDiscretizationMethod(tdy);CHKERRQ(ierr); }
  tdy->method = method;
  switch (method) {
  case TWO_POINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"TWO_POINT_FLUX is not yet implemented");
    break;
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"MULTIPOINT_FLUX is not yet implemented");
    break;
  case BDM:
    ierr = TDyBDMInitialize(tdy);CHKERRQ(ierr);
    break;
  case WY:
    ierr = TDyWYInitialize(tdy);CHKERRQ(ierr);
    break;
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
    break;
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for MULTIPOINT_FLUX");
    break;
  case BDM:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for BDM");
    break;
  case WY:
    ierr = TSSetIFunction(ts,NULL,TDyWYResidual,tdy);CHKERRQ(ierr);
    break;
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

PetscErrorCode TDyComputeSystem(TDy tdy,Mat K,Vec F){
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)(tdy->dm),&comm);CHKERRQ(ierr);
  switch (tdy->method) {
  case TWO_POINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"TWO_POINT_FLUX is not yet implemented");
    break;
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"MULTIPOINT_FLUX is not yet implemented");
    break;
  case BDM:
    ierr = TDyBDMComputeSystem(tdy,K,F);CHKERRQ(ierr);
    break;
  case WY:
    ierr = TDyWYComputeSystem(tdy,K,F);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateState(TDy tdy,PetscReal *P){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt  dim,dim2,i,j,c,cStart,cEnd;
  PetscReal Se,dSe_dPc,n=1.0,m=1.0,alpha=1.6717e-5,Kr; /* FIX: generalize */
  ierr = DMGetDimension(tdy->dm,&dim);CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
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

PetscErrorCode TDyQuadrature(PetscQuadrature q,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *x,*w;
  PetscInt d,nv=1;
  for(d=0;d<dim;d++) nv *= 2;
  ierr = PetscMalloc1(nv*dim,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(nv    ,&w);CHKERRQ(ierr);
  switch(nv*dim){
  case 2: /* line */
    x[0] = -1.0; w[0] = 1.0;
    x[1] =  1.0; w[1] = 1.0;
    break;
  case 8: /* quad */
    x[0] = -1.0; x[1] = -1.0; w[0] = 1.0;
    x[2] =  1.0; x[3] = -1.0; w[1] = 1.0;
    x[4] = -1.0; x[5] =  1.0; w[2] = 1.0;
    x[6] =  1.0; x[7] =  1.0; w[3] = 1.0;
    break;
  case 24: /* hex */
    x[0]  = -1.0; x[1]  = -1.0; x[2]  = -1.0; w[0] = 1.0;
    x[3]  =  1.0; x[4]  = -1.0; x[5]  = -1.0; w[1] = 1.0;
    x[6]  = -1.0; x[7]  =  1.0; x[8]  = -1.0; w[2] = 1.0;
    x[9]  =  1.0; x[10] =  1.0; x[11] = -1.0; w[3] = 1.0;
    x[12] = -1.0; x[13] = -1.0; x[14] =  1.0; w[4] = 1.0;
    x[15] =  1.0; x[16] = -1.0; x[17] =  1.0; w[5] = 1.0;
    x[18] = -1.0; x[19] =  1.0; x[20] =  1.0; w[6] = 1.0;
    x[21] =  1.0; x[22] =  1.0; x[23] =  1.0; w[7] = 1.0;
  }
  ierr = PetscQuadratureSetData(q,dim,1,nv,x,w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscInt TDyGetNumberOfCellVertices(DM dm){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,c,q,i,cStart,cEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  nq = -1;
  for(c=cStart;c<cEnd;c++){
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    q = 0;
    for (i=0;i<closureSize*2;i+=2){
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh cells must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

PetscInt TDyGetNumberOfFaceVertices(DM dm){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,f,q,i,fStart,fEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  nq = -1;
  for(f=fStart;f<fEnd;f++){
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    q = 0;
    for (i=0;i<closureSize*2;i+=2){
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh faces must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

/* Returns

   |x-y|_L1

   where x, and y are dim-dimensional arrays
 */
PetscReal TDyL1norm(PetscReal *x,PetscReal *y,PetscInt dim){
  PetscInt i;
  PetscReal norm;
  norm = 0;
  for(i=0;i<dim;i++) norm += PetscAbsReal(x[i]-y[i]);
  return norm;
}

/* Returns

   a * (b - c)

   where a, b, and c are dim-dimensional arrays
 */
PetscReal TDyADotBMinusC(PetscReal *a,PetscReal *b,PetscReal *c,PetscInt dim){
  PetscInt i;
  PetscReal norm;
  norm = 0;
  for(i=0;i<dim;i++) norm += a[i]*(b[i]-c[i]);
  return norm;
}

PetscReal TDyADotB(PetscReal *a,PetscReal *b,PetscInt dim){
  PetscInt i;
  PetscReal norm = 0;
  for(i=0;i<dim;i++) norm += a[i]*b[i];
  return norm;
}

/* Check if the image of the quadrature point is coincident with
   the vertex, if so we create a map:

   map(cell,local_cell_vertex) --> vertex

   Allocates memory inside routine, user must free.
*/
PetscErrorCode TDyCreateCellVertexMap(TDy tdy,PetscInt **map){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt dim,i,v,vStart,vEnd,nv,c,cStart,cEnd,closureSize,*closure;
  PetscQuadrature quad;
  PetscReal x[24],DF[72],DFinv[72],J[8];
  DM dm = tdy->dm;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  nv = TDyGetNumberOfCellVertices(dm);
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quad);CHKERRQ(ierr);
  ierr = TDyQuadrature(quad,dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc(nv*(cEnd-cStart)*sizeof(PetscInt),map);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quad,x,DF,DFinv,J);CHKERRQ(ierr);
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    for(v=0;v<nv;v++){
      for (i=0;i<closureSize*2;i+=2){
	if ((closure[i] >= vStart) && (closure[i] < vEnd)) {
	  if (TDyL1norm(&(x[v*dim]),&(tdy->X[closure[i]*dim]),dim) > 1e-12) continue;
	  (*map)[c*nv+v] = closure[i];
	  break;
	}
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* Create a map:

   map(cell,local_cell_vertex,direction) --> face

   To do this, I loop over the vertices of this cell and find
   connected faces. Then I use the local ordering of the vertices to
   determine where the normal of this face points. Finally I check if
   the normal points into the cell. If so, then the index is given a
   negative as a flag later in the assembly process. Since the Hasse
   diagram always begins with cells, there isn't a conflict with 0
   being a possible point.
*/
PetscErrorCode TDyCreateCellVertexDirFaceMap(TDy tdy,PetscInt **map){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt d,dim,i,f,fStart,fEnd,v,nv,q,c,cStart,cEnd,closureSize,*closure,fclosureSize,*fclosure,local_dirs[24];
  DM dm = tdy->dm;
  if(!(tdy->vmap)){
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Must first create TDyCreateCellVertexMap on tdy->vmap");
  }
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if(dim == 2){
    local_dirs[0] = 2; local_dirs[1] = 1;
    local_dirs[2] = 3; local_dirs[3] = 0;
    local_dirs[4] = 0; local_dirs[5] = 3;
    local_dirs[6] = 1; local_dirs[7] = 2;
  }else if(dim == 3){
    local_dirs[0]  = 6; local_dirs[1]  = 5; local_dirs[2]  = 3;
    local_dirs[3]  = 7; local_dirs[4]  = 4; local_dirs[5]  = 2;
    local_dirs[6]  = 4; local_dirs[7]  = 7; local_dirs[8]  = 1;
    local_dirs[9]  = 5; local_dirs[10] = 6; local_dirs[11] = 0;
    local_dirs[12] = 2; local_dirs[13] = 1; local_dirs[14] = 7;
    local_dirs[15] = 3; local_dirs[16] = 0; local_dirs[17] = 6;
    local_dirs[18] = 0; local_dirs[19] = 3; local_dirs[20] = 5;
    local_dirs[21] = 1; local_dirs[22] = 2; local_dirs[23] = 4;
  }
  nv = TDyGetNumberOfCellVertices(dm);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*nv*(cEnd-cStart)*sizeof(PetscInt),map);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    for(q=0;q<nv;q++){
      for (i=0;i<closureSize*2;i+=2){
        if ((closure[i] >= fStart) && (closure[i] < fEnd)) {
          fclosure = NULL;
          ierr = DMPlexGetTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,&fclosure);CHKERRQ(ierr);
          for(f=0;f<fclosureSize*2;f+=2){
            if (fclosure[f] == tdy->vmap[c*nv+q]){
              for(v=0;v<fclosureSize*2;v+=2){
                for(d=0;d<dim;d++){
                  if (fclosure[v] == tdy->vmap[c*nv+local_dirs[q*dim+d]]) {
                    (*map)[c*nv*dim+q*dim+d] = closure[i];
                    if (TDyADotBMinusC(&(tdy->N[closure[i]*dim]),&(tdy->X[closure[i]*dim]),&(tdy->X[c*dim]),dim) < 0) {
                      (*map)[c*nv*dim+q*dim+d] *= -1;
                      break;
                    }
                  }
                }
              }
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,&fclosure);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDyComputeErrorNorms(TDy tdy,Vec U,PetscReal *normp,PetscReal *normv,PetscReal *normd){
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)(tdy->dm),&comm);CHKERRQ(ierr);
  switch (tdy->method) {
  case TWO_POINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"TWO_POINT_FLUX is not yet implemented");
    break;
  case MULTIPOINT_FLUX:
    SETERRQ(comm,PETSC_ERR_SUP,"MULTIPOINT_FLUX is not yet implemented");
    break;
  case BDM:
    if(normp != NULL) { *normp = TDyBDMPressureNorm(tdy,U); }
    if(normv != NULL) { *normv = TDyBDMVelocityNorm(tdy,U); }
    if(normd != NULL) { *normd = TDyBDMDivergenceNorm(tdy,U); }
    break;
  case WY:
    if(normv || normd){
      ierr = TDyWYRecoverVelocity(tdy,U);CHKERRQ(ierr);
    }
    if(normp != NULL) { *normp = TDyWYPressureNorm(tdy,U); }
    if(normv != NULL) { *normv = TDyWYVelocityNorm(tdy); }
    if(normd != NULL) { *normd = TDyWYDivergenceNorm(tdy); }
    break;
  }
  PetscFunctionReturn(0);
}
