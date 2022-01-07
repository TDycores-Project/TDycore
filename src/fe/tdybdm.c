#include <private/tdycoreimpl.h>
#include <private/tdyfeimpl.h>
#include <private/tdybdmimpl.h>
#include <petscblaslapack.h>
#include <tdytimers.h>

/* (dim*vertices_per_cell+1)^2 */
#define MAX_LOCAL_SIZE 625

PetscErrorCode TDyBDMSetQuadrature(TDy tdy, TDyQuadratureType qtype) {
  PetscFunctionBegin;
  TDyBDM *bdm = tdy->context;
  bdm->qtype = qtype;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreate_BDM(void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Allocate a new context for the WY method.
  TDyBDM *bdm;
  ierr = PetscMalloc(sizeof(TDyBDM), &bdm);
  *context = bdm;

  // initialize data
  bdm->qtype = FULL;
  bdm->vmap = NULL; bdm->emap = NULL;
  bdm->quad = NULL;
  bdm->faces = NULL; bdm->LtoG = NULL; bdm->orient = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy_BDM(void *context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyBDM *bdm = context;

  if (bdm->vmap  ) { ierr = PetscFree(bdm->vmap  ); CHKERRQ(ierr); }
  if (bdm->emap  ) { ierr = PetscFree(bdm->emap  ); CHKERRQ(ierr); }
  if (bdm->fmap  ) { ierr = PetscFree(bdm->fmap  ); CHKERRQ(ierr); }
  if (bdm->faces ) { ierr = PetscFree(bdm->faces ); CHKERRQ(ierr); }
  if (bdm->LtoG  ) { ierr = PetscFree(bdm->LtoG  ); CHKERRQ(ierr); }
  if (bdm->orient) { ierr = PetscFree(bdm->orient); CHKERRQ(ierr); }
  if (bdm->quad  ) { ierr = PetscQuadratureDestroy(&(bdm->quad)); CHKERRQ(ierr); }

  ierr = PetscFree(bdm->V); CHKERRQ(ierr);
  ierr = PetscFree(bdm->X); CHKERRQ(ierr);
  ierr = PetscFree(bdm->N); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetFromOptions_BDM(void *context, TDyOptions *options) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  TDyBDM *bdm = context;

  // Set options.
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: BDM options",""); CHKERRQ(ierr);
  TDyQuadratureType qtype = FULL;
  ierr = PetscOptionsEnum("-tdy_quadrature","Quadrature type for finite element methods",
    "TDyWYSetQuadrature",TDyQuadratureTypes,(PetscEnum)qtype,
    (PetscEnum *)&bdm->qtype,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDMFields_BDM(void *context, DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  // Define 1 DOF on cell center of each cell
  PetscInt dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  PetscFE fe;
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "p_", -1, &fe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "p");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(dm); CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetup_BDM(void *context, DM dm, EOS *eos, MaterialProp *matprop,
                            CharacteristicCurves *cc, Conditions *conditions) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt pStart,pEnd,c,cStart,cEnd,f,f_abs,fStart,fEnd,nfv,ncv,v,vStart,vEnd,
           mStart,mEnd,i,nlocal,closureSize,*closure,d,dim;

  TDyBDM *bdm = context;

  // Compute/store plex geometry.
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  PetscLogEvent t1 = TDyGetTimer("ComputePlexGeometry");
  TDyStartTimer(t1);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  PetscInt eStart, eEnd;
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(bdm->V));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(bdm->X));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(bdm->N));
  CHKERRQ(ierr);
  PetscSection coordSection;
  Vec coordinates;
  PetscReal *coords;
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal (dm, &coordinates); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(PetscInt p=pStart; p<pEnd; p++) {
    if((p >= vStart) && (p < vEnd)) {
      PetscInt offset;
      ierr = PetscSectionGetOffset(coordSection,p,&offset); CHKERRQ(ierr);
      for(PetscInt d=0; d<dim; d++) bdm->X[p*dim+d] = coords[offset+d];
    } else {
      if((dim == 3) && (p >= eStart) && (p < eEnd)) continue;
      PetscLogEvent t11 = TDyGetTimer("DMPlexComputeCellGeometryFVM");
      TDyStartTimer(t11);
      ierr = DMPlexComputeCellGeometryFVM(dm,p,&(bdm->V[p]),
                                          &(bdm->X[p*dim]),
                                          &(bdm->N[p*dim])); CHKERRQ(ierr);
      TDyStopTimer(t11);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);

  /* Get plex limits */
  ierr = DMPlexGetChart        (dm,  &pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

  /* Create H-div section */
  PetscSection sec;
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,2); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Pressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,1,"Velocity"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,1,1); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);

  /* Setup 1 dof per cell for field 0 */
  for(c=cStart; c<cEnd; c++) {
    ierr = PetscSectionSetFieldDof(sec,c,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,c,1); CHKERRQ(ierr);
  }

  /* Setup dofs_per_face considering quads and hexes only */
  PetscInt dofs_per_face = 1;
  for(d=0; d<(dim-1); d++) dofs_per_face *= 2;
  for(f=fStart; f<fEnd; f++) {
    ierr = PetscSectionSetFieldDof(sec,f,1,dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,f,  dofs_per_face); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm,&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  bdm->ncv = TDyGetNumberOfCellVertices(dm);

  /* Build vmap and emap */
  ierr = CreateCellVertexMap(dm, bdm->ncv, bdm->X, &(bdm->vmap)); CHKERRQ(ierr);
  ierr = CreateCellVertexDirFaceMap(dm, bdm->ncv, bdm->X, bdm->N, bdm->vmap,
                                       &(bdm->emap)); CHKERRQ(ierr);

  /* Build map(face,local_vertex) --> vertex */
  nfv = TDyGetNumberOfFaceVertices(dm);
  ierr = PetscMalloc(nfv*(fEnd-fStart)*sizeof(PetscInt),
                     &(bdm->fmap)); CHKERRQ(ierr);
  for(f=fStart; f<fEnd; f++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,
                                      &closureSize,&closure); CHKERRQ(ierr);
    i = 0;
    for(c=0; c<closureSize*2; c+=2) {
      if ((closure[c] < vStart) || (closure[c] >= vEnd)) continue;
      bdm->fmap[nfv*(f-fStart)+i] = closure[c];
      i += 1;
    }
    #if defined(PETSC_USE_DEBUG)
    if(i != nfv) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Irregular number of vertices per face found");
    }
    #endif
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,
                                          &closureSize,&closure); CHKERRQ(ierr);
  }

  /* use vmap, emap, and fmap to build a LtoG map for local element
     assembly */
  ncv = bdm->ncv;
  nlocal = dim*ncv + 1;
  ierr = PetscMalloc((cEnd-cStart)*nlocal*sizeof(PetscInt),
                     &(bdm->LtoG)); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*nlocal*sizeof(PetscInt),
                     &(bdm->orient)); CHKERRQ(ierr);
  PetscBool found;
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&mStart,&mEnd); CHKERRQ(ierr);
    if(mStart<0) mStart = -(mStart+1);
    bdm->LtoG[(c-cStart+1)*nlocal-1] = mStart;
    for(v=0; v<ncv; v++) {
      for(d=0; d<dim; d++) {
        /* which face is this local dof on? */
        f = bdm->emap[(c-cStart)*ncv*dim+v*dim+d];
        f_abs = PetscAbsInt(f);
        ierr = DMPlexGetPointGlobal(dm,f_abs,&mStart,&mEnd); CHKERRQ(ierr);
        if(mStart<0) mStart = -(mStart+1);
        found = PETSC_FALSE;
        for(i=0; i<nfv; i++) {
          if(bdm->vmap[ncv*(c-cStart)+v] == bdm->fmap[nfv*(f_abs-fStart)+i]) {
            bdm->LtoG  [(c-cStart)*nlocal + v*dim + d] = mStart + i;
            bdm->orient[(c-cStart)*nlocal + v*dim + d] = PetscSign(f);
            found = PETSC_TRUE;
          }
        }
        if(!found) {
          SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Could not find a face vertex for this cell");
        }
      }
    }
  }

  /* map(cell,dim,side) --> global_face */
  ierr = PetscMalloc((cEnd-cStart)*PetscPowInt(2,dim)*sizeof(PetscInt),
                     &(bdm->faces)); CHKERRQ(ierr);
  #if defined(PETSC_USE_DEBUG)
  for(c=0; c<((cEnd-cStart)*(2*dim)); c++) { bdm->faces[c] = -1; }
  #endif
  PetscInt s;
  for(c=cStart; c<cEnd; c++) {
    for(d=0; d<dim; d++) {
      for(s=0; s<2; s++) {
        v = s*PetscPowInt(2,d);
        bdm->faces[(c-cStart)*dim*2+d*2+s] = PetscAbsInt(bdm->emap[(c-cStart)*ncv*dim+v*dim+d]);
      }
    }
  }
  #if defined(PETSC_USE_DEBUG)
  for(c=0; c<((cEnd-cStart)*(2*dim)); c++) {
    if(bdm->faces[c] < 0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,dir,side) -> face");
    }
  }
  #endif

  // Initialize material properties.
  PetscInt nc = cEnd-cStart;
  ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(bdm->K)); CHKERRQ(ierr);
  ierr = MaterialPropComputePermeability(matprop, nc, bdm->X, bdm->K); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscReal TDyKDotADotB(PetscReal *K,PetscReal *A,PetscReal *B,PetscInt dim) {
  TDY_START_FUNCTION_TIMER()
  PetscInt i,j;
  PetscReal inner,outer=0;
  for(i=0; i<dim; i++) {
    inner = 0;
    for(j=0; j<dim; j++) {
      inner += K[j*dim+i]*A[j];
    }
    outer += inner*B[i];
  }
  TDY_STOP_FUNCTION_TIMER()
  return outer;
}

PetscErrorCode Inverse(PetscScalar *K,PetscInt nn) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  PetscBLASInt n,lwork=nn*nn;
  ierr = PetscBLASIntCast(nn,&n); CHKERRQ(ierr);
  PetscBLASInt info,*pivots;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&pivots); CHKERRQ(ierr);
  PetscScalar work[n*n];

  // Find LU factors of K
  LAPACKgetrf_(&n,&n,&K[0],&n,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Find inverse
  LAPACKgetri_(&n,&K[0],&n,pivots,work,&lwork,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"illegal argument value");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"singular matrix");

  ierr = PetscFree(pivots); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/*

  <g,w>

 */
PetscErrorCode IntegratePressureBoundary(TDyBDM *bdm, DM dm,
                                         Conditions *conditions, PetscInt f,
                                         PetscInt c, PetscReal *gvdotn) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscQuadrature quadrature,face_quadrature;
  const PetscScalar *fquad_x,*fquad_w;
  PetscReal *single_point,*single_weight,lside[24],x[3],DF[9],DFinv[9],J[1],basis[72],g;
  PetscReal fJ[9],dummy[200],normal[3];
  PetscInt i,j,q,v,d,dim,ncv,nfq,nq1d,face_side,face_dir;
  ncv = bdm->ncv;
  nq1d = 3;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  /* relative to this cell, where is this face on the reference element? */
  if(dim==2) {
    lside[0] = 0; lside[1] = 0;
    lside[2] = 1; lside[3] = 0;
    lside[4] = 0; lside[5] = 1;
    lside[6] = 1; lside[7] = 1;
  } else {
    lside[0]  = 0; lside[1]  = 0; lside[2]  = 0;
    lside[3]  = 1; lside[4]  = 0; lside[5]  = 0;
    lside[6]  = 0; lside[7]  = 1; lside[8]  = 0;
    lside[9]  = 1; lside[10] = 1; lside[11] = 0;
    lside[12] = 0; lside[13] = 0; lside[14] = 1;
    lside[15] = 1; lside[16] = 0; lside[17] = 1;
    lside[18] = 0; lside[19] = 1; lside[20] = 1;
    lside[21] = 1; lside[22] = 1; lside[23] = 1;
  }
  face_side = -1; face_dir = -1;
  for(v=0; v<ncv; v++) {
    for(d=0; d<dim; d++) {
      normal[d] = bdm->N[f*dim+d];
      if(PetscAbsInt(bdm->emap[c*ncv*dim+v*dim+d]) == f) {
        face_side = lside[v*dim+d];
        face_dir  = d;
      }
    }
  }
  if(TDyADotBMinusC(normal,&(bdm->X[f*dim]),&(bdm->X[c*dim]),dim) < 0){
    for(d=0; d<dim; d++) normal[d] *= -1;
  }

  /* face quadrature setup */
  ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&face_quadrature); CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(face_quadrature,NULL,NULL,&nfq,&fquad_x,&fquad_w); CHKERRQ(ierr);
  ierr = DMPlexComputeCellGeometryFEM(dm,f,face_quadrature,dummy,dummy,dummy,fJ); CHKERRQ(ierr);

  /* dummy 1 point quadrature to get the mapping information */
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quadrature); CHKERRQ(ierr);
  ierr = PetscMalloc1(3,&single_point); CHKERRQ(ierr);
  ierr = PetscMalloc1(1,&single_weight); CHKERRQ(ierr);

  /* integrate on the face */
  for(q=0;q<nfq;q++){

    /* extend the dim-1 quadrature point to dim */
    j = 0;
    single_point[0] = 0; single_point[1] = 0; single_point[2] = 0;
    for(i=0; i<dim; i++) {
      if(i == face_dir) {
        single_point[i] = PetscPowInt(-1,face_side+1);
      } else {
        single_point[i] = fquad_x[q*(dim-1)+j];
        j += 1;
      }
    }

    /* get volumetric mapping information and the basis */
    ierr = PetscQuadratureSetData(quadrature,dim,1,1,single_point,single_weight); CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quadrature,x,DF,DFinv,J); CHKERRQ(ierr);
    if(dim==2){
      HdivBasisQuad(single_point,basis,DF,J[0]);
    }else{
      HdivBasisHex(single_point,basis,DF,J[0]);
    }
    /* -<g,v.n>|_q */
    ierr = ConditionsComputeBoundaryPressure(conditions, 1, x, &g);CHKERRQ(ierr);
    for(i=0; i<ncv*dim; i++) gvdotn[i] -= g*TDyADotB(&(basis[i*dim]),normal,dim)*fquad_w[q]*fJ[q];
  }

  ierr = PetscQuadratureDestroy(&quadrature); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&face_quadrature); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/*
    x: dim  *nq = 3*27 = 81
   DF: dim^2*nq = 9*27 = 243
    J:       nq =   27 = 27
*/
PetscErrorCode TDyBDMComputeSystem(TDy tdy,Mat K,Vec F) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt dim,dim2,nlocal,pStart,pEnd,c,cStart,cEnd,q,nq,nfq,nv,vi,vj,di,dj,
    local_row,local_col,isbc,f,nq1d=3;
  PetscScalar x[81],DF[243],DFinv[243],J[27],Klocal[MAX_LOCAL_SIZE],
              Flocal[MAX_LOCAL_SIZE],force,basis_hdiv[72];
  const PetscScalar *quad_x,*fquad_x;
  const PetscScalar *quad_w,*fquad_w;
  PetscQuadrature quadrature;
  PetscQuadrature face_quadrature;
  TDyBDM *bdm = tdy->context;
  DM dm = tdy->dm;
  Conditions *conditions = tdy->conditions;

  /* Get domain constants */
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr); dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nv = bdm->ncv;
  nlocal = dim*nv + 1;

  /* Get quadrature */
  switch(bdm->qtype) {
  case FULL:
    ierr = PetscDTGaussTensorQuadrature(dim  ,1,nq1d,-1,+1,&     quadrature); CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&face_quadrature); CHKERRQ(ierr);
    break;
  case LUMPED:
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quadrature); CHKERRQ(ierr);
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&face_quadrature); CHKERRQ(ierr);
    ierr = SetQuadrature(     quadrature,dim  ); CHKERRQ(ierr);
    ierr = SetQuadrature(face_quadrature,dim-1); CHKERRQ(ierr);
    break;
  }
  ierr = PetscQuadratureGetData(     quadrature,NULL,NULL,&nq ,& quad_x,& quad_w); CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(face_quadrature,NULL,NULL,&nfq,&fquad_x,&fquad_w); CHKERRQ(ierr);

  for(c=cStart; c<cEnd; c++) {

    /* Only assemble the cells that this processor owns */
    ierr = DMPlexGetPointGlobal(dm,c,&pStart,&pEnd); CHKERRQ(ierr);
    if (pStart < 0) continue;
    const PetscInt *LtoG = &(bdm->LtoG[(c-cStart)*nlocal]);
    const PetscInt *orient = &(bdm->orient[(c-cStart)*nlocal]);
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quadrature,x,DF,DFinv,J); CHKERRQ(ierr);
    ierr = PetscMemzero(Klocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(Flocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);

    /* Invert permeability, in place */
    Inverse(&(bdm->K[dim2*(c-cStart)]),dim);

    /* Integrate (Kappa^-1 u_i, v_j) */
    for(q=0; q<nq; q++) {

      /* Evaluate the H-div basis */
      if(dim==2) {
        HdivBasisQuad(&(quad_x[dim*q]),basis_hdiv,&DF[dim2*q],J[q]);
      } else {
        HdivBasisHex(&(quad_x[dim*q]),basis_hdiv,&DF[dim2*q],J[q]);
      }

      /* Double loop over local vertices */
      for(vi=0; vi<nv; vi++) {
        for(vj=0; vj<nv; vj++) {

          /* Double loop over directions */
          for(di=0; di<dim; di++) {
            local_row = vi*dim+di;
            for(dj=0; dj<dim; dj++) {
              local_col = vj*dim+dj;
              /* (K^-1 u, v) */
              Klocal[local_col*nlocal+local_row] += TDyKDotADotB(&(bdm->K[dim2*(c-cStart)]),
                  &(basis_hdiv[dim*local_row]),
                  &(basis_hdiv[dim*local_col]),dim)*quad_w[q]*J[q];
            }
          } /* end directions */

        }
      } /* end vertices */

      /* Integrate forcing if present */
      if (ConditionsHasForcing(conditions)) {
        ierr = ConditionsComputeForcing(conditions, 1, &(x[q*dim]), &force);CHKERRQ(ierr);
        Flocal[nlocal-1] += -force*quad_w[q]*J[q];
      }

    } /* end quadrature */


    /* < g,div(v) > on boundaries, uses mean value of g with 1 point quadrature */
    PetscInt coneSize;
    const PetscInt *cone;
    ierr = DMPlexGetConeSize(dm,c,&coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&cone); CHKERRQ(ierr);
    for(f=0;f<coneSize;f++){
      ierr = DMGetLabelValue(dm,"marker",cone[f],&isbc); CHKERRQ(ierr);
      if(isbc == 1 && ConditionsHasBoundaryPressure(conditions)) {
        ierr = IntegratePressureBoundary(bdm,dm,conditions,cone[f],c,Flocal);
      }
    } /* end faces */

    /* apply orientation flips */
    for(vi=0; vi<nlocal-1; vi++) {
      Flocal[vi] *= (PetscScalar)orient[vi];
      for(vj=0; vj<nlocal-1; vj++) {
        Klocal[vj*nlocal+vi] *= (PetscScalar)(orient[vi]*orient[vj]);
      }
      Klocal[(nlocal-1)*nlocal+vi] = -(PetscScalar)orient[vi]; /* < div(u), w > */
      Klocal[vi*nlocal+nlocal-1  ] = -(PetscScalar)orient[vi]; /* < p, div(v) > */
    }

    //PrintMatrix(Klocal,nlocal,nlocal,PETSC_TRUE);
    //PrintMatrix(Flocal,1,nlocal,PETSC_TRUE);

    /* assembly */
    ierr = MatSetValues(K,nlocal,LtoG,nlocal,LtoG,Klocal,ADD_VALUES); CHKERRQ(ierr);
    ierr = VecSetValues(F,nlocal,LtoG,Flocal,ADD_VALUES); CHKERRQ(ierr);

  } /* end cell */

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&face_quadrature); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&     quadrature); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/*
  Velocity norm given in (3.40) of Wheeler2012.

  ||u-uh||^2 = sum_E sum_e |E|/|e| ||(u-uh).n||^2

  where ||(u-uh).n|| is evaluated with nq1d=2 quadrature. This
  integrates the normal velocity error over the face, normalized by
  the area of the face and then weighted by cell volume.

 */
PetscErrorCode TDyComputeErrorNorms_BDM(void *context, DM dm, Conditions *conditions,
                                        Vec U, PetscReal *p_norm, PetscReal *v_norm) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd,dim,gref,fStart,fEnd,junk,d,s,f,offset;
  PetscReal p,*u,norm,norm_sum;
  PetscSection sec;

  TDyBDM *bdm = context;

  // Pressure norm
  if (p_norm) {
    if(!ConditionsHasBoundaryPressure(conditions)) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
          "Must set the pressure function with TDySetDirichletValueFunction");
    }
    ierr = VecGetArray(U,&u); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
    ierr = DMGetLocalSection(dm,&sec); CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
    norm = 0; norm_sum = 0;
    for(c=cStart; c<cEnd; c++) {
      ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk); CHKERRQ(ierr);
      if(gref<0) continue;
      ierr = PetscSectionGetOffset(sec,c,&offset); CHKERRQ(ierr);
      ierr = ConditionsComputeBoundaryPressure(conditions, 1, &(bdm->X[c*dim]),
                                               &p);CHKERRQ(ierr);
      norm += bdm->V[c]*PetscSqr(u[offset]-p);
    }
    ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
        PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);
    *p_norm = PetscSqrtReal(norm_sum);
    ierr = VecRestoreArray(U,&u); CHKERRQ(ierr);
    TDY_STOP_FUNCTION_TIMER()
  }

  // Velocity norm
  if (v_norm) {
    if(!ConditionsHasBoundaryVelocity(conditions)) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
          "Must set the velocity function with TDySetDirichletFluxFunction");
    }
    ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

    PetscInt i,j,ncv,q,nlocal,nq,dd,nq1d=3;
    const PetscScalar *quad_x,*quad_w;
    PetscReal xq[3],x[100],DF[100],DFinv[100],J[100],N[72],vel[3],ve,va,flux0,flux,
              norm,norm_sum;
    PetscQuadrature quad;
    PetscScalar *u;
    ierr = VecGetArray(U,&u); CHKERRQ(ierr);
    ncv  = bdm->ncv;
    ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&quad); CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,&quad_x,&quad_w); CHKERRQ(ierr);
    nlocal = dim*ncv + 1;

    PetscQuadrature cquad;
    PetscReal *points,*weights;
    ierr = PetscMalloc1(dim,&points); CHKERRQ(ierr);
    ierr = PetscMalloc1(1,&weights); CHKERRQ(ierr);

    PetscReal cx[3],cDF[9],cDFinv[9],cJ[1];
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&cquad); CHKERRQ(ierr);

    /* loop cells */
    norm = 0; norm_sum = 0;
    for(c=cStart; c<cEnd; c++) {

      const PetscInt *LtoG = &(bdm->LtoG[(c-cStart)*nlocal]);
      const PetscInt *orient = &(bdm->orient[(c-cStart)*nlocal]);

      ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk); CHKERRQ(ierr);
      if (gref < 0) continue;

      /* loop faces */
      for(d=0; d<dim; d++) {
        for(s=0; s<2; s++) {
          f = bdm->faces[(c-cStart)*dim*2+d*2+s];

          ierr = DMPlexComputeCellGeometryFEM(dm,f,quad,x,DF,DFinv,J); CHKERRQ(ierr);

          /* loop quadrature */
          flux0 = flux = 0;
          for(q=0; q<nq; q++) {

            /* extend the dim-1 quadrature point to dim */
            j = 0;
            xq[0] = 0; xq[1] = 0; xq[2] = 0;
            for(i=0; i<dim; i++) {
              if(i == d) {
                xq[i] = PetscPowInt(-1,s+1);
              } else {
                xq[i] = quad_x[q*(dim-1)+j];
                j += 1;
              }
            }

            /* interpolate normal component at this point/face */
            if(dim==2) {
              points[0] = xq[0]; points[1] = xq[1];
              ierr = PetscQuadratureSetData(cquad,dim,1,1,points,weights); CHKERRQ(ierr);
              ierr = DMPlexComputeCellGeometryFEM(dm,c,cquad,cx,cDF,cDFinv,cJ); CHKERRQ(ierr);
              HdivBasisQuad(xq,N,cDF,cJ[0]);
            } else {
              points[0] = xq[0]; points[1] = xq[1]; points[2] = xq[2];
              ierr = PetscQuadratureSetData(cquad,dim,1,1,points,weights); CHKERRQ(ierr);
              ierr = DMPlexComputeCellGeometryFEM(dm,c,cquad,cx,cDF,cDFinv,cJ); CHKERRQ(ierr);
              HdivBasisHex(xq,N,cDF,cJ[0]);
            }
            vel[0] = 0; vel[1] = 0; vel[2] = 0;
            for(i=0;i<nlocal-1;i++) {
              for(dd=0;dd<dim;dd++) {
                vel[dd] += ((PetscReal)orient[i])*N[dim*i+dd]*u[LtoG[i]];
              }
            }
            va = TDyADotB(vel,&(bdm->N[dim*f]),dim);

            /* exact value normal to this point/face */
            ierr = ConditionsComputeBoundaryVelocity(conditions, 1, &(x[q*dim]),
                vel);CHKERRQ(ierr);
            ve = TDyADotB(vel,&(bdm->N[dim*f]),dim);

            /* quadrature */
            flux  += va*quad_w[q]*J[q];
            flux0 += ve*quad_w[q]*J[q];
          }
          norm += PetscSqr((flux-flux0)/bdm->V[f])*bdm->V[c];
        }
      }

    }
    ierr = PetscQuadratureDestroy(&cquad); CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&quad); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
        PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
    norm_sum = PetscSqrtReal(norm_sum);
    ierr = VecRestoreArray(U,&u); CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&quad); CHKERRQ(ierr);
    *v_norm = norm_sum;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
