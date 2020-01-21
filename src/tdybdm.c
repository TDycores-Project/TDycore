#include <private/tdycoreimpl.h>
#include <petscblaslapack.h>

/* (dim*vertices_per_cell+1)^2 */
#define MAX_LOCAL_SIZE 625

/*
  u <-- 1/J DF u
*/
void TDyPiola(PetscReal *u,PetscReal *DF,PetscReal J,PetscInt dim){
  PetscInt i,j;
  PetscReal v[3];
  for(i=0;i<dim;i++){
    v[i] = 0;
    for(j=0;j<dim;j++){
      v[i] += DF[i*dim+j]*u[j];
    }
  }
  for(i=0;i<dim;i++) u[i] = v[i]/J;
}

/*
  BDM1 basis functions on [-1,1] with degrees of freedom chosen to
  match Wheeler2009. Indices map <-- local_vertex*dim + dir.

  2---3
  |   |
  0---1

 */
void HdivBasisQuad(const PetscReal *x,PetscReal *B,PetscReal *DF,PetscReal J) {
  PetscInt i;
  B[ 0] = (-x[0]*x[1] + x[0] + x[1] - 1)*0.25;
  B[ 1] = (x[1]*x[1] - 1)*0.125;
  B[ 2] = (x[0]*x[0] - 1)*0.125;
  B[ 3] = (-x[0]*x[1] + x[0] + x[1] - 1)*0.25;
  B[ 4] = (-x[0]*x[1] + x[0] - x[1] + 1)*0.25;
  B[ 5] = (x[1]*x[1] - 1)*0.125;
  B[ 6] = (-x[0]*x[0] + 1)*0.125;
  B[ 7] = (x[0]*x[1] - x[0] + x[1] - 1)*0.25;
  B[ 8] = (x[0]*x[1] + x[0] - x[1] - 1)*0.25;
  B[ 9] = (-x[1]*x[1] + 1)*0.125;
  B[10] = (x[0]*x[0] - 1)*0.125;
  B[11] = (-x[0]*x[1] - x[0] + x[1] + 1)*0.25;
  B[12] = (x[0]*x[1] + x[0] + x[1] + 1)*0.25;
  B[13] = (-x[1]*x[1] + 1)*0.125;
  B[14] = (-x[0]*x[0] + 1)*0.125;
  B[15] = (x[0]*x[1] + x[0] + x[1] + 1)*0.25;
  if(DF){
    for(i=0;i<8;i++) TDyPiola(&(B[2*i]),DF,J,2);
  }
}

void HdivBasisHex(const PetscReal *x,PetscReal *B) {
  B[0]  = +0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125;
  B[1]  = +0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125;
  B[2]  = +0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125;
  B[3]  = +0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] - 0.125*x[1] - 0.125*x[2] + 0.125;
  B[4]  = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125;
  B[5]  = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125;
  B[6]  = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] - 0.125;
  B[7]  = +0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] + 0.125;
  B[8]  = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] - 0.125;
  B[9]  = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] + 0.125;
  B[10] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] + 0.125;
  B[11] = +0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0] +
          0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] - 0.125;
  B[12] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] - 0.125;
  B[13] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] - 0.125;
  B[14] = +0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0] -
          0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] + 0.125;
  B[15] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] + 0.125;
  B[16] = +0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] - 0.125;
  B[17] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] + 0.125;
  B[18] = +0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] -
          0.125*x[1]*x[2] - 0.125*x[1] - 0.125*x[2] - 0.125;
  B[19] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125;
  B[20] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125;
  B[21] = +0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125;
  B[22] = +0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125;
  B[23] = +0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
          0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125;
}

PetscErrorCode TDyBDMInitialize(TDy tdy) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt pStart,pEnd,c,cStart,cEnd,f,f_abs,fStart,fEnd,nfv,ncv,v,vStart,vEnd,
           mStart,mEnd,i,nlocal,closureSize,*closure;
  PetscSection sec;
  PetscInt d,dim,dofs_per_face = 1;
  PetscBool found;
  DM dm = tdy->dm;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  /* Get plex limits */
  ierr = DMPlexGetChart        (dm,  &pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

  /* Create H-div section */
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
  for(d=0; d<(dim-1); d++) dofs_per_face *= 2;
  for(f=fStart; f<fEnd; f++) {
    ierr = PetscSectionSetFieldDof(sec,f,1,dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,f,  dofs_per_face); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm,&sec); CHKERRQ(ierr);

  /* Build vmap and emap */
  ierr = TDyCreateCellVertexMap(tdy,&(tdy->vmap)); CHKERRQ(ierr);
  ierr = TDyCreateCellVertexDirFaceMap(tdy,&(tdy->emap)); CHKERRQ(ierr);

  /* Build map(face,local_vertex) --> vertex */
  nfv = TDyGetNumberOfFaceVertices(dm);
  ierr = PetscMalloc(nfv*(fEnd-fStart)*sizeof(PetscInt),
                     &(tdy->fmap)); CHKERRQ(ierr);
  for(f=fStart; f<fEnd; f++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,
                                      &closureSize,&closure); CHKERRQ(ierr);
    i = 0;
    for(c=0; c<closureSize*2; c+=2) {
      if ((closure[c] < vStart) || (closure[c] >= vEnd)) continue;
      tdy->fmap[nfv*(f-fStart)+i] = closure[c];
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
  tdy->ncv = TDyGetNumberOfCellVertices(dm);
  ncv = tdy->ncv;
  nlocal = dim*ncv + 1;
  ierr = PetscMalloc((cEnd-cStart)*nlocal*sizeof(PetscInt),
                     &(tdy->LtoG)); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*nlocal*sizeof(PetscInt),
                     &(tdy->orient)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&mStart,&mEnd); CHKERRQ(ierr);
    if(mStart<0) mStart = -(mStart+1);
    tdy->LtoG[(c-cStart+1)*nlocal-1] = mStart;
    for(v=0; v<ncv; v++) {
      for(d=0; d<dim; d++) {
        /* which face is this local dof on? */
        f = tdy->emap[(c-cStart)*ncv*dim+v*dim+d];
        f_abs = PetscAbsInt(f);
        ierr = DMPlexGetPointGlobal(dm,f_abs,&mStart,&mEnd); CHKERRQ(ierr);
	if(mStart<0) mStart = -(mStart+1);    
        found = PETSC_FALSE;
        for(i=0; i<nfv; i++) {
          if(tdy->vmap[ncv*(c-cStart)+v] == tdy->fmap[nfv*(f_abs-fStart)+i]) {
            tdy->LtoG  [(c-cStart)*nlocal + v*dim + d] = mStart + i;
            tdy->orient[(c-cStart)*nlocal + v*dim + d] = PetscSign(f);
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
                     &(tdy->faces)); CHKERRQ(ierr);
  #if defined(PETSC_USE_DEBUG)
  for(c=0; c<((cEnd-cStart)*(2*dim)); c++) { tdy->faces[c] = -1; }
  #endif
  PetscInt s;
  for(c=cStart; c<cEnd; c++) {
    for(d=0; d<dim; d++) {
      for(s=0; s<2; s++) {
        v = s*PetscPowInt(2,d);
        tdy->faces[(c-cStart)*dim*2+d*2+s] = PetscAbsInt(tdy->emap[(c-cStart)*ncv*dim+v*dim+d]);
      }
    }
  }
  #if defined(PETSC_USE_DEBUG)
  for(c=0; c<((cEnd-cStart)*(2*dim)); c++) {
    if(tdy->faces[c] < 0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,dir,side) -> face");
    }
  }
  #endif

  PetscFunctionReturn(0);
}

PetscReal TDyKDotADotB(PetscReal *K,PetscReal *A,PetscReal *B,PetscInt dim) {
  PetscInt i,j;
  PetscReal inner,outer=0;
  for(i=0; i<dim; i++) {
    inner = 0;
    for(j=0; j<dim; j++) {
      inner += K[j*dim+i]*A[j];
    }
    outer += inner*B[i];
  }
  return outer;
}

PetscErrorCode Inverse(PetscScalar *K,PetscInt nn) {
  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

/* x:  dim  *nq = 3*27 = 81
   DF: dim^2*nq = 9*27 = 243
   J:        nq =   27 = 27
*/
PetscErrorCode TDyBDMComputeSystem(TDy tdy,Mat K,Vec F) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt dim,dim2,nlocal,pStart,pEnd,c,cStart,cEnd,q,nq,nfq,nv,vi,vj,di,dj,
    local_row,local_col,isbc,f,nq1d=3;
  PetscScalar x[81],DF[243],DFinv[243],J[27],Klocal[MAX_LOCAL_SIZE],
              Flocal[MAX_LOCAL_SIZE],force,basis_hdiv[72],pressure;
  const PetscScalar *quad_x,*fquad_x;
  const PetscScalar *quad_w,*fquad_w;
  PetscQuadrature quadrature;
  PetscQuadrature face_quadrature;
  DM dm = tdy->dm;

  /* Get domain constants */
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr); dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nv = tdy->ncv;
  nlocal = dim*nv + 1;
  
  /* Get quadrature */
  switch(tdy->qtype) {
  case FULL:
    ierr = PetscDTGaussTensorQuadrature(dim  ,1,nq1d,-1,+1,&     quadrature); CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&face_quadrature); CHKERRQ(ierr);
    break;
  case LUMPED:
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quadrature); CHKERRQ(ierr);
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&face_quadrature); CHKERRQ(ierr);
    ierr = TDyQuadrature(     quadrature,dim  ); CHKERRQ(ierr);
    ierr = TDyQuadrature(face_quadrature,dim-1); CHKERRQ(ierr);
    break;
  }
  ierr = PetscQuadratureGetData(     quadrature,NULL,NULL,&nq ,& quad_x,& quad_w); CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(face_quadrature,NULL,NULL,&nfq,&fquad_x,&fquad_w); CHKERRQ(ierr);
  
  for(c=cStart; c<cEnd; c++) {

    /* Only assemble the cells that this processor owns */
    ierr = DMPlexGetPointGlobal(dm,c,&pStart,&pEnd); CHKERRQ(ierr);
    if (pStart < 0) continue;
    const PetscInt *LtoG = &(tdy->LtoG[(c-cStart)*nlocal]);
    const PetscInt *orient = &(tdy->orient[(c-cStart)*nlocal]);
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quadrature,x,DF,DFinv,J); CHKERRQ(ierr);
    ierr = PetscMemzero(Klocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(Flocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);

    /* Invert permeability, in place */
    Inverse(&(tdy->K[dim2*(c-cStart)]),dim);
    
    /* Integrate (Kappa^-1 u_i, v_j) */
    for(q=0; q<nq; q++) {

      /* Evaluate the H-div basis */
      if(dim==2) {
        HdivBasisQuad(&(quad_x[dim*q]),basis_hdiv,&DF[dim2*q],J[q]);
      } else {
        HdivBasisHex(&(quad_x[dim*q]),basis_hdiv);
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
              Klocal[local_col*nlocal+local_row] += TDyKDotADotB(&(tdy->K[dim2*(c-cStart)]),
								 &(basis_hdiv[dim*local_row]),
								 &(basis_hdiv[dim*local_col]),dim)*quad_w[q]*J[q];
            }
          } /* end directions */

        }
      } /* end vertices */

      /* Integrate forcing if present */
      if (tdy->ops->computeforcing) {
        ierr = (*tdy->ops->computeforcing)(tdy,&(x[q*dim]),&force,tdy->forcingctx);CHKERRQ(ierr);
        Flocal[nlocal-1] += -force*quad_w[q]*J[q];
      }

    } /* end quadrature */

    
    /* < g,div(v) > on boundaries, uses mean value of g with 1 point quadrature */
    PetscInt coneSize,v,d;
    const PetscInt *cone;
    ierr = DMPlexGetConeSize(dm,c,&coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&cone); CHKERRQ(ierr);
    for(f=0;f<coneSize;f++){
      ierr = DMGetLabelValue(dm,"marker",cone[f],&isbc); CHKERRQ(ierr);
      if(isbc == 1 && tdy->ops->computedirichletvalue){
	ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[cone[f]*dim]), &pressure, tdy->dirichletvaluectx);CHKERRQ(ierr);
	for(v=0; v<nv; v++) {
	  for(d=0; d<dim; d++) {
	    local_col = dim*v + d;
	    if(PetscAbsInt(tdy->emap[c*nv*dim+v*dim+d]) == cone[f]) {
	      Flocal[local_col] = -pressure;
	    }
	  }
	}
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
  PetscFunctionReturn(0);
}

PetscReal TDyBDMPressureNorm(TDy tdy,Vec U) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscSection sec;
  PetscInt c,cStart,cEnd,offset,dim,gref,junk;
  PetscReal p,*u,norm,norm_sum;
  DM dm = tdy->dm;
  if(!(tdy->ops->computedirichletvalue)) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the pressure function with TDySetDirichletValueFunction");
  }
  norm = 0;
  ierr = VecGetArray(U,&u); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMGetSection(dm,&sec); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk); CHKERRQ(ierr);
    if(gref<0) continue;
    ierr = PetscSectionGetOffset(sec,c,&offset); CHKERRQ(ierr);
    ierr = (*tdy->ops->computedirichletvalue)(tdy,&(tdy->X[c*dim]),&p,tdy->dirichletvaluectx);CHKERRQ(ierr);
    norm += tdy->V[c]*PetscSqr(u[offset]-p);
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  ierr = VecRestoreArray(U,&u); CHKERRQ(ierr);
  PetscFunctionReturn(norm_sum);
}

/*
  Velocity norm given in (3.40) of Wheeler2012.

  ||u-uh||^2 = sum_E sum_e |E|/|e| ||(u-uh).n||^2

  where ||(u-uh).n|| is evaluated with nq1d=2 quadrature. This
  integrates the normal velocity error over the face, normalized by
  the area of the face and then weighted by cell volume.

 */
PetscReal TDyBDMVelocityNorm(TDy tdy,Vec U) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd,dim,gref,fStart,fEnd,junk,d,s,f;
  DM dm = tdy->dm;
  if(!(tdy->ops->computedirichletflux)) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the velocity function with TDySetDirichletFluxFunction");
  }
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

  PetscInt i,j,ncv,q,nlocal,nq,dd,nq1d=2;
  const PetscScalar *quad_x,*quad_w;
  PetscReal xq[3],x[100],DF[100],DFinv[100],J[100],N[72],vel[3],ve,va,flux0,flux,
            norm,norm_sum;
  PetscQuadrature quad;
  PetscScalar *u;
  ierr = VecGetArray(U,&u); CHKERRQ(ierr);
  ncv  = tdy->ncv;
  ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&quad); CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,&quad_x,&quad_w); CHKERRQ(ierr);
  nlocal = dim*ncv + 1;

  PetscQuadrature cquad;
  PetscReal *points,*weights;
  ierr = PetscMalloc1(dim,&points); CHKERRQ(ierr);
  ierr = PetscMalloc1(1,&weights); CHKERRQ(ierr);

  PetscReal cx[2],cDF[4],cDFinv[4],cJ[1];  
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&cquad); CHKERRQ(ierr);
  
  /* loop cells */
  norm = 0; norm_sum = 0;
  for(c=cStart; c<cEnd; c++) {

    const PetscInt *LtoG = &(tdy->LtoG[(c-cStart)*nlocal]);
    const PetscInt *orient = &(tdy->orient[(c-cStart)*nlocal]);
    
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk); CHKERRQ(ierr);
    if (gref < 0) continue;

    /* loop faces */
    for(d=0; d<dim; d++) {
      for(s=0; s<2; s++) {
        f = tdy->faces[(c-cStart)*dim*2+d*2+s];
	
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
            HdivBasisQuad(xq,N,NULL,cJ[0]);
          } else {
            HdivBasisHex(xq,N);
          }
	  
          vel[0] = 0; vel[1] = 0; vel[2] = 0;
	  for(i=0;i<nlocal-1;i++) {
	    for(dd=0;dd<dim;dd++) {
	      //printf("%d %d %f %f\n",i,dd,N[dim*i+dd],u[LtoG[i]]);
	      vel[dd] += ((PetscReal)orient[i])*N[dim*i+dd]*u[LtoG[i]];
	    }
	  }
	  //printf("\n%f %f  %f %f ",x[q*dim],x[q*dim+1],vel[0],vel[1]);
          va = TDyADotB(vel,&(tdy->N[dim*f]),dim);
	  
          /* exact value normal to this point/face */
          ierr = (*tdy->ops->computedirichletflux)(tdy,&(x[q*dim]),vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
	  //printf(" %f %f\n",vel[0],vel[1]);
          ve = TDyADotB(vel,&(tdy->N[dim*f]),dim);
	  
          /* quadrature */
          flux  += va*quad_w[q]*J[q];
          flux0 += ve*quad_w[q]*J[q];
        }
        norm += PetscSqr((flux-flux0)/tdy->V[f])*tdy->V[c];
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
  PetscFunctionReturn(norm_sum);
}
