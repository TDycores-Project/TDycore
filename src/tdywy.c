#include "tdycore.h"
#include <petscblaslapack.h>

#define MAX_LOCAL_SIZE 50

PetscErrorCode Pullback(PetscScalar *K,PetscScalar *DFinv,PetscScalar *Kappa,PetscScalar J,PetscInt nn)
{
  /*
    K(dxd)     flattened array in row major (but doesn't matter as it is symmetric)
    DFinv(dxd) flattened array in row major format (how PETSc generates it)
    J          det(DF)

    returns Kappa^-1 = ( J DF^-1 K (DF^-1)^T )^-1
   */

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscScalar  zero=0,one=1;
  PetscBLASInt n,lwork=nn*nn;
  ierr = PetscBLASIntCast(nn,&n);CHKERRQ(ierr);
  PetscBLASInt info,*pivots;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  PetscScalar KDFinvT[n*n],work[n*n];
  /* LAPACK wants things in column major, so we need to transpose both
     K and DFinv. However, we are computing K (DF^-1)^T and thus we
     leave DFinv as is. The resulting KDFinvT is in column major
     format. */
  BLASgemm_("T","N",&n,&n,&n,&one,K    ,&n,DFinv  ,&n,&zero,KDFinvT  ,&n);
  /* Here we are computing J * DFinv * KDFinvT. Since DFinv is row
     major and LAPACK wants things column major, we need to transpose
     it. */
  BLASgemm_("T","N",&n,&n,&n,&J  ,DFinv,&n,KDFinvT,&n,&zero,&Kappa[0],&n);

  // Find LU factors of Kappa
  LAPACKgetrf_(&n,&n,&Kappa[0],&n,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Find inverse
  LAPACKgetri_(&n,&Kappa[0],&n,pivots,work,&lwork,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"illegal argument value");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"singular matrix");

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormStencil(PetscScalar *A,PetscScalar *B,PetscScalar *C,
			   PetscScalar *G,PetscScalar *D,
			   PetscInt qq,PetscInt rr)
{
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // return C(rxr) = B.T A^-1 B in col major
  //        D(r  ) = B.T A^-1 G in col major

  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt q,r,o = 1,info,*pivots;
  PetscScalar zero = 0,one = 1;
  ierr = PetscBLASIntCast(qq,&q);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rr,&r);CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  // Copy B because we will need it again
  PetscScalar AinvB[qq*rr];
  ierr = PetscMemcpy(AinvB,B,sizeof(PetscScalar)*(qq*rr));CHKERRQ(ierr); // AinvB in col major

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Solve AinvB = (A^-1 * B) by back-substitution
  LAPACKgetrs_("N",&q,&r,A,&q,pivots,AinvB,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Solve G = (A^-1 * G) by back-substitution
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,G,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (B.T * AinvB) and (B.T * G)
  BLASgemm_("T","N",&r,&r,&q,&one,B,&q,AinvB,&q,&zero,&C[0],&r); // B.T * AinvB
  BLASgemm_("T","N",&r,&o,&q,&one,B,&q,G    ,&q,&zero,&D[0],&r); // B.T * G

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RecoverVelocity(PetscScalar *A,PetscScalar *F,PetscInt qq)
{
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // return U = A^-1 (G - B P)
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt q,o = 1,info,*pivots;
  ierr = PetscBLASIntCast(qq,&q);CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Solve by back-substitution
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,F,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyWYLocalElementCompute(DM dm,TDy tdy)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd;
  PetscInt dim,dim2,i,q,nq = 4;
  PetscReal wgt   = 0.25; // 1/s from the paper
  PetscReal Ehat  = 4;    // area of ref element ( [-1,1] x [-1,1] )
  PetscReal ehat  = 2;    // length of ref element edge
  PetscScalar x[24],DF[72],DFinv[72],J[8],Kinv[9],f; /* allocated at maximum possible size */

  /* using quadrature points as a local numbering, what are the
     outward normals for each dof at each local vertex? Need to
     generalize this. */
  PetscScalar n0[8] = {-1, 0,+1, 0,-1, 0,+1, 0};
  PetscScalar n1[8] = { 0,-1, 0,-1, 0,+1, 0,+1};
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    tdy->Flocal[c] = 0;	  
    ierr = DMPlexComputeCellGeometryFEM(dm,c,tdy->quad,x,DF,DFinv,J);CHKERRQ(ierr); // DF and DFinv are row major
    for(q=0;q<nq;q++){

      // compute Kappa^-1 which will be in column major format (shouldn't matter as it is symmetric)
      ierr = Pullback(&(tdy->K[dim2*(c-cStart)]),&DFinv[dim2*q],Kinv,J[q],dim);CHKERRQ(ierr);

      // at each vertex, we have a dim by dim system which we will store
      i = c*(dim2*nq)+q*(dim2);
      tdy->Alocal[i  ]  = (Kinv[0]*n0[q*dim] + Kinv[2]*n0[q*dim+1])*n0[q*dim  ]; // (K n0, n0)
      tdy->Alocal[i  ] += (Kinv[1]*n0[q*dim] + Kinv[3]*n0[q*dim+1])*n0[q*dim+1];
      tdy->Alocal[i  ] *= Ehat/ehat/ehat*wgt;
      tdy->Alocal[i+1]  = (Kinv[0]*n1[q*dim] + Kinv[2]*n1[q*dim+1])*n0[q*dim  ]; // (K n1, n0)
      tdy->Alocal[i+1] += (Kinv[1]*n1[q*dim] + Kinv[3]*n1[q*dim+1])*n0[q*dim+1];
      tdy->Alocal[i+1] *= Ehat/ehat/ehat*wgt;
      tdy->Alocal[i+2]  = (Kinv[0]*n0[q*dim] + Kinv[2]*n0[q*dim+1])*n1[q*dim  ]; // (K n0, n1)
      tdy->Alocal[i+2] += (Kinv[1]*n0[q*dim] + Kinv[3]*n0[q*dim+1])*n1[q*dim+1];
      tdy->Alocal[i+2] *= Ehat/ehat/ehat*wgt;
      tdy->Alocal[i+3]  = (Kinv[0]*n1[q*dim] + Kinv[2]*n1[q*dim+1])*n1[q*dim  ]; // (K n1, n1)
      tdy->Alocal[i+3] += (Kinv[1]*n1[q*dim] + Kinv[3]*n1[q*dim+1])*n1[q*dim+1];
      tdy->Alocal[i+3] *= Ehat/ehat/ehat*wgt;

      // integrate the forcing function using the same quadrature
      if (tdy->forcing) {
	(*tdy->forcing)(&(x[q*dim]),&f);
	tdy->Flocal[c] += f*J[q];
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDyWyQuadrature(PetscQuadrature q,PetscInt dim,PetscInt nq)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *x,*w;
  ierr = PetscMalloc1(nq*dim,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(nq    ,&w);CHKERRQ(ierr);
  switch(nq*dim){ 
  case 6: /* tri */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Triangles not yet supported in WHEELER_YOTOV");
  case 8: /* quad */
    x[0] = -1.0; x[1] = -1.0; w[0] = 0.25;
    x[2] =  1.0; x[3] = -1.0; w[1] = 0.25;
    x[4] = -1.0; x[5] =  1.0; w[2] = 0.25;
    x[6] =  1.0; x[7] =  1.0; w[3] = 0.25;
    break;
  case 24: /* hex */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Hexes not yet supported in WHEELER_YOTOV");    
  }
  ierr = PetscQuadratureSetData(q,dim,1,nq,x,w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscInt GetNumberOfVertices(DM dm){
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

PetscErrorCode TDyWYInitialize(DM dm,TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt i,j,d,dim,q,nq,c,cStart,cEnd,f,fStart,fEnd,v,vStart,vEnd,p,pStart,pEnd;
  PetscInt  closureSize,  *closure;
  PetscInt fclosureSize, *fclosure;
  PetscScalar x[24],DF[72],DFinv[72],J[8]; /* allocated at maximum possible size */
  PetscScalar norm;
  PetscInt local_dirs[8] = {2,1, 3,0, 0,3, 1,2}; /* need to generalize */
  PetscSection sec;
  
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
 
  /* Check that the number of vertices per cell are constant. Soft
     limitation, method is flexible but my data structures are not. */
  nq = GetNumberOfVertices(dm);

  /* Create a PETSc quadrature, we don't really use this, it is just
     to evaluate the Jacobian via the PETSc interface. */
  ierr = PetscQuadratureCreate(comm,&(tdy->quad));CHKERRQ(ierr);
  ierr = TDyWyQuadrature(tdy->quad,dim,nq);CHKERRQ(ierr);
  
  /* Build vmap and emap */
  ierr = PetscMalloc(    nq*(cEnd-cStart)*sizeof(PetscInt),&(tdy->vmap));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*nq*(cEnd-cStart)*sizeof(PetscInt),&(tdy->emap));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexComputeCellGeometryFEM(dm,c,tdy->quad,x,DF,DFinv,J);CHKERRQ(ierr);
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);

    /* Check if the image of the quadrature point is coincident with
       the vertex, if so we create a map, local_element_vertex -->
       global_vertex_point */
    for(q=0;q<nq;q++){
      for (i=0;i<closureSize*2;i+=2){
	if ((closure[i] >= vStart) && (closure[i] < vEnd)) {
	  norm = 0;
	  for(d=0;d<dim;d++) norm += PetscAbsReal(x[q*dim+d]-tdy->X[closure[i]*dim+d]);
	  if (norm > 1e-12) continue;
	  tdy->vmap[c*nq+q] = closure[i];
	  break;
	}
      }
    }
    /* We need a map for (local_element_vertex,direction) -->
       global_face_point. To do this, I loop over the vertices of this
       cell and find connected faces. Then I use the local ordering of
       the vertices to determine where the normal of this face
       points. Finally I check if the normal points into the cell. If
       so, then the index is given a negative as a flag later in the
       assembly process. Since the Hasse diagram always begins with
       cells, there isn't a conflict with 0 being a possible point. */
    for(q=0;q<nq;q++){
      for (i=0;i<closureSize*2;i+=2){
    	if ((closure[i] >= fStart) && (closure[i] < fEnd)) {
    	  fclosure = NULL;
    	  ierr = DMPlexGetTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,&fclosure);CHKERRQ(ierr);
    	  for(f=0;f<fclosureSize*2;f+=2){
    	    if (fclosure[f] == tdy->vmap[c*nq+q]){
    	      for(v=0;v<fclosureSize*2;v+=2){
    		for(d=0;d<dim;d++){
    		  if (fclosure[v] == tdy->vmap[c*nq+local_dirs[q*dim+d]]) {
    		    tdy->emap[c*nq*dim+q*dim+d] = closure[i];
		    norm = 0;
		    for(j=0;j<dim;j++) norm += tdy->N[closure[i]*dim+j] * (tdy->X[closure[i]*dim+j] - tdy->X[c*dim+j]);
		    if (norm < 0) {
		      tdy->emap[c*nq*dim+q*dim+d] *= -1;
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

  /* Allocate space for Alocal and Flocal */
  ierr = PetscMalloc(dim*dim*nq*(cEnd-cStart)*sizeof(PetscReal),&(tdy->Alocal));CHKERRQ(ierr);
  ierr = PetscMalloc(           (cEnd-cStart)*sizeof(PetscReal),&(tdy->Flocal));CHKERRQ(ierr);

  /* This needs to be generalized */
  if(dim>2){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Need to rework velocity data structure for 3D.");
  }
  ierr = PetscMalloc(dim*(fEnd-fStart)*sizeof(PetscReal),&(tdy->vel));CHKERRQ(ierr); 
    
  /* Setup the section, 1 dof per cell */
  ierr = PetscSectionCreate(comm,&sec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"LiquidPressure");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view");CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TDyWYComputeSystem(DM dm,TDy tdy,Mat K,Vec F){
  PetscErrorCode ierr;
  PetscInt v,vStart,vEnd;
  PetscInt   fStart,fEnd;
  PetscInt c,cStart,cEnd;
  PetscInt   gStart,lStart,junk;
  PetscInt element_vertex,nA,nB,q,dim,dim2,nq;
  PetscInt element_row,local_row,global_row;
  PetscInt element_col,local_col,global_col;
  PetscScalar A[MAX_LOCAL_SIZE],B[MAX_LOCAL_SIZE],C[MAX_LOCAL_SIZE],G[MAX_LOCAL_SIZE],D[MAX_LOCAL_SIZE],sign_row,sign_col;
  PetscInt Amap[MAX_LOCAL_SIZE],Bmap[MAX_LOCAL_SIZE];
  PetscScalar pdirichlet;
  PetscFunctionBegin;

  ierr = TDyWYLocalElementCompute(dm,tdy);CHKERRQ(ierr);
  nq   = GetNumberOfVertices(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  dim2 = dim*dim;
    
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){ // loop vertices

    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    // determine the size and mapping of the vertex-local systems
    nA = 0; nB = 0;
    for (c = 0; c < closureSize*2; c += 2) {
      if ((closure[c] >= fStart) && (closure[c] < fEnd)) { Amap[nA] = closure[c]; nA += 1; }
      if ((closure[c] >= cStart) && (closure[c] < cEnd)) { Bmap[nB] = closure[c]; nB += 1; }
    }
    ierr = PetscMemzero(A,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(B,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(C,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(G,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(D,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);

    for (c=0;c<closureSize*2;c+=2){ // loop connected cells
      if ((closure[c] < cStart) || (closure[c] >= cEnd)) continue;

	// for the cell, which local vertex is this vertex?
	element_vertex = -1;
	for(q=0;q<nq;q++){
	  if(v == tdy->vmap[closure[c]*nq+q]){
	    element_vertex = q;
	    break;
	  }
	}
	if(element_vertex < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	for(element_row=0;element_row<dim;element_row++){ // which test function, local to the element/vertex
	  global_row = tdy->emap[closure[c]*nq*dim+element_vertex*dim+element_row]; // DMPlex point index of the face
	  sign_row   = PetscSign(global_row);
	  global_row = PetscAbsInt(global_row);
	  local_row  = -1;
	  for(q=0;q<nA;q++){
	    if(Amap[q] == global_row) {
	      local_row = q; // row into block matrix A, local to vertex
	      break;
	    }
	  }if(local_row < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  local_col  = -1;
	  for(q=0;q<nB;q++){
	    if(Bmap[q] == closure[c]) {
	      local_col = q; // col into block matrix B, local to vertex
	      break;
	    }
	  }if(local_col < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  // B here is B.T in the paper, assembled in column major
	  B[local_col*nA+local_row] += 0.5*sign_row*tdy->V[global_row];

	  // boundary conditions
	  PetscInt isbc;
	  ierr = DMGetLabelValue(dm,"marker",global_row,&isbc);CHKERRQ(ierr);
	  if(isbc == 1){
	    (*tdy->dirichlet)(&(tdy->X[v*dim]),&pdirichlet);
	    G[local_row] += 0.5*sign_row*pdirichlet*tdy->V[global_row];
	  }
	  
	  for(element_col=0;element_col<dim;element_col++){ // which trial function, local to the element/vertex
	    global_col = tdy->emap[closure[c]*nq*dim+element_vertex*dim+element_col]; // DMPlex point index of the face
	    sign_col   = PetscSign(global_col);
	    global_col = PetscAbsInt(global_col);
	    local_col  = -1; // col into block matrix A, local to vertex
	    for(q=0;q<nA;q++){
	      if(Amap[q] == global_col) {
		local_col = q;
		break;
	      }
	    }if(local_col < 0) {
	      printf("Looking for %d in ",global_col);
	      for(q=0;q<nA;q++){ printf("%d ",Amap[q]); }
	      printf("\n");
	      CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
	    }
	    /* Assembled col major, but should be symmetric */
	    A[local_col*nA+local_row] += tdy->Alocal[closure[c]    *(dim*dim*nq)+
						     element_vertex*(dim*dim   )+
						     element_row   *(dim       )+
						     element_col]*sign_row*sign_col*tdy->V[global_row]*tdy->V[global_col];
	  }
	}
    }    
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);
    ierr = FormStencil(&A[0],&B[0],&C[0],&G[0],&D[0],nA,nB);CHKERRQ(ierr);
    
    /* C and D are in column major, but C is always symmetric and D is
       a vector so it should not matter. */
    for(c=0;c<nB;c++){
      ierr = DMPlexGetPointGlobal(dm,Bmap[c],&gStart,&junk);CHKERRQ(ierr);
      if(gStart < 0) continue;
      ierr = VecSetValue(F,gStart,D[c],ADD_VALUES);CHKERRQ(ierr);
      for(q=0;q<nB;q++){
	ierr = DMPlexGetPointGlobal(dm,Bmap[q],&lStart,&junk);CHKERRQ(ierr);
	if (lStart < 0) lStart = -lStart-1;	
	ierr = MatSetValue(K,gStart,lStart,C[q*nB+c],ADD_VALUES);CHKERRQ(ierr);
      }
    }

  }

  /* Integrate in the forcing */
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&gStart,&junk);CHKERRQ(ierr);
    ierr = DMPlexGetPointLocal (dm,c,&lStart,&junk);CHKERRQ(ierr);
    if(gStart < 0) continue;
    ierr = VecSetValue(F,gStart,tdy->Flocal[c],ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TDyWYRecoverVelocity(DM dm,TDy tdy,Vec U)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt v,vStart,vEnd;
  PetscInt   fStart,fEnd;
  PetscInt c,cStart,cEnd;
  PetscInt element_vertex,nA,nB,q,nq,offset,dim,dim2;
  PetscInt element_row,local_row,global_row;
  PetscInt element_col,local_col,global_col;
  PetscScalar A[MAX_LOCAL_SIZE],F[MAX_LOCAL_SIZE],sign_row,sign_col;
  PetscInt Amap[MAX_LOCAL_SIZE],Bmap[MAX_LOCAL_SIZE];
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  PetscSection section;
  PetscScalar *u,pdirichlet;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  nq   = GetNumberOfVertices(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  dim2 = dim*dim;
  for(v=vStart;v<vEnd;v++){ // loop vertices

    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    // determine the size and mapping of the vertex-local systems
    nA = 0; nB = 0;
    for (c = 0; c < closureSize*2; c += 2) {
      if ((closure[c] >= fStart) && (closure[c] < fEnd)) { Amap[nA] = closure[c]; nA += 1; }
      if ((closure[c] >= cStart) && (closure[c] < cEnd)) { Bmap[nB] = closure[c]; nB += 1; }
    }
    ierr = PetscMemzero(A,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(F,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);

    for (c=0;c<closureSize*2;c+=2){ // loop connected cells
      if ((closure[c] < cStart) || (closure[c] >= cEnd)) continue;

	// for the cell, which local vertex is this vertex?
	element_vertex = -1;
	for(q=0;q<nq;q++){
	  if(v == tdy->vmap[closure[c]*nq+q]){
	    element_vertex = q;
	    break;
	  }
	}
	if(element_vertex < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	for(element_row=0;element_row<dim;element_row++){ // which test function, local to the element/vertex
	  global_row = tdy->emap[closure[c]*nq*dim+element_vertex*dim+element_row]; // DMPlex point index of the face
	  sign_row   = PetscSign(global_row);
	  global_row = PetscAbsInt(global_row);
	  local_row  = -1;
	  for(q=0;q<nA;q++){
	    if(Amap[q] == global_row) {
	      local_row = q; // row into block matrix A, local to vertex
	      break;
	    }
	  }if(local_row < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  local_col  = -1;
	  for(q=0;q<nB;q++){
	    if(Bmap[q] == closure[c]) {
	      local_col = q; // col into block matrix B, local to vertex
	      break;
	    }
	  }if(local_col < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  // B P
	  ierr = PetscSectionGetOffset(section,closure[c],&offset);CHKERRQ(ierr);
	  F[local_row] += 0.5*sign_row*u[offset]*tdy->V[global_row];

	  // boundary conditions
	  PetscInt isbc;
	  ierr = DMGetLabelValue(dm,"marker",global_row,&isbc);CHKERRQ(ierr);
	  if(isbc == 1){
	    (*tdy->dirichlet)(&(tdy->X[v*dim]),&pdirichlet);
	    F[local_row] -= 0.5*sign_row*pdirichlet*tdy->V[global_row];
	  }
	  
	  for(element_col=0;element_col<dim;element_col++){ // which trial function, local to the element/vertex
	    global_col = tdy->emap[closure[c]*nq*dim+element_vertex*dim+element_col]; // DMPlex point index of the face
	    sign_col   = PetscSign(global_col);
	    global_col = PetscAbsInt(global_col);
	    local_col  = -1; // col into block matrix A, local to vertex
	    for(q=0;q<nA;q++){
	      if(Amap[q] == global_col) {
		local_col = q;
		break;
	      }
	    }if(local_col < 0) {
	      printf("Looking for %d in ",global_col);
	      for(q=0;q<nA;q++){ printf("%d ",Amap[q]); }
	      printf("\n");
	      CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
	    }
	    // Assembled col major, but should be symmetric
	    A[local_col*nA+local_row] += tdy->Alocal[closure[c]    *(dim2*nq)+
						     element_vertex*(dim2   )+
						     element_row   *(dim    )+
						     element_col]*sign_row*sign_col*tdy->V[global_row]*tdy->V[global_col];
	  }
	}
    }
    /* solves for velocities at a vertex */
    ierr = RecoverVelocity(&A[0],&F[0],nA);CHKERRQ(ierr);

    /* load velocities into structure */
    for(q=0;q<nA;q++){
      global_row = Amap[q];
      const PetscInt *cone;
      ierr = DMPlexGetCone(dm,global_row,&cone);CHKERRQ(ierr);
      offset = -1;
      if(cone[0] == v){
	offset = 0;
      }else if(cone[1] == v){
	offset = 1;
      }else{
	CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
      }
      tdy->vel[dim*(global_row-fStart)+offset] = F[q]; 
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);    
  }
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  norm = sqrt( sum_i^n  V_i * ( p(X_i) - P_i )^2 )
  
  where n is the number of cells.
 */
PetscReal TDyWYPressureNorm(DM dm,TDy tdy,Vec U)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscSection sec;
  PetscInt c,cStart,cEnd,offset,dim,gref,junk;
  PetscReal p,*u,norm,norm_sum;
  norm = 0;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm,&sec);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk);CHKERRQ(ierr);
    if(gref<0) continue;
    ierr = PetscSectionGetOffset(sec,c,&offset);CHKERRQ(ierr);
    tdy->dirichlet(&(tdy->X[c*dim]),&p);
    norm += tdy->V[c]*PetscSqr(u[offset]-p);
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)U));CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(norm_sum);
}

/*

 */
PetscReal TDyWYVelocityNorm(DM dm,TDy tdy)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt nv,i,j,f,fStart,fEnd,c,cStart,cEnd,nf,dim,gref;
  PetscReal flux0,flux,norm,norm_sum,sign,v[3],vn;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  norm = 0;
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&fEnd);CHKERRQ(ierr);
    if (gref < 0) continue;
    const PetscInt *faces;
    ierr = DMPlexGetConeSize(dm,c,&nf   );CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&faces);CHKERRQ(ierr);
    for(i=0;i<nf;i++){
      f = faces[i];
      const PetscInt *verts;
      ierr = DMPlexGetConeSize(dm,f,&nv   );CHKERRQ(ierr);
      ierr = DMPlexGetCone    (dm,f,&verts);CHKERRQ(ierr);
      sign = PetscSign(tdy->N[dim*f  ]*(tdy->X[dim*f  ]-tdy->X[dim*c  ])+
		       tdy->N[dim*f+1]*(tdy->X[dim*f+1]-tdy->X[dim*c+1]));
      flux0 = 0; flux = 0;
      for(j=0;j<nv;j++){
	tdy->flux(&(tdy->X[verts[j]*dim]),&(v[0]));
	vn = v[0]*tdy->N[dim*f] + v[1]*tdy->N[dim*f+1];
	flux0 += sign*vn                        *0.5*tdy->V[f];
	flux  += sign*tdy->vel[dim*(f-fStart)+j]*0.5*tdy->V[f];
      }
      norm += tdy->V[c]/tdy->V[f]*PetscSqr(flux-flux0);
    }
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  PetscFunctionReturn(norm_sum);
}


/*

 */
PetscReal TDyWYDivergenceNorm(DM dm,TDy tdy)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt nv,i,j,f,fStart,fEnd,c,cStart,cEnd,nf,dim,gref;
  PetscReal div0,div,flux0,flux,norm,norm_sum,sign,v[3],vn;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  norm = 0;
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&fEnd);CHKERRQ(ierr);
    if (gref < 0) continue;
    const PetscInt *faces;
    ierr = DMPlexGetConeSize(dm,c,&nf   );CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&faces);CHKERRQ(ierr);
    div0 = 0; div = 0;
    for(i=0;i<nf;i++){
      f = faces[i];
      const PetscInt *verts;
      ierr = DMPlexGetConeSize(dm,f,&nv   );CHKERRQ(ierr);
      ierr = DMPlexGetCone    (dm,f,&verts);CHKERRQ(ierr);
      sign = PetscSign(tdy->N[dim*f  ]*(tdy->X[dim*f  ]-tdy->X[dim*c  ])+
		       tdy->N[dim*f+1]*(tdy->X[dim*f+1]-tdy->X[dim*c+1]));
      flux0 = 0; flux = 0;
      for(j=0;j<nv;j++){
	tdy->flux(&(tdy->X[verts[j]*dim]),&(v[0]));
	vn = v[0]*tdy->N[dim*f] + v[1]*tdy->N[dim*f+1];
	flux0 += sign*vn                        *0.5*tdy->V[f];
	flux  += sign*tdy->vel[dim*(f-fStart)+j]*0.5*tdy->V[f];
	div0  += flux0;
	div   += flux; 
      }
    }
    norm += tdy->V[c]*PetscSqr(div-div0);
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  PetscFunctionReturn(norm_sum);
}
