#include <private/tdycoreimpl.h>
#include <private/tdyfeimpl.h>
#include <private/tdyutils.h>
#include <private/tdywyimpl.h>
#include <petscblaslapack.h>
#include <private/tdydiscretizationimpl.h>
#include <tdytimers.h>

#define MAX_LOCAL_SIZE 144

PetscErrorCode Pullback(PetscScalar *K,PetscScalar *DFinv,PetscScalar *Kappa,
                        PetscScalar J,PetscInt nn) {
  /*
    K(dxd)     flattened array in row major (but doesn't matter as it is symmetric)
    DFinv(dxd) flattened array in row major format (how PETSc generates it)
    J          det(DF)

    returns Kappa^-1 = ( J DF^-1 K (DF^-1)^T )^-1
   */

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  PetscScalar  zero=0,one=1;
  PetscBLASInt n,lwork=nn*nn;
  ierr = PetscBLASIntCast(nn,&n); CHKERRQ(ierr);
  PetscBLASInt info,*pivots;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&pivots); CHKERRQ(ierr);

  PetscScalar KDFinvT[n*n],work[n*n];
  /* LAPACK wants things in column major, so we need to transpose both
     K and DFinv. However, we are computing K (DF^-1)^T and thus we
     leave DFinv as is. The resulting KDFinvT is in column major
     format. */
  BLASgemm_("T","N",&n,&n,&n,&one,K,&n,DFinv,&n,&zero,KDFinvT,&n);
  /* Here we are computing J * DFinv * KDFinvT. Since DFinv is row
     major and LAPACK wants things column major, we need to transpose
     it. */
  BLASgemm_("T","N",&n,&n,&n,&J,DFinv,&n,KDFinvT,&n,&zero,&Kappa[0],&n);

  // Find LU factors of Kappa
  LAPACKgetrf_(&n,&n,&Kappa[0],&n,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  // Find inverse
  LAPACKgetri_(&n,&Kappa[0],&n,pivots,work,&lwork,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"illegal argument value");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"singular matrix");

  ierr = PetscFree(pivots); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode FormStencil(PetscScalar *A,PetscScalar *B,PetscScalar *C,
                           PetscScalar *G,PetscScalar *D,
                           PetscInt qq,PetscInt rr) {
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // return C(rxr) = B.T A^-1 B in col major
  //        D(r  ) = B.T A^-1 G in col major

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscBLASInt q,r,o = 1,info,*pivots;
  PetscScalar zero = 0,one = 1;
  ierr = PetscBLASIntCast(qq,&q); CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rr,&r); CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots); CHKERRQ(ierr);

  // Copy B because we will need it again
  PetscScalar AinvB[qq*rr];
  ierr = PetscMemcpy(AinvB,B,sizeof(PetscScalar)*(qq*rr));
  CHKERRQ(ierr); // AinvB in col major

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  // Solve AinvB = (A^-1 * B) by back-substitution
  LAPACKgetrs_("N",&q,&r,A,&q,pivots,AinvB,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Solve G = (A^-1 * G) by back-substitution
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,G,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (B.T * AinvB) and (B.T * G)
  BLASgemm_("T","N",&r,&r,&q,&one,B,&q,AinvB,&q,&zero,&C[0],&r); // B.T * AinvB
  BLASgemm_("T","N",&r,&o,&q,&one,B,&q,G,&q,&zero,&D[0],&r);     // B.T * G

  ierr = PetscFree(pivots); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode RecoverVelocity(PetscScalar *A,PetscScalar *F,PetscInt qq) {
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // return U = A^-1 (G - B P)
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscBLASInt q,o = 1,info,*pivots;
  ierr = PetscBLASIntCast(qq,&q); CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots); CHKERRQ(ierr);

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  // Solve by back-substitution
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,F,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  ierr = PetscFree(pivots); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscReal ElementSide(PetscInt q,PetscInt d) {
  PetscInt i,mod = 1;
  for(i=1; i<=d; i++) mod *= 2;
  return ((((q / mod) % 2 )== 0) ? -1 : +1 );
}

PetscReal KDotADotB(PetscReal *K,PetscReal *A,PetscReal *B,PetscInt dim) {
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

PetscErrorCode TDyWYLocalElementCompute(TDy tdy) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd;
  PetscInt dim,dim2,i,j,q,nq;
  PetscReal wgt   = 1;    // 1/s from the paper
  PetscReal Ehat  = 1;    // area of ref element cell ( [-1,1]^(dim  ) )
  PetscReal ehat  = 1;    // area of ref element face ( [-1,1]^(dim-1) )
  PetscScalar x[24],DF[72],DFinv[72],J[8],Kinv[9],n0[3],n1[3],
              f; /* allocated at maximum possible size */
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  TDyWY *wy = tdy->context;
  Conditions *conditions = tdy->conditions;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nq   = wy->ncv;
  dim2 = dim*dim;
  for(i=0; i<dim; i++) {
    Ehat *= 2;
    wgt  *= 0.5;
  }
  ehat = Ehat / 2;
  for(c=cStart; c<cEnd; c++) {
    wy->Flocal[c] = 0;
    // DF and DFinv are row major
    ierr = DMPlexComputeCellGeometryFEM(dm,c,wy->quad,x,DF,DFinv,J); CHKERRQ(ierr);
    for(q=0; q<nq; q++) {

      if(J[q]<0){
        PetscPrintf(((PetscObject)dm)->comm,"cell %d:  DF = \n",c);
        PrintMatrix(DF,dim,dim,PETSC_TRUE);
        SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Determinant of the jacobian is negative");
      }
      // compute Kappa^-1 which will be in column major format (shouldn't matter as it is symmetric)
      ierr = Pullback(&(wy->K[dim2*(c-cStart)]),&DFinv[dim2*q],Kinv,J[q],dim);
      CHKERRQ(ierr);

      // at each vertex, we have a dim by dim system which we will store
      for(i=0; i<dim; i++) {
        n0[0] = 0; n0[1] = 0; n0[2] = 0;
        n0[i] = ElementSide(q,i);
        for(j=0; j<dim; j++) {
          n1[0] = 0; n1[1] = 0; n1[2] = 0;
          n1[j] = ElementSide(q,j);
          wy->Alocal[c*(dim2*nq)+q*(dim2)+j*dim+i] = KDotADotB(Kinv,n0,n1,
              dim)*Ehat/ehat/ehat*wgt;
        }
      }

      // integrate the forcing function using the same quadrature
      if (ConditionsHasForcing(conditions)) {
        ierr = ConditionsComputeForcing(conditions, 1, &(x[q*dim]), &f);CHKERRQ(ierr);
        wy->Flocal[c] += f*J[q];
      }
    }

  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyWYSetQuadrature(TDy tdy, TDyQuadratureType qtype) {
  PetscFunctionBegin;
  TDyWY *wy = tdy->context;
  wy->qtype = qtype;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreate_WY(void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // Allocate a new context for the WY method.
  TDyWY *wy;
  ierr = PetscCalloc(sizeof(TDyWY), &wy); CHKERRQ(ierr);
  *context = wy;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy_WY(void *context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyWY *wy = context;

  if (wy->vmap  ) { ierr = PetscFree(wy->vmap  ); CHKERRQ(ierr); }
  if (wy->emap  ) { ierr = PetscFree(wy->emap  ); CHKERRQ(ierr); }
  if (wy->Alocal) { ierr = PetscFree(wy->Alocal); CHKERRQ(ierr); }
  if (wy->Flocal) { ierr = PetscFree(wy->Flocal); CHKERRQ(ierr); }
  if (wy->vel   ) { ierr = PetscFree(wy->vel   ); CHKERRQ(ierr); }
  if (wy->fmap  ) { ierr = PetscFree(wy->fmap  ); CHKERRQ(ierr); }
  if (wy->faces ) { ierr = PetscFree(wy->faces ); CHKERRQ(ierr); }
  if (wy->quad  ) { ierr = PetscQuadratureDestroy(&(wy->quad)); CHKERRQ(ierr); }

  if (wy->Sr) { ierr = PetscFree(wy->Sr); CHKERRQ(ierr); }
  if (wy->dS_dP) { ierr = PetscFree(wy->dS_dP); CHKERRQ(ierr); }
  if (wy->d2S_dP2) { ierr = PetscFree(wy->d2S_dP2); CHKERRQ(ierr); }
  if (wy->S) { ierr = PetscFree(wy->S); CHKERRQ(ierr); }
  if (wy->Kr) { ierr = PetscFree(wy->Kr); CHKERRQ(ierr); }
  if (wy->dKr_dS) { ierr = PetscFree(wy->dKr_dS); CHKERRQ(ierr); }
  if (wy->porosity) { ierr = PetscFree(wy->porosity); CHKERRQ(ierr); }
  if (wy->K0) { ierr = PetscFree(wy->K0); CHKERRQ(ierr); }
  if (wy->K) { ierr = PetscFree(wy->K); CHKERRQ(ierr); }

  ierr = PetscFree(wy->V); CHKERRQ(ierr);
  ierr = PetscFree(wy->X); CHKERRQ(ierr);
  ierr = PetscFree(wy->N); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscInt TDyGetNumDMFields_WY(void *context) {
  PetscFunctionBegin;
  PetscInt ndof = 1; // Liquid pressure
  PetscFunctionReturn(ndof);
}

PetscErrorCode TDySetFromOptions_WY(void *context, TDyOptions *options) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  TDyWY *wy = context;

  // Set defaults.
  wy->Pref = 101325;

  // Set options.
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: WY options",""); CHKERRQ(ierr);
  TDyQuadratureType qtype = FULL;
  ierr = PetscOptionsEnum("-tdy_quadrature","Quadrature type for finite element methods",
    "TDyWYSetQuadrature",TDyQuadratureTypes,(PetscEnum)qtype,
    (PetscEnum *)&wy->qtype,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Set characteristic curve data.
  wy->vangenuchten_m = options->vangenuchten_m;
  wy->vangenuchten_alpha = options->vangenuchten_alpha;
  wy->mualem_poly_x0 = options->mualem_poly_x0;
  wy->mualem_poly_x1 = options->mualem_poly_x1;
  wy->mualem_poly_x2 = options->mualem_poly_x2;
  wy->mualem_poly_dx = options->mualem_poly_dx;

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDMFields_WY(void *context, DM dm) {
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

PetscErrorCode TDySetup_WY(void *context, DM dm, EOS *eos,
                           MaterialProp *matprop, CharacteristicCurves *cc,
                           Conditions* conditions) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt i,dim,c,cStart,cEnd,f,fStart,fEnd,vStart,vEnd,p,pStart,pEnd;
  PetscInt  closureSize,  *closure;

  TDyWY *wy = context;

  // Compute/store plex geometry.
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  wy->dim = dim;
  PetscLogEvent t1 = TDyGetTimer("ComputePlexGeometry");
  TDyStartTimer(t1);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  PetscInt eStart, eEnd;
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(wy->V));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(wy->X));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(wy->N));
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
      for(PetscInt d=0; d<dim; d++) wy->X[p*dim+d] = coords[offset+d];
    } else {
      if((dim == 3) && (p >= eStart) && (p < eEnd)) continue;
      PetscLogEvent t11 = TDyGetTimer("DMPlexComputeCellGeometryFVM");
      TDyStartTimer(t11);
      ierr = DMPlexComputeCellGeometryFVM(dm,p,&(wy->V[p]),
                                          &(wy->X[p*dim]),
                                          &(wy->N[p*dim])); CHKERRQ(ierr);
      TDyStopTimer(t11);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);

  /* Check that the number of vertices per cell are constant. Soft
     limitation, method is flexible but my data structures are not. */
  wy->ncv = TDyGetNumberOfCellVertices(dm);

  /* Create a PETSc quadrature, we don't really use this, it is just
     to evaluate the Jacobian via the PETSc interface. */
  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = PetscQuadratureCreate(comm,&(wy->quad)); CHKERRQ(ierr);
  ierr = SetQuadrature(wy->quad,dim); CHKERRQ(ierr);

  /* Build vmap and emap */
  ierr = CreateCellVertexMap(dm, wy->ncv, wy->X, &(wy->vmap)); CHKERRQ(ierr);
  ierr = CreateCellVertexDirFaceMap(dm, wy->ncv, wy->X, wy->N, wy->vmap,
                                    &(wy->emap)); CHKERRQ(ierr);

  /* Allocate space for Alocal and Flocal */
  ierr = PetscMalloc(dim*dim*wy->ncv*(cEnd-cStart)*sizeof(PetscReal),
                     &(wy->Alocal)); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),
                     &(wy->Flocal)); CHKERRQ(ierr);

  /* Allocate space for velocities and create a local_vertex->face map */
  wy->nfv = TDyGetNumberOfFaceVertices(dm);
  ierr = PetscMalloc(wy->nfv*(fEnd-fStart)*sizeof(PetscReal),
                     &(wy->vel )); CHKERRQ(ierr);
  ierr = PetscMalloc(wy->nfv*(fEnd-fStart)*sizeof(PetscInt ),
                     &(wy->fmap)); CHKERRQ(ierr);
  for(f=fStart; f<fEnd; f++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    c = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) {
        wy->fmap[wy->nfv*(f-fStart)+c] = closure[i];
        c += 1;
      }
    }
    #if defined(PETSC_USE_DEBUG)
    if(c != wy->nfv) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(face,local_vertex) -> vertex");
    }
    #endif
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,
                                          &closureSize,&closure); CHKERRQ(ierr);
  }

  /* map(cell,dim,side) --> global_face */
  ierr = PetscMalloc((cEnd-cStart)*(2*dim)*sizeof(PetscInt),&(wy->faces));
  CHKERRQ(ierr);
  #if defined(PETSC_USE_DEBUG)
  for(c=0; c<((cEnd-cStart)*(2*dim)); c++) { wy->faces[c] = -1; }
  #endif
  PetscInt v,d,s;
  for(c=cStart; c<cEnd; c++) {
    for(d=0; d<dim; d++) {
      for(s=0; s<2; s++) {
        v = s*PetscPowInt(2,d);
        wy->faces[(c-cStart)*dim*2+d*2+s] = PetscAbsInt(wy->emap[(c-cStart)*wy->ncv*dim
                                             +v*dim+d]);
      }
    }
  }
  #if defined(PETSC_USE_DEBUG)
  for(c=0; c<((cEnd-cStart)*(2*dim)); c++) {
    if(wy->faces[c] < 0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,dir,side) -> face");
    }
  }
  #endif

  // Initialize material properties.
  PetscInt nc = cEnd-cStart;
  ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(wy->K)); CHKERRQ(ierr);
  ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(wy->K0)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->porosity)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->Kr)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->dKr_dS)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->S)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->dS_dP)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->d2S_dP2)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(wy->Sr)); CHKERRQ(ierr);

  PetscInt points[nc];
  for (PetscInt c = 0; c < nc; ++c) {
    points[c] = cStart + c;
  }

  // By default, we use the Van Genuchten saturation model.
  {
    PetscReal parameters[2*nc];
    for (PetscInt c = 0; c < nc; ++c) {
      parameters[2*c]   = wy->vangenuchten_m;
      parameters[2*c+1] = wy->vangenuchten_alpha;
    }
    ierr = SaturationSetType(cc->saturation, SAT_FUNC_VAN_GENUCHTEN, nc, points,
                             parameters); CHKERRQ(ierr);
  }

  // By default, we use the the Mualem relative permeability model.
  {
    PetscInt num_params = 9;
    PetscReal parameters[num_params*nc];
    for (PetscInt c = 0; c < nc; ++c) {
      PetscReal m = wy->vangenuchten_m;
      PetscReal poly_x0 = wy->mualem_poly_x0;
      PetscReal poly_x1 = wy->mualem_poly_x1;
      PetscReal poly_x2 = wy->mualem_poly_x2;
      PetscReal poly_dx = wy->mualem_poly_dx;

      PetscInt offset = num_params*c;
      parameters[offset    ]   = m;
      parameters[offset + 1] = poly_x0;
      parameters[offset + 2] = poly_x1;
      parameters[offset + 3] = poly_x2;
      parameters[offset + 4] = poly_dx;

      // Set up cubic polynomial coefficients for the cell.
      PetscReal coeffs[4];
      ierr = RelativePermeability_Mualem_GetSmoothingCoeffs(m, poly_x0, poly_x1, poly_x2, poly_dx, coeffs);
      CHKERRQ(ierr);
      parameters[offset + 5] = coeffs[0];
      parameters[offset + 6] = coeffs[1];
      parameters[offset + 7] = coeffs[2];
      parameters[offset + 8] = coeffs[3];
    }
    ierr = RelativePermeabilitySetType(cc->rel_perm, REL_PERM_FUNC_MUALEM, nc,
                                       points, parameters); CHKERRQ(ierr);
  }

  // Compute material properties.
  ierr = MaterialPropComputePermeability(matprop, nc, wy->X, wy->K0); CHKERRQ(ierr);
  memcpy(wy->K, wy->K0, 9*sizeof(PetscReal)*nc);
  MaterialPropComputePorosity(matprop, nc, wy->X, wy->porosity);

  /* Setup the section, 1 dof per cell */
  PetscSection sec;
  ierr = PetscSectionCreate(comm,&sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Pressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1); CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

#if 0
/*
  nq = nq1d^2 = 9
  x  = nq*dim = 27
  DF = dim^2 *nq = 81
 */
PetscErrorCode IntegrateOnFaceConstant(TDy tdy,PetscInt c,PetscInt f,
				       PetscReal *integral) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  PetscReal value = 0;
  PetscInt dim;
  (*integral) = 0;
  ierr = DMGetDimension((&(&tdy->discretization)->tdydm)->dm,&dim); CHKERRQ(ierr);

  if (tdy->ops->compute_boundary_pressure) {
    ierr = (*tdy->ops->compute_boundary_pressure)(tdy,
  					      &(tdy->X[f*dim]),
  					      &value,
  					      tdy->boundary_pressure_ctx);CHKERRQ(ierr);
  }
  (*integral) = value*tdy->V[f];
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode IntegrateOnFace(TDy tdy,PetscInt c,PetscInt f,
                               PetscReal *integral) {
  TDY_START_FUNCTION_TIMER()

  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt v,ncv,i,j,d,q,nq,dim,face_dir,face_side,fStart,fEnd,lside[24],nq1d = 2;
  PetscQuadrature quadrature;
  const PetscScalar *quad_x,*quad_w;
  PetscReal xq[3],x[27],J[9],N[24],DF[81],DFinv[81],value;
  DM dm = (&(&tdy->discretization)->tdydm)->dm;
  ncv  = tdy->ncv;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&quadrature);
  CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&nq,&quad_x,&quad_w);
  CHKERRQ(ierr);
  ierr = DMPlexComputeCellGeometryFEM(dm,f,quadrature,x,DF,DFinv,J);
  CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

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

  /* relative to this cell, where is this face? */
  face_side = -1;
  face_dir  = -1;
  for(v=0; v<ncv; v++) {
    for(d=0; d<dim; d++) {
      if(PetscAbsInt(tdy->emap[c*ncv*dim+v*dim+d]) == f) {
        face_side = lside[v*dim+d];
        face_dir  = d;
      }
    }
  }

  /* loop over quadrature points */
  (*integral) = 0;
  for(q=0; q<nq; q++) {

    /* extend the dim-1 quadrature point to dim */
    j = 0;
    xq[0] = 0; xq[1] = 0; xq[2] = 0;
    for(i=0; i<dim; i++) {
      if(i == face_dir) {
        xq[i] = PetscPowInt(-1,face_side+1);
      } else {
        xq[i] = quad_x[q*(dim-1)+j];
        j += 1;
      }
    }

    /* <g,v.n> */
    if (tdy->ops->compute_boundary_pressure) {
      ierr = (*tdy->ops->compute_boundary_pressure)(tdy, &(x[dim*q]), &value, tdy->boundary_pressure_ctx);CHKERRQ(ierr);
    }

    if(dim==2) {
      //HdivBasisQuad(xq,N);
    } else {
      //HdivBasisHex(xq,N);
    }
    for(v=0; v<ncv; v++) {
      for(d=0; d<dim; d++) {
        if(PetscAbsInt(tdy->emap[c*ncv*dim+v*dim+d]) == f) {
          (*integral) += value*TDyADotB(&(tdy->N[f*dim]),&(N[v*dim]),dim)*quad_w[q]*J[q];
        }
      }
    }
  }
  ierr = PetscQuadratureDestroy(&quadrature); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode TDyWYComputeSystem(TDy tdy,Mat K,Vec F) {
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt v,vStart,vEnd;
  PetscInt   fStart,fEnd;
  PetscInt c,cStart,cEnd;
  PetscInt   gStart,lStart,junk;
  PetscInt element_vertex,nA,nB,q,dim,dim2,nq;
  PetscInt element_row,local_row,global_row;
  PetscInt element_col,local_col,global_col;
  PetscScalar A[MAX_LOCAL_SIZE],B[MAX_LOCAL_SIZE],C[MAX_LOCAL_SIZE],
              G[MAX_LOCAL_SIZE],D[MAX_LOCAL_SIZE],sign_row,sign_col;
  PetscInt Amap[MAX_LOCAL_SIZE],Bmap[MAX_LOCAL_SIZE];
  PetscScalar pdirichlet,wgt,tol=1e4*PETSC_MACHINE_EPSILON;
  TDyWY *wy = tdy->context;
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  Conditions *conditions = tdy->conditions;
  PetscFunctionBegin;

  ierr = TDyWYLocalElementCompute(tdy); CHKERRQ(ierr);
  nq   = wy->ncv;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  wgt  = PetscPowReal(0.5,dim-1);

  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) { // loop vertices

    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);
    CHKERRQ(ierr);

    // determine the size and mapping of the vertex-local systems
    nA = 0; nB = 0;
    for (c = 0; c < closureSize*2; c += 2) {
      if ((closure[c] >= fStart) && (closure[c] < fEnd)) { Amap[nA] = closure[c]; nA += 1; }
      if ((closure[c] >= cStart) && (closure[c] < cEnd)) { Bmap[nB] = closure[c]; nB += 1; }
    }
    #if defined(PETSC_USE_DEBUG)
    if(PetscMax(nA,nB)*PetscMax(nA,nB) > MAX_LOCAL_SIZE) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "MAX_LOCAL_SIZE not set large enough");
    }
    #endif
    ierr = PetscMemzero(A,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(B,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(C,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(G,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(D,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);

    for (c=0; c<closureSize*2; c+=2) { // loop connected cells
      PetscInt c1 = closure[c]; // connected cell index
      if ((c1 < cStart) || (c1 >= cEnd)) continue;

      // for the cell, which local vertex is this vertex?
      element_vertex = -1;
      for(q=0; q<nq; q++) {
        if(v == wy->vmap[c1*nq+q]) {
          element_vertex = q;
          break;
        }
      }
      if(element_vertex < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

      for(element_row=0; element_row<dim;
          element_row++) { // which test function, local to the element/vertex
        global_row = wy->emap[c1*nq*dim+element_vertex*dim
                               +element_row]; // DMPlex point index of the face
        sign_row   = PetscSign(global_row);
        global_row = PetscAbsInt(global_row);
        local_row  = -1;
        for(q=0; q<nA; q++) {
          if(Amap[q] == global_row) {
            local_row = q; // row into block matrix A, local to vertex
            break;
          }
        } if(local_row < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

        local_col  = -1;
        for(q=0; q<nB; q++) {
          if(Bmap[q] == c1) {
            local_col = q; // col into block matrix B, local to vertex
            break;
          }
        } if(local_col < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

        // B here is B.T in the paper, assembled in column major
        B[local_col*nA+local_row] += wgt*sign_row*wy->V[global_row];

        // Pressure boundary conditions
        PetscInt isbc;
        ierr = DMGetLabelValue(dm,"boundary",global_row,&isbc); CHKERRQ(ierr);
        if(isbc == 1 && ConditionsHasBoundaryPressure(conditions)) {
          //ierr = IntegrateOnFace(tdy,c1,global_row,&pdirichlet); CHKERRQ(ierr);
          //G[local_row] = wgt*pdirichlet;
          ierr = ConditionsComputeBoundaryPressure(conditions, 1,
                                                   &(wy->X[global_row*dim]),
                                                   &pdirichlet); CHKERRQ(ierr);
          G[local_row] = wgt*sign_row*pdirichlet*wy->V[global_row];
        }

        for(element_col=0; element_col<dim;
            element_col++) { // which trial function, local to the element/vertex
          global_col = wy->emap[c1*nq*dim+element_vertex*dim
                                 +element_col]; // DMPlex point index of the face
          sign_col   = PetscSign(global_col);
          global_col = PetscAbsInt(global_col);
          local_col  = -1; // col into block matrix A, local to vertex
          for(q=0; q<nA; q++) {
            if(Amap[q] == global_col) {
              local_col = q;
              break;
            }
          } if(local_col < 0) {
            printf("Looking for %d in ",global_col);
            for(q=0; q<nA; q++) { printf("%d ",Amap[q]); }
            printf("\n");
            CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
          }
          /* Assembled col major, but should be symmetric */
          A[local_col*nA+local_row] += wy->Alocal[c1*(dim2*nq)+
                                       element_vertex*(dim2   )+
                                       element_row   *(dim    )+
                                       element_col]*sign_row*sign_col*wy->V[global_row]*wy->V[global_col];
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);
    CHKERRQ(ierr);
    ierr = FormStencil(&A[0],&B[0],&C[0],&G[0],&D[0],nA,nB); CHKERRQ(ierr);

    /* C and D are in column major, but C is always symmetric and D is
       a vector so it should not matter. */
    PetscReal maxC = 0;
    for(c=0; c<nB*nB; c++) { maxC = PetscMax(maxC,PetscAbsReal(C[c])); }
    for(c=0; c<nB; c++) {
      ierr = DMPlexGetPointGlobal(dm,Bmap[c],&gStart,&junk); CHKERRQ(ierr);
      if(gStart < 0) continue;
      ierr = VecSetValue(F,gStart,D[c],ADD_VALUES); CHKERRQ(ierr);
      for(q=0; q<nB; q++) {
        if (PetscAbsReal(C[q*nB+c])<(tol*maxC)) continue;
        ierr = DMPlexGetPointGlobal(dm,Bmap[q],&lStart,&junk); CHKERRQ(ierr);
        if (lStart < 0) lStart = -lStart-1;
        ierr = MatSetValue(K,gStart,lStart,C[q*nB+c],ADD_VALUES); CHKERRQ(ierr);
      }
    }

  }

  /* Integrate in the forcing */
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gStart,&junk); CHKERRQ(ierr);
    ierr = DMPlexGetPointLocal (dm,c,&lStart,&junk); CHKERRQ(ierr);
    if(gStart < 0) continue;
    ierr = VecSetValue(F,gStart,wy->Flocal[c],ADD_VALUES); CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyWYRecoverVelocity(TDy tdy,Vec U) {
  TDY_START_FUNCTION_TIMER()
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt v,vStart,vEnd;
  PetscInt   fStart,fEnd;
  PetscInt c,cStart,cEnd;
  PetscInt element_vertex,nA,nB,q,nq,nv,offset,dim,dim2;
  PetscInt element_row,local_row,global_row;
  PetscInt element_col,local_col,global_col;
  PetscScalar A[MAX_LOCAL_SIZE],F[MAX_LOCAL_SIZE],sign_row,sign_col;
  PetscInt Amap[MAX_LOCAL_SIZE],Bmap[MAX_LOCAL_SIZE];
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  TDyWY *wy = tdy->context;
  Conditions *conditions = tdy->conditions;

  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscSection section;
  PetscScalar *u,pdirichlet,wgt;
  Vec localU;
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  nq   = wy->ncv;
  nv   = wy->nfv;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  wgt  = PetscPowReal(0.5,dim-1);

  for(v=vStart; v<vEnd; v++) { // loop vertices

    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);
    CHKERRQ(ierr);

    // determine the size and mapping of the vertex-local systems
    nA = 0; nB = 0;
    for (c = 0; c < closureSize*2; c += 2) {
      if ((closure[c] >= fStart) && (closure[c] < fEnd)) { Amap[nA] = closure[c]; nA += 1; }
      if ((closure[c] >= cStart) && (closure[c] < cEnd)) { Bmap[nB] = closure[c]; nB += 1; }
    }
    ierr = PetscMemzero(A,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);
    ierr = PetscMemzero(F,sizeof(PetscScalar)*MAX_LOCAL_SIZE); CHKERRQ(ierr);

    for (c=0; c<closureSize*2; c+=2) { // loop connected cells
      if ((closure[c] < cStart) || (closure[c] >= cEnd)) continue;

      // for the cell, which local vertex is this vertex?
      element_vertex = -1;
      for(q=0; q<nq; q++) {
        if(v == wy->vmap[closure[c]*nq+q]) {
          element_vertex = q;
          break;
        }
      }
      if(element_vertex < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

      for(element_row=0; element_row<dim;
          element_row++) { // which test function, local to the element/vertex
        global_row = wy->emap[closure[c]*nq*dim+element_vertex*dim
                               +element_row]; // DMPlex point index of the face
        sign_row   = PetscSign(global_row);
        global_row = PetscAbsInt(global_row);
        local_row  = -1;
        for(q=0; q<nA; q++) {
          if(Amap[q] == global_row) {
            local_row = q; // row into block matrix A, local to vertex
            break;
          }
        } if(local_row < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

        local_col  = -1;
        for(q=0; q<nB; q++) {
          if(Bmap[q] == closure[c]) {
            local_col = q; // col into block matrix B, local to vertex
            break;
          }
        } if(local_col < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

        // B P
        ierr = PetscSectionGetOffset(section,closure[c],&offset); CHKERRQ(ierr);
        F[local_row] += wgt*sign_row*u[offset]*wy->V[global_row];

        // boundary conditions
        PetscInt isbc;
        ierr = DMGetLabelValue(dm,"boundary",global_row,&isbc); CHKERRQ(ierr);
        if(isbc == 1 && ConditionsHasBoundaryPressure(conditions)) {
          //ierr = IntegrateOnFaceConstant(tdy,closure[c],global_row,&pdirichlet); CHKERRQ(ierr);
          //F[local_row] -= wgt*sign_row*pdirichlet;
          ierr = ConditionsComputeBoundaryPressure(conditions, 1,
                                                   &(wy->X[global_row*dim]),
                                                   &pdirichlet); CHKERRQ(ierr);
          F[local_row] += -wgt*sign_row*pdirichlet*wy->V[global_row];
        }

        for(element_col=0; element_col<dim;
            element_col++) { // which trial function, local to the element/vertex
          global_col = wy->emap[closure[c]*nq*dim+element_vertex*dim
                                 +element_col]; // DMPlex point index of the face
          sign_col   = PetscSign(global_col);
          global_col = PetscAbsInt(global_col);
          local_col  = -1; // col into block matrix A, local to vertex
          for(q=0; q<nA; q++) {
            if(Amap[q] == global_col) {
              local_col = q;
              break;
            }
          } if(local_col < 0) {
            printf("Looking for %d in ",global_col);
            for(q=0; q<nA; q++) { printf("%d ",Amap[q]); }
            printf("\n");
            CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
          }
          // Assembled col major, but should be symmetric
          A[local_col*nA+local_row] += wy->Alocal[closure[c]    *(dim2*nq)+
                                       element_vertex*(dim2   )+
                                       element_row   *(dim    )+
                                       element_col]*sign_row*sign_col*wy->V[global_row]*wy->V[global_col];
        }
      }
    }
    /* solves for velocities at a vertex */
    ierr = RecoverVelocity(&A[0],&F[0],nA); CHKERRQ(ierr);

    /* load velocities into structure */
    for(q=0; q<nA; q++) {
      global_row = Amap[q];
      for(offset=0; offset<nv; offset++) {
        if(v == wy->fmap[nv*(global_row-fStart)+offset]) {
          wy->vel[nv*(global_row-fStart)+offset] = F[q];
          break;
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* Compute pressure and velocity norms.
  Pressure norm = sqrt( sum_i^n  V_i * ( p(X_i) - P_i )^2 )
  Velocity norm given in section 5 of Wheeler2012.

  ||u-uh||^2 = sum_E sum_e |E| ( 1/|e| int(u.n) - 1/|e| int(uh.n) )^2

  where the integrals are evaluated by nq1d=1 quadrature. It compares
  the L2 difference in the mean normal velocity over faces of each
  cell, weighted by the cell area.
*/
PetscErrorCode TDyComputeErrorNorms_WY(void *context, DM dm, Conditions *conditions,
                                       Vec U, PetscReal *p_norm, PetscReal *v_norm) {
  TDY_START_FUNCTION_TIMER()
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd,offset,dim,gref,fStart,fEnd,junk,d,s,f;
  PetscSection sec;
  PetscReal p,*u,norm,norm_sum;

  TDyWY *wy = context;

  if (p_norm) {
    if(!ConditionsHasBoundaryPressure(conditions)) {
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
      ierr = ConditionsComputeBoundaryPressure(conditions, 1, &(wy->X[c*dim]),
                                               &p);CHKERRQ(ierr);
      norm += wy->V[c]*PetscSqr(u[offset]-p);
    }
    ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
        PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);
    norm_sum = PetscSqrtReal(norm_sum);
    ierr = VecRestoreArray(U,&u); CHKERRQ(ierr);
  }

  if (v_norm) {
    if(!ConditionsHasBoundaryVelocity(conditions)) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
          "Must set the velocity function with TDySetDirichletFluxFunction");
    }
    ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

    PetscInt i,j,ncv,nfv,q,nq,nq1d=1;
    const PetscScalar *quad_x,*quad_w;
    PetscReal xq[3],x[100],DF[100],DFinv[100],J[100],N[24],vel[3],ve,va,flux0,flux,
              norm,norm_sum,Nn,C;
    PetscQuadrature quad;
    ncv  = wy->ncv;
    nfv  = wy->nfv;
    ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&quad); CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,&quad_x,&quad_w);
    CHKERRQ(ierr);

    /* loop cells */
    norm = 0; norm_sum = 0;
    for(c=cStart; c<cEnd; c++) {
      ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk); CHKERRQ(ierr);
      if (gref < 0) continue;

      /* loop faces */
      for(d=0; d<dim; d++) {
        for(s=0; s<2; s++) {
          f = wy->faces[(c-cStart)*dim*2+d*2+s];
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
            if(dim==2) {
              //HdivBasisQuad(xq,N);
            } else {
              //HdivBasisHex(xq,N);
            }
            va = 0;
            for(i=0; i<ncv; i++) {
              Nn = PetscPowInt(-1,s+1)*N[i*dim+d];
              for(j=0; j<nfv; j++) {
                if(wy->vmap[ncv*(c-cStart)+i] == wy->fmap[nfv*(f-fStart)+j]) break;
              }
              if(j==nfv) continue;
              C  = wy->vel[nfv*(f-fStart)+j];
              va += Nn*C;
            }
            ierr = ConditionsComputeBoundaryVelocity(conditions, 1, &(x[q*dim]),
                                                     vel);CHKERRQ(ierr);
            ve = TDyADotB(vel,&(wy->N[dim*f]),dim);
            flux  += va*quad_w[q]*J[q];
            flux0 += ve*quad_w[q]*J[q];
          }
          norm += PetscSqr((flux-flux0)/wy->V[f])*wy->V[c];
        }
      }
    }
    ierr = PetscQuadratureDestroy(&quad); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
        PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
    norm_sum = PetscSqrtReal(norm_sum);
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(norm_sum);
}

PetscErrorCode TDyUpdateState_WY(void *context, DM dm,
                                 EOS *eos, MaterialProp *matprop,
                                 CharacteristicCurves *cc,
                                 PetscInt num_cells, PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyWY *wy = context;

  PetscInt cStart = 0, cEnd = num_cells;
  PetscInt nc = cEnd - cStart;

  // Compute the capillary pressure on all cells.
  PetscReal Pc[nc];
  for (PetscInt c=0;c<nc;c++) {
    Pc[c] = wy->Pref - U[c];
  }

  // Compute the saturation and its derivatives.
  ierr = SaturationCompute(cc->saturation, wy->Sr, Pc, wy->S, wy->dS_dP,
                           wy->d2S_dP2); CHKERRQ(ierr);

  // Compute the effective saturation on cells.
  PetscReal Se[nc];
  for (PetscInt c=0;c<nc;c++) {
    Se[c] = (wy->S[c] - wy->Sr[c])/(1.0 - wy->Sr[c]);
  }

  // Compute the relative permeability and its derivative (w.r.t. Se).
  ierr = RelativePermeabilityCompute(cc->rel_perm, Se, wy->Kr, wy->dKr_dS);
  CHKERRQ(ierr);

  // Correct dKr/dS using the chain rule, and update the permeability.
  PetscInt dim2 = wy->dim*wy->dim;
  for (PetscInt c=0;c<nc;c++) {
    PetscReal dSe_dS = 1.0/(1.0 - wy->Sr[c]);
    wy->dKr_dS[c] *= dSe_dS; // correct dKr/dS

    for(PetscInt j=0; j<dim2; j++) {
      wy->K[c*dim2+j] = wy->K0[c*dim2+j] * wy->Kr[c];
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyWYResidual(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {
  TDY_START_FUNCTION_TIMER()
  PetscFunctionBegin;
  PetscErrorCode ierr;
  TDy      tdy = (TDy)ctx;
  DM       dm;
  Vec      Ul;
  PetscInt c,cStart,cEnd,nv,gref,nf,f,fStart,fEnd,i,j,dim;
  PetscReal *p,*dp_dt,*r,wgt,sign,div;
  TDyWY *wy = tdy->context;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  nv   = wy->nfv;
  wgt  = 1/((PetscReal)nv);
  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U,Ul); CHKERRQ(ierr);
  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy,p,cEnd-cStart); CHKERRQ(ierr);
  ierr = TDyWYLocalElementCompute(tdy); CHKERRQ(ierr);
  ierr = TDyWYRecoverVelocity(tdy,Ul); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&fEnd); CHKERRQ(ierr);
    if (gref < 0) continue;

    /* compute the divergence */
    const PetscInt *faces;
    ierr = DMPlexGetConeSize(dm,c,&nf   ); CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&faces); CHKERRQ(ierr);
    div = 0;
    for(i=0; i<nf; i++) {
      f = faces[i];
      const PetscInt *verts;
      ierr = DMPlexGetConeSize(dm,f,&nv   ); CHKERRQ(ierr);
      ierr = DMPlexGetCone    (dm,f,&verts); CHKERRQ(ierr);
      sign = PetscSign(TDyADotBMinusC(&(wy->N[dim*f]),&(wy->X[dim*f]),
                                      &(wy->X[dim*c]),dim));
      for(j=0; j<nv; j++) {
        div += sign*wy->vel[nv*(f-fStart)+j]*wgt*wy->V[f];
      }
    }

    r[c] = wy->porosity[c-cStart]*wy->dS_dP[c-cStart]*dp_dt[c] + div - wy->Flocal[c-cStart];
    //PetscPrintf(PETSC_COMM_WORLD,"R[%2d] = %+e %+e %+e = %+e\n",
    // 	c,wy->porosity[c-cStart]*wy->dS_dP[c-cStart]*dp_dt[c],
    // 		div,wy->Flocal[c-cStart],r[c]);
  }

  /* Cleanup */
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

