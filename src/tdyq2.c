#include "../include/private/tdycoreimpl.h"
#include <petscblaslapack.h>

/*
u = -K \nabla p              (1)
\nabla\cdot u = f,           (2)

weak form:
(v,K_inv*u) - (\nabla\cdot v, p) = 0   (3)
-(w,\nabla\cdot u) + (w,f)       = 0   (4)
*/

/* (dim*vertices_per_cell+1)^2 */
#define MAX_LOCAL_SIZE 625

void PermWheeler2012_1(double *xx,double *K) {
  K[0] = 2; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
}

void PermWheeler2012_2(double *xx,double *K) {
  K[0] = 1;     K[1] = 1;     K[2] = 1;
  K[3] = 1;     K[4] = 2;     K[5] = 1;
  K[6] = 1;     K[7] = 1;     K[8] = 2;
}

void InvPermWheeler2012_1(double *xx,double *K_inv) {
  double K[4];
  double det; 
  PermWheeler2012_1(xx,K);
  det = K[0]*K[3] - K[1]*K[2];
  
  K_inv[0] =  K[3]/det; K_inv[1] = -K[1]/det;
  K_inv[2] = -K[2]/det; K_inv[3] =  K[0]/det;
}

void InvPermWheeler2012_2(double *xx,double *K_inv) {
  double K[9];
  double det; 
  PermWheeler2012_2(xx,K);
  det = (K[0]*K[4]*K[8]+K[1]*K[5]*K[6]+K[2]*K[3]*K[7]) - (K[0]*K[5]*K[7]+K[1]*K[3]*K[8]+K[2]*K[4]*K[6]);

  K_inv[0] =  (K[4]*K[8]-K[5]*K[7])/det;  K_inv[1] = -(K[1]*K[8]-K[2]*K[7])/det;  K_inv[2] =  (K[1]*K[5]-K[2]*K[4])/det;
  K_inv[3] = -(K[3]*K[8]-K[5]*K[6])/det;  K_inv[4] =  (K[0]*K[8]-K[2]*K[6])/det;  K_inv[5] = -(K[0]*K[5]-K[2]*K[3])/det;
  K_inv[6] =  (K[3]*K[7]-K[4]*K[6])/det;  K_inv[7] = -(K[0]*K[7]-K[1]*K[6])/det;  K_inv[8] =  (K[0]*K[4]-K[1]*K[3])/det;
}

void ForcingWheeler2012_1(double *xx,double *f) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*xx[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*xx[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*xx[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*xx[1]);
  double K[4];
  PermWheeler2012_1(xx,K);
  (*f)  = K[0]*sx*sx*sy*sy;
  (*f) -= K[0]*sy*sy*cx*cx;
  (*f) -= K[1]*(PetscCosReal(6*PETSC_PI*(xx[0]-xx[1]))-PetscCosReal(6*PETSC_PI*(xx[0]+xx[1])))*0.25;
  (*f) -= K[2]*(PetscCosReal(6*PETSC_PI*(xx[0]-xx[1]))-PetscCosReal(6*PETSC_PI*(xx[0]+xx[1])))*0.25;
  (*f) += K[3]*sx*sx*sy*sy;
  (*f) -= K[3]*sx*sx*cy*cy;
  (*f) *= 18*PETSC_PI*PETSC_PI;
}

void ForcingWheeler2012_2(double *xx,double *f) {
  PetscReal x2 = xx[0]*xx[0], y2 = xx[1]*xx[1], z2 = xx[2]*xx[2];
  PetscReal xm12 = PetscSqr(xx[0]-1);
  PetscReal ym12 = PetscSqr(xx[1]-1);
  PetscReal zm12 = PetscSqr(xx[2]-1);
  double K[9]; PermWheeler2012_2(xx,K);

  PetscReal a1 = 2*xx[0]*(xx[0]-1)*(2*xx[0]-1);
  PetscReal b1 = 2*xx[1]*(xx[1]-1)*(2*xx[1]-1);
  PetscReal c1 = 2*xx[2]*(xx[2]-1)*(2*xx[2]-1);

  PetscReal a2 = 12*x2 - 12*xx[0] + 2;
  PetscReal b2 = 12*y2 - 12*xx[1] + 2;
  PetscReal c2 = 12*z2 - 12*xx[2] + 2;

  (*f) =
    -K[0]*a2*y2*ym12*z2*zm12 -K[1]*a1     *b1*z2*zm12 -K[2]*a1     *y2*ym12*c1 +
    -K[3]*a1*b1     *z2*zm12 -K[4]*x2*xm12*b2*z2*zm12 -K[5]*x2*xm12*b1     *c1 +
    -K[6]*a1*y2*ym12*c1      -K[7]*x2*xm12*b1*c1      -K[8]*x2*xm12*y2*ym12*c2;
}
//========================================================
//         Compute f0 and f1 for the equation (3)
//========================================================

// (v,K_inv*u),
static void f0_u(
  PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
  //for (PetscInt d=0; d<dim; d++) {
  //  f0[d] = u[uOff[0] + d]; // TODO: coefficients
  //}

  // If we add K_inv
  if (dim == 2){
  double K_inv[4], xx;
  InvPermWheeler2012_1(&xx,K_inv);
  f0[0] = K_inv[0]*u[uOff[0] + 0] + K_inv[1]*u[uOff[0] + 1];
  f0[1] = K_inv[2]*u[uOff[0] + 0] + K_inv[3]*u[uOff[0] + 1];   
  }
  else{
  double xx, K_inv[9];
  InvPermWheeler2012_2(&xx,K_inv);
  f0[0] = K_inv[0]*u[uOff[0] + 0] + K_inv[1]*u[uOff[0] + 1] + K_inv[2]*u[uOff[0] + 2];
  f0[1] = K_inv[3]*u[uOff[0] + 0] + K_inv[4]*u[uOff[0] + 1] + K_inv[5]*u[uOff[0] + 2];
  f0[2] = K_inv[6]*u[uOff[0] + 0] + K_inv[7]*u[uOff[0] + 1] + K_inv[8]*u[uOff[0] + 2];
  }
}
// (\nabla\cdot v, p)
static void f1_u(
  PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
  for (PetscInt c=0; c<dim; c++) {
    for (PetscInt d=0; d<dim; d++) {
      f1[c*dim+d] = (c == d) ? u[uOff[1] + 0] : 0.;
    }
  }
}

//========================================================
//         Compute f0 and f1 for the equation (4)
//========================================================

// -(w,\nabla\cdot u) + (w,f)
static void f0_p(
  PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
  PetscScalar div = 0;
  for (PetscInt d=0; d<dim; d++){
    div -= u_x[d*dim + d];
  }
  
  if (dim == 2){
    double xx, f;
    ForcingWheeler2012_1(&xx,&f);
    f0[0] = div + f;
  }
  else {
    double xx, f;
    ForcingWheeler2012_2(&xx,&f);
    f0[0] = div + f;
  }
}

//========================================================
//      Compute Jacobian g0-g3 for the equation (3)
//========================================================

static void f0_u_J(
  PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[], 
  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
  /*
  for (PetscInt d=0; d<dim; d++) {
    g0[d] = 1.0; // TODO: coefficients
  }
  */
  //If we add K_inv 
  if (dim == 2){
  double K_inv[4], xx;
  InvPermWheeler2012_1(&xx,K_inv);
  g0[0] = K_inv[0] + K_inv[1];
  g0[1] = K_inv[2] + K_inv[3];   
  }
  else{
  double xx, K_inv[9];
  InvPermWheeler2012_2(&xx,K_inv);
  g0[0] = K_inv[0] + K_inv[1] + K_inv[2];
  g0[1] = K_inv[3] + K_inv[4] + K_inv[5];
  g0[2] = K_inv[6] + K_inv[7] + K_inv[8];
  }
  
}

static void f1_u_J(
  PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[], 
  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]) {
  for (PetscInt c=0; c<dim; c++) {
    for (PetscInt d=0; d<dim; d++) {
      g2[c*dim+d] = (c == d) ? 1 : 0.;
    }
  }
}

//========================================================
//      Compute Jacobian g0-g3 for the equation (4)
//========================================================

static void f0_p_J(
  PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[], 
  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
  for (PetscInt d=0; d<dim; d++) {
    g1[d*dim + d] = -1.0; // TODO: coefficients
  }
}

PetscErrorCode TDyQ2Initialize(TDy tdy) {

  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt dim;
  DM dm = tdy->dm;
  PetscFE fe[2];
  PetscDS ds;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, 3, PETSC_FALSE, 2, 2, &fe[0]);CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, 0, 2, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fe[0], "Velocity");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fe[1], "Pressure");CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe[0]);CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);

  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 1, f0_p, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, f0_u_J, NULL,  NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 1, NULL, NULL,  f1_u_J, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 1, 0, NULL, f0_p_J, NULL,  NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyQ2ComputeSystem(TDy tdy,Mat K,Vec F) {
  
  PetscFunctionBegin;
  SNES            snes;                 /* nonlinear solver */
  DM dm = tdy->dm;
  Vec             u,r;                  /* solution, residual vectors */
  PetscErrorCode  ierr;
  PetscInt        its;
  PetscReal res = 0.0;
  MPI_Comm       comm = PETSC_COMM_WORLD;

  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);

  ierr = SNESGetFunction(snes, &r, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Residual\n");CHKERRQ(ierr);
  ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscReal TDyQ2PressureNorm(TDy tdy,Vec U) {
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
PetscReal TDyQ2VelocityNorm(TDy tdy,Vec U) {
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

  PetscInt i,j,ncv,q,nlocal,nq,dd,nq1d=3;
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

  PetscReal cx[3],cDF[9],cDFinv[9],cJ[1];
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
          va = TDyADotB(vel,&(tdy->N[dim*f]),dim);

          /* exact value normal to this point/face */
          ierr = (*tdy->ops->computedirichletflux)(tdy,&(x[q*dim]),vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
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
