//#include "../include/private/tdycoreimpl.h"
#include <private/tdycoreimpl.h>
#include <petscblaslapack.h>
#include <petscviewerhdf5.h>

/*
u = -K \nabla p              (1)
\nabla\cdot u = f,           (2)

weak form:
(v,K_inv*u) - (\nabla\cdot v, p)  = 0   (3)
-(w,\nabla\cdot u) +  (w,f)       = 0   (4)
*/

/* (dim*vertices_per_cell+1)^2 */
#define MAX_LOCAL_SIZE 625

PetscReal alpha = 1;

PetscErrorCode PermWheeler2012_1(const PetscReal x[], PetscScalar *K) {
  K[0] = 2; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
  PetscFunctionReturn(0);
}

PetscErrorCode PermWheeler2012_2(const PetscReal x[], PetscScalar *K) {
  K[0] = alpha;     K[1] = 1;     K[2] = 1;
  K[3] = 1;         K[4] = 2;     K[5] = 1;
  K[6] = 1;         K[7] = 1;     K[8] = 2;
  PetscFunctionReturn(0);
}

PetscErrorCode InvPermWheeler2012_1(const PetscReal x[], PetscScalar *K_inv) {
  PetscScalar K[4];
  PetscScalar det; 
  PermWheeler2012_1(x,K);
  det = K[0]*K[3] - K[1]*K[2];
  
  K_inv[0] =  K[3]/det; K_inv[1] = -K[1]/det;
  K_inv[2] = -K[2]/det; K_inv[3] =  K[0]/det;
  PetscFunctionReturn(0);
}

PetscErrorCode InvPermWheeler2012_2(const PetscReal x[],PetscScalar *K_inv) {
  PetscScalar K[9];
  PetscScalar det; 
  PermWheeler2012_2(x,K);
  det = (K[0]*K[4]*K[8]+K[1]*K[5]*K[6]+K[2]*K[3]*K[7]) - (K[0]*K[5]*K[7]+K[1]*K[3]*K[8]+K[2]*K[4]*K[6]);

  K_inv[0] =  (K[4]*K[8]-K[5]*K[7])/det;  K_inv[1] = -(K[1]*K[8]-K[2]*K[7])/det;  K_inv[2] =  (K[1]*K[5]-K[2]*K[4])/det;
  K_inv[3] = -(K[3]*K[8]-K[5]*K[6])/det;  K_inv[4] =  (K[0]*K[8]-K[2]*K[6])/det;  K_inv[5] = -(K[0]*K[5]-K[2]*K[3])/det;
  K_inv[6] =  (K[3]*K[7]-K[4]*K[6])/det;  K_inv[7] = -(K[0]*K[7]-K[1]*K[6])/det;  K_inv[8] =  (K[0]*K[4]-K[1]*K[3])/det;
  PetscFunctionReturn(0);
}

PetscErrorCode ForcingWheeler2012_1(const PetscReal x[],PetscScalar *f) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*x[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*x[1]);
  PetscScalar K[4]; PermWheeler2012_1(x,K);

  (*f)  = K[0]*sx*sx*sy*sy;
  (*f) -= K[0]*sy*sy*cx*cx;
  (*f) -= K[1]*(PetscCosReal(6*PETSC_PI*(x[0]-x[1]))-PetscCosReal(6*PETSC_PI*(x[0]+x[1])))*0.25;
  (*f) -= K[2]*(PetscCosReal(6*PETSC_PI*(x[0]-x[1]))-PetscCosReal(6*PETSC_PI*(x[0]+x[1])))*0.25;
  (*f) += K[3]*sx*sx*sy*sy;
  (*f) -= K[3]*sx*sx*cy*cy;
  (*f) *= 18*PETSC_PI*PETSC_PI;
  PetscFunctionReturn(0);
}


PetscErrorCode ForcingWheeler2012_2(const PetscReal x[],PetscScalar *f) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  PetscScalar K[9]; PermWheeler2012_2(x,K);

  PetscReal a1 = 2*x[0]*(x[0]-1)*(2*x[0]-1);
  PetscReal b1 = 2*x[1]*(x[1]-1)*(2*x[1]-1);
  PetscReal c1 = 2*x[2]*(x[2]-1)*(2*x[2]-1);

  PetscReal a2 = 12*x2 - 12*x[0] + 2;
  PetscReal b2 = 12*y2 - 12*x[1] + 2;
  PetscReal c2 = 12*z2 - 12*x[2] + 2;

  *f =
    -K[0]*a2*y2*ym12*z2*zm12 -K[1]*a1     *b1*z2*zm12 -K[2]*a1     *y2*ym12*c1 +
    -K[3]*a1*b1     *z2*zm12 -K[4]*x2*xm12*b2*z2*zm12 -K[5]*x2*xm12*b1     *c1 +
    -K[6]*a1*y2*ym12*c1      -K[7]*x2*xm12*b1*c1      -K[8]*x2*xm12*y2*ym12*c2;
  PetscFunctionReturn(0);
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
  /*
  for (PetscInt d=0; d<dim; d++) {
    f0[d] = u[uOff[0] + d]; // TODO: coefficients
  }
  */
  // If we add K_inv
  if (dim == 2){
  PetscScalar K_inv[4];
  InvPermWheeler2012_1(x,K_inv);
  f0[0] = K_inv[0]*u[uOff[0] + 0] + K_inv[1]*u[uOff[0] + 1];
  f0[1] = K_inv[2]*u[uOff[0] + 0] + K_inv[3]*u[uOff[0] + 1];   
  }
  else{
  PetscScalar K_inv[9];
  InvPermWheeler2012_2(x,K_inv);
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
      f1[c*dim+d] = (c == d) ? -u[uOff[1] + 0] : 0.;
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
    PetscScalar f;
    ForcingWheeler2012_1(x,&f);
    f0[0] = div + f;
  }
  else {
    PetscScalar f;
    ForcingWheeler2012_2(x,&f);
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
  PetscScalar K_inv[4];
  InvPermWheeler2012_1(x,K_inv);
  g0[0] = K_inv[0] + K_inv[1];
  g0[1] = K_inv[2] + K_inv[3];   
  }
  else{
  PetscScalar K_inv[9];
  InvPermWheeler2012_2(x,K_inv);
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
      g2[c*dim+d] = (c == d) ? -1. : 0.;
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
    g1[d*dim + d] = -1.0; 
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
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_FALSE, 2, 2, &fe[0]);CHKERRQ(ierr);
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

static PetscErrorCode TDyQ2ApplyResidual(DM dm, Vec U, Vec F, void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeResidualFEM(dm, U, F, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void PermTest2D(const double *x,double *K) {
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}
PetscErrorCode func_p(PetscInt dim, PetscReal time, const PetscReal x[0], PetscInt Nf, PetscScalar *p, void *ctx) { *p = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1]); return 0;}
PetscErrorCode func_u(PetscInt dim, PetscReal time, const PetscReal x[0], PetscInt Nf, PetscScalar *v, void *ctx) {
  double K[4]; PermTest2D(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]);
  v[1] = -K[2]*(1-2*x[0]) - K[3]*(1-2*x[1]);
  return 0;
}

PetscErrorCode TDyQ2ComputeSystem(TDy tdy,Mat K,Vec F) {
  DM dm = tdy->dm;
  Vec             u;                  /* solution, residual vectors */
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  ierr = TDyQ2ApplyResidual(dm, u, F, NULL);CHKERRQ(ierr);

  ISColoring iscoloring;
  MatFDColoring color;
  MatColoring mc;

  ierr = MatColoringCreate(K,&mc);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(mc,2);CHKERRQ(ierr);
  ierr = MatColoringSetType(mc,MATCOLORINGSL);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
  ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(K,iscoloring,&color);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))TDyQ2ApplyResidual,NULL);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(K,iscoloring,color);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringApply(K,color,u,dm);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);

  ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&color);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  //ierr = MatView(K,NULL);CHKERRQ(ierr);

  if (1) { // Debug: project exact solution (cribbed from demo/steady/steady
    PetscViewer viewer;
    Vec U, R;
    PetscErrorCode (*funcs[2])() = {func_u, func_p};
    ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0., funcs, NULL, ADD_VALUES, U);CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject)dm),"project.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)U,"Solution");CHKERRQ(ierr);
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    ierr = VecView(U,viewer);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(dm,&R);CHKERRQ(ierr);
    ierr = MatMult(K, U, R);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)R,"Residual");CHKERRQ(ierr);
    ierr = VecView(R,viewer);CHKERRQ(ierr);

    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&U);CHKERRQ(ierr);
    ierr = VecDestroy(&R);CHKERRQ(ierr);
  }
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
static void bd_integral(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *uint)
{
  PetscReal sum = 0;
  for (PetscInt d=0; d<dim; d++) {
    sum += u[uOff[0] + d]*n[d];
  }
  uint[0] = sum;
}

PetscReal TDyQ2VelocityNorm(TDy tdy,Vec U) { 
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt fStart,fEnd,dim;
  DM dm = tdy->dm;
  
  if(!(tdy->ops->computedirichletflux)) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the velocity function with TDySetDirichletFluxFunction");
  }
  
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);

    DMLabel     boundary;
    PetscScalar norm = 0.0;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "boundary", &boundary);CHKERRQ(ierr);
    ierr = DMLabelCreateIndex(boundary,fStart,fEnd);CHKERRQ(ierr);
    //Need to add scaling factor |E|/|e|
    ierr = DMPlexComputeBdIntegral(dm, U, boundary, PETSC_DETERMINE, PETSC_NULL, bd_integral, &norm, NULL);CHKERRQ(ierr);

    ierr = DMLabelDestroy(&boundary);CHKERRQ(ierr);

  PetscFunctionReturn(norm);
}


