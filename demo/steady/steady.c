#include "tdycore.h"
#include "private/tdycoreimpl.h"

PetscReal alpha = 1;

void PermTest2D(double *x,double *K) {
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

void PermTest3D(double *x,double *K) {
  K[0] = 4.321; K[1] = 1;     K[2] = 0.5;
  K[3] = 1    ; K[4] = 1.234; K[5] = 1;
  K[6] = 0.5  ; K[7] = 1;     K[8] = 1.1;
}

/*--- -dim {2|3} -problem 1 ---------------------------------------------------------------*/

PetscErrorCode PressureConstant(TDy tdy,double *x,double *p,void *ctx) { (*p) = 3.14; PetscFunctionReturn(0);}
PetscErrorCode VelocityConstant(TDy tdy,double *x,double *v,void *ctx) { v[0] = 0; v[1] = 0; v[2] = 0; PetscFunctionReturn(0);}
PetscErrorCode ForcingConstant(TDy tdy,double *x,double *f,void *ctx) { (*f) = 0; PetscFunctionReturn(0);}

/*--- -dim 2 -problem 2 ---------------------------------------------------------------*/

PetscErrorCode PressureQuadratic2D(TDy tdy,double *x,double *p,void *ctx) { (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1]); PetscFunctionReturn(0);}
PetscErrorCode VelocityQuadratic2D(TDy tdy,double *x,double *v,void *ctx) {
  double K[4]; PermTest2D(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]);
  v[1] = -K[2]*(1-2*x[0]) - K[3]*(1-2*x[1]);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingQuadratic2D(TDy tdy,double *x,double *f,void *ctx) { double K[4]; PermTest2D(x,K); (*f) = 2*K[0]+2*K[3]; PetscFunctionReturn(0);}

PetscErrorCode PressureQuadratic3D(TDy tdy,double *x,double *p,void *ctx) { (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1])+x[2]*(1-x[2]); PetscFunctionReturn(0);}
PetscErrorCode VelocityQuadratic3D(TDy tdy,double *x,double *v,void *ctx) {
  double K[9]; PermTest3D(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]) - K[2]*(1-2*x[2]);
  v[1] = -K[3]*(1-2*x[0]) - K[4]*(1-2*x[1]) - K[5]*(1-2*x[2]);
  v[2] = -K[6]*(1-2*x[0]) - K[7]*(1-2*x[1]) - K[8]*(1-2*x[2]);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingQuadratic3D(TDy tdy,double *x,double *f,void *ctx) { double K[4]; PermTest3D(x,K); (*f) = 2*(K[0]+K[3]+K[8]); PetscFunctionReturn(0);}

/*--- -paper Wheeler2006 -------------------------------------------------------*/

/* -problem 1 */

PetscErrorCode PressureWheeler2006_1(TDy tdy,double *x,double *f,void *ctx) {
  (*f)  = PetscPowReal(1-x[0],4);
  (*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
  (*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
  PetscFunctionReturn(0);
}
PetscErrorCode VelocityWheeler2006_1(TDy tdy,double *x,double *v,void *ctx) {
  double vx,vy,K[4];
  PermTest2D(x,K);
  vx  = -4*PetscPowReal(1-x[0],3);
  vx += -PetscPowReal(1-x[1],3);
  vx += +PetscSinReal(x[1]-1)*PetscSinReal(x[0]-1);
  vy  = -3*PetscPowReal(1-x[1],2)*(1-x[0]);
  vy += -PetscCosReal(x[0]-1)*PetscCosReal(x[1]-1);
  v[0] = -(K[0]*vx+K[1]*vy);
  v[1] = -(K[2]*vx+K[3]*vy);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingWheeler2006_1(TDy tdy,double *x,double *f, void *ctx) {
  double K[4];
  PermTest2D(x,K);
  (*f)  = -K[0]*(12*PetscPowReal(1-x[0],
                                 2)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  (*f) += -K[1]*( 3*PetscPowReal(1-x[1],
                                 2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[2]*( 3*PetscPowReal(1-x[1],
                                 2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[3]*(-6*(1-x[0])*(x[1]-1)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  PetscFunctionReturn(0);
}

/* -problem 2 */

void PermWheeler2006_2(double *x,double *K) {
  K[0] = 4 + PetscSqr(x[0]+2) + PetscSqr(x[1]);
  K[1] = 1 + PetscSinReal(x[0]*x[1]);
  K[2] = K[1];
  K[3] = 2;
}
PetscErrorCode PressureWheeler2006_2(TDy tdy,double *x,double *f,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  (*f) = sx*sx*sy*sy;
  PetscFunctionReturn(0);
}
PetscErrorCode VelocityWheeler2006_2(TDy tdy,double *x,double *v,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*x[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*x[1]);
  v[0] = -6*PETSC_PI*((sin(x[0]*x[1]) + 1)*sx*cy + (x[1]*x[1] + PetscSqr(x[0]+2) + 4)*sy*cx)*sx*sy;
  v[1] = -6*PETSC_PI*((sin(x[0]*x[1]) + 1)*sy*cx + 2*sx*cy)*sx*sy;
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingWheeler2006_2(TDy tdy,double *x,double *f, void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*x[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*x[1]);
  (*f)  = -12*PETSC_PI*(PetscSinReal(x[0]*x[1]) + 1)*sx*sy*cx*cy;
  (*f) +=  3*PETSC_PI*(x[1]*x[1] + PetscSqr(x[0] + 2) + 4)*sx*sx*sy*sy;
  (*f) -=  3*PETSC_PI*(x[1]*x[1] + PetscSqr(x[0] + 2) + 4)*sy*sy*cx*cx;
  (*f) +=  6*PETSC_PI*sx*sx*sy*sy;
  (*f) -=  6*PETSC_PI*sx*sx*cy*cy;
  (*f) -=  x[0]*sx*sy*sy*cx*PetscCosReal(x[0]*x[1]);
  (*f) -=  x[1]*sx*sx*sy*cy*PetscCosReal(x[0]*x[1]);
  (*f) -=  2*(x[0] + 2)*sx*sy*sy*cx;
  (*f) *= 6*PETSC_PI;
  PetscFunctionReturn(0);
}


/*--- -paper Wheeler2012 -------------------------------------------------------*/

/* -problem 1 */

void PermWheeler2012_1(double *x,double *K) {
  K[0] = 2   ; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
}

PetscErrorCode PressureWheeler2012_1(TDy tdy,double *x,double *f,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  (*f) = sx*sx*sy*sy;
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityWheeler2012_1(TDy tdy,double *x,double *v,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*x[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*x[1]);
  double px,py,K[4];
  PermWheeler2012_1(x,K);
  px = 6*PETSC_PI*sx*sy*sy*cx;
  py = 6*PETSC_PI*sx*sx*sy*cy;
  v[0] = -(K[0]*px+K[1]*py);
  v[1] = -(K[2]*px+K[3]*py);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingWheeler2012_1(TDy tdy,double *x,double *f, void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*x[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*x[1]);
  double K[4];
  PermWheeler2012_1(x,K);
  (*f)  = K[0]*sx*sx*sy*sy;
  (*f) -= K[0]*sy*sy*cx*cx;
  (*f) -= K[1]*(PetscCosReal(6*PETSC_PI*(x[0]-x[1]))-PetscCosReal(6*PETSC_PI*(x[0]+x[1])))*0.25;
  (*f) -= K[2]*(PetscCosReal(6*PETSC_PI*(x[0]-x[1]))-PetscCosReal(6*PETSC_PI*(x[0]+x[1])))*0.25;
  (*f) += K[3]*sx*sx*sy*sy;
  (*f) -= K[3]*sx*sx*cy*cy;
  (*f) *= 18*PETSC_PI*PETSC_PI;
  PetscFunctionReturn(0);
}

/* -problem 2 */

void PermWheeler2012_2(double *x,double *K) {
  K[0] = alpha; K[1] = 1; K[2] = 1;
  K[3] = 1    ; K[4] = 2; K[5] = 1;
  K[6] = 1    ; K[7] = 1; K[8] = 2;
}

PetscErrorCode PressureWheeler2012_2(TDy tdy,double *x,double *f,void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  (*f) = x2*y2*z2*xm12*ym12*zm12;
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityWheeler2012_2(TDy tdy,double *x,double *v,void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  PetscReal px = (x2*y2*z2*(2*x[0] - 2)*ym12*zm12 + 2*x[0]*y2*z2*xm12*ym12*zm12);
  PetscReal py = (x2*y2*z2*xm12*(2*x[1] - 2)*zm12 + 2*x2*x[1]*z2*xm12*ym12*zm12);
  PetscReal pz = (x2*y2*z2*xm12*ym12*(2*x[2] - 2) + 2*x2*y2*x[2]*xm12*ym12*zm12);
  double K[9]; PermWheeler2012_2(x,K);
  v[0] = -K[0]*px - K[1]*py - K[2]*pz;
  v[1] = -K[3]*px - K[4]*py - K[5]*pz;
  v[2] = -K[6]*px - K[7]*py - K[8]*pz;
  PetscFunctionReturn(0);
}

PetscErrorCode ForcingWheeler2012_2(TDy tdy,double *x,double *f, void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm1  = (x[0]-1);
  PetscReal ym1  = (x[1]-1);
  PetscReal zm1  = (x[2]-1);
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  double K[9]; PermWheeler2012_2(x,K);
  (*f) = -2*K[0]*y2*z2*ym12*zm12*(x2 + 4*x[0]*xm1 + xm12) -
         4*K[1]*x[0]*x[1]*z2*xm1*ym1*zm12*(x[0]*x[1] + x[0]*ym1 + x[1]*xm1 + xm1*ym1) -
         4*K[2]*x[0]*y2*x[2]*xm1*ym12*zm1*(x[0]*x[2] + x[0]*zm1 + x[2]*xm1 + xm1*zm1) -
         4*K[3]*x[0]*x[1]*z2*xm1*ym1*zm12*(x[0]*x[1] + x[0]*ym1 + x[1]*xm1 + xm1*ym1) -
         2*K[4]*x2*z2*xm12*zm12*(y2 + 4*x[1]*ym1 + ym12) -
         4*K[5]*x2*x[1]*x[2]*xm12*ym1*zm1*(x[1]*x[2] + x[1]*zm1 + x[2]*ym1 + ym1*zm1) -
         4*K[6]*x[0]*y2*x[2]*xm1*ym12*zm1*(x[0]*x[2] + x[0]*zm1 + x[2]*xm1 + xm1*zm1) -
         4*K[7]*x2*x[1]*x[2]*xm12*ym1*zm1*(x[1]*x[2] + x[1]*zm1 + x[2]*ym1 + ym1*zm1) -
         2*K[8]*x2*y2*xm12*ym12*(z2 + 4*x[2]*zm1 + zm12);
  PetscFunctionReturn(0);
}

/*--- -dim 3 -problem 3 ---------------------------------------------------------------*/

PetscErrorCode Pressure3(TDy tdy,double *x,double *f,void *ctx) {
  (*f) = PetscCosReal(x[0])*PetscCosReal(x[1])*PetscCosReal(x[2]);
  PetscFunctionReturn(0);
}

PetscErrorCode Velocity3(TDy tdy,double *x,double *v,void *ctx) {
  double K[9]; PermWheeler2012_2(x,K);
  v[0]  =  K[0]*PetscSinReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
             x[2]) + K[1]*PetscSinReal(x[1])*PetscCosReal(x[0])*PetscCosReal(
             x[2]) + K[2]*PetscSinReal(x[2])*PetscCosReal(x[0])*PetscCosReal(x[1]);
  v[1]  =  K[3]*PetscSinReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
             x[2]) + K[4]*PetscSinReal(x[1])*PetscCosReal(x[0])*PetscCosReal(
             x[2]) + K[5]*PetscSinReal(x[2])*PetscCosReal(x[0])*PetscCosReal(x[1]);
  v[2]  =  K[6]*PetscSinReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
             x[2]) + K[7]*PetscSinReal(x[1])*PetscCosReal(x[0])*PetscCosReal(
             x[2]) + K[8]*PetscSinReal(x[2])*PetscCosReal(x[0])*PetscCosReal(x[1]);
  PetscFunctionReturn(0);
}

PetscErrorCode Forcing3(TDy tdy,double *x,double *f,void *ctx) {
  double K[9]; PermWheeler2012_2(x,K);
  (*f) = K[0]*PetscCosReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
           x[2]) - K[1]*PetscSinReal(x[0])*PetscSinReal(x[1])*PetscCosReal(
           x[2]) - K[2]*PetscSinReal(x[0])*PetscSinReal(x[2])*PetscCosReal(
           x[1]) - K[3]*PetscSinReal(x[0])*PetscSinReal(x[1])*PetscCosReal(
           x[2]) + K[4]*PetscCosReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
           x[2]) - K[5]*PetscSinReal(x[1])*PetscSinReal(x[2])*PetscCosReal(
           x[0]) - K[6]*PetscSinReal(x[0])*PetscSinReal(x[2])*PetscCosReal(
           x[1]) - K[7]*PetscSinReal(x[1])*PetscSinReal(x[2])*PetscCosReal(
           x[0]) + K[8]*PetscCosReal(x[0])*PetscCosReal(x[1])*PetscCosReal(x[2]);
  PetscFunctionReturn(0);
}


PetscErrorCode PerturbVerticesRandom(DM dm,PetscReal h) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value,dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  /* this is the 'marker' label which marks boundary entities */
  ierr = DMGetLabelByNum(dm,2,&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value); CHKERRQ(ierr);
    if(dim==2) {
      if(value==-1) {
        /* perturb randomly O(h*sqrt(2)/3) */
        PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*(h*0.471404);
        PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
        coords[offset  ] += r*PetscCosReal(t);
        coords[offset+1] += r*PetscSinReal(t);
      }
    } else {
      /* This is because 'marker' is broken in 3D */
      if(coords[offset] > 0 && coords[offset] < 1 &&
          coords[offset+1] > 0 && coords[offset+1] < 1 &&
          coords[offset+2] > 0 && coords[offset+2] < 1) {
        coords[offset+2] += (((PetscReal)rand())/((PetscReal)RAND_MAX)-0.5)*h*0.1; // <-- not removing yet but notice 1/10
      }
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode PerturbVerticesSmooth(DM dm) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,dim;
  PetscReal    x,y,z;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMGetLabelByNum(dm,2,&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    if(dim==2) {
      x = coords[offset]; y = coords[offset+1];
      coords[offset]   = x + 0.03*PetscSinReal(3*PETSC_PI*x)*PetscCosReal(3*PETSC_PI*y);
      coords[offset+1] = y - 0.04*PetscCosReal(3*PETSC_PI*x)*PetscSinReal(3*PETSC_PI*y);
    } else {
      x = coords[offset]; y = coords[offset+1]; z = coords[offset+2];
      coords[offset]   = x + 0.03*PetscSinReal(3*PETSC_PI*x)*PetscCosReal(3*PETSC_PI*y)*PetscCosReal(3*PETSC_PI*z);
      coords[offset+1] = y - 0.04*PetscCosReal(3*PETSC_PI*x)*PetscSinReal(3*PETSC_PI*y)*PetscCosReal(3*PETSC_PI*z);
      coords[offset+2] = z + 0.05*PetscCosReal(3*PETSC_PI*x)*PetscCosReal(3*PETSC_PI*y)*PetscSinReal(3*PETSC_PI*z);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  For debugging purposes, it can be helpful to locate sources of
  bugs/errors by looking at the error in applying the operator. To do
  this, we compute what we know the exact solution should be, and then
  apply the operator to this vector. If everything is implemented
  correctly, then K*U-F = 0. If not, then large values will indicate
  what parts of the operator are incorrectly formed.

  On entry:
  - tdy, the TDy context
  - U, a Global PETSc Vector, contents do not matter
  - K, the system matrix computed from TDyComputeSystem
  - f, the function pointer to the exact pressure solution
  - R, the system right-hand side computed from TDyComputeSystem

  On exit:
  - U, the function f projected onto the mesh using 3^d point quadrature
  - R, the residual formed by taking R=abs(K*U-R)

 */
PetscErrorCode OperatorApplicationResidual(TDy tdy,Vec U,Mat K,PetscErrorCode (*f)(TDy,PetscReal*,PetscReal*,void*),Vec R){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt dim,c,cStart,cEnd,q,nq1d=3,nq=27;
  PetscQuadrature quadrature;
  PetscReal x[81],J[27],DF[243],DFinv[243],value,mean,volume;
  const PetscScalar *quad_x,*quad_w;
  DM dm;
  ierr = TDyGetDM(tdy,&dm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscDTGaussTensorQuadrature(dim,1,nq1d,-1,+1,&quadrature); CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&nq,&quad_x,&quad_w); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    mean = 0; volume = 0;
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quadrature,x,DF,DFinv,J); CHKERRQ(ierr);
    for(q=0;q<nq;q++){
      (*f)(NULL,&(x[q*dim]),&value,NULL);
      mean   += value*quad_w[q]*J[q];
      volume +=       quad_w[q]*J[q];
    }
    ierr = VecSetValue(U,c,mean/volume,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(U); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(U); CHKERRQ(ierr);
  ierr = VecScale(R,-1); CHKERRQ(ierr);
  ierr = MatMultAdd(K,U,R,R); CHKERRQ(ierr);
  ierr = VecAbs(R); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadrature); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2, problem = 2;
  PetscInt successful_exit_code=0;
  PetscBool perturb = PETSC_FALSE;
  char exofile[256];
  PetscBool exo = PETSC_FALSE;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",dim,&dim,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N","Number of elements in 1D","",N,&N,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-problem","Problem number","",problem,&problem,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool("-perturb","Perturb interior vertices","",perturb,
                          &perturb,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Permeability scaling","",alpha,&alpha,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsString("-exo","Mesh file in exodus format","",
			    exofile,exofile,256,&exo); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  DMLabel marker;
  if(exo){
    ierr = DMPlexCreateExodusFromFile(PETSC_COMM_WORLD,exofile,
				      PETSC_TRUE,&dm); CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = DMCreateLabel(dm,"marker"); CHKERRQ(ierr);
    ierr = DMGetLabel(dm,"marker",&marker); CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm,1,marker); CHKERRQ(ierr);
  }else{
    const PetscInt  faces[3] = {N,N,N  };
    const PetscReal lower[3] = {0.0,0.0,0.0};
    const PetscReal upper[3] = {1.0,1.0,1.0};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
			       NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
    if(perturb) {
      //ierr = PerturbVerticesSmooth(dm); CHKERRQ(ierr);
      ierr = PerturbVerticesRandom(dm,1./N); CHKERRQ(ierr);
    }
  }
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);


  switch(problem) {
      case 1:
      if (dim == 2) {
        ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
      } else {
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
      }
      ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFluxFunction(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
      break;

      case 2:

      if (dim == 2) {
        ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingQuadratic2D,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureQuadratic2D,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityQuadratic2D,NULL); CHKERRQ(ierr);
      } else {
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2012_2,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2012_2,NULL); CHKERRQ(ierr);
      }
      break;

      case 3:
      if (dim == 2) {
        ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2006_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2006_1,NULL); CHKERRQ(ierr);
      } else {
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,Forcing3,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,Pressure3,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,Velocity3,NULL); CHKERRQ(ierr);
      }
      break;

      case 4:
      if (dim == 2) {
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2012_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2012_1,NULL); CHKERRQ(ierr);
      } else {
      
      }
      break;
  }

  ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Compute system */
  Mat K;
  Vec U,Ue,F;
  ierr = DMCreateGlobalVector(dm,&U ); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&Ue); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F ); CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K ); CHKERRQ(ierr);
  ierr = TDyComputeSystem(tdy,K,F); CHKERRQ(ierr);

  /* Solve system */
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);
  
  /* Output solution */
  PetscViewer viewer;
  PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),"sol.vtk",FILE_MODE_WRITE,&viewer);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer); CHKERRQ(ierr); // the approximate solution
  ierr = OperatorApplicationResidual(tdy,Ue,K,tdy->ops->computedirichletvalue,F); 
  ierr = VecView(F,viewer); CHKERRQ(ierr); // the residual K*Ue-F
  ierr = VecView(Ue,viewer); CHKERRQ(ierr);  // the exact solution
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  /* Evaluate error norms */
  PetscReal normp,normv;
  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%e %e\n",normp,normv); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U);

  /* Cleanup */
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&Ue); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
