#include "tdycore.h"
#include "private/tdycoreimpl.h"

PetscReal alpha = 1;

void PermTest2D(PetscReal *x,PetscReal *K) {
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

void PermTest3D(PetscReal *x,PetscReal *K) {
  K[0] = 4.321; K[1] = 1;     K[2] = 0.5;
  K[3] = 1    ; K[4] = 1.234; K[5] = 1;
  K[6] = 0.5  ; K[7] = 1;     K[8] = 1.1;
}

/*--- -dim {2|3} -problem 1 ---------------------------------------------------------------*/

PetscErrorCode PressureConstant(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) { (*p) = 3.14; PetscFunctionReturn(0);}
PetscErrorCode VelocityConstant(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) { v[0] = 0; v[1] = 0; v[2] = 0; PetscFunctionReturn(0);}
PetscErrorCode ForcingConstant(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) { (*f) = 0; PetscFunctionReturn(0);}

/*--- -dim 2 -problem 2 ---------------------------------------------------------------*/

PetscErrorCode PressureQuadratic2D(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) { (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1]); PetscFunctionReturn(0);}
PetscErrorCode VelocityQuadratic2D(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
  double K[4]; PermTest2D(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]);
  v[1] = -K[2]*(1-2*x[0]) - K[3]*(1-2*x[1]);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingQuadratic2D(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) { double K[4]; PermTest2D(x,K); (*f) = 2*K[0]+2*K[3]; PetscFunctionReturn(0);}

PetscErrorCode PressureQuadratic3D(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) { (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1])+x[2]*(1-x[2]); PetscFunctionReturn(0);}
PetscErrorCode VelocityQuadratic3D(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
  double K[9]; PermTest3D(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]) - K[2]*(1-2*x[2]);
  v[1] = -K[3]*(1-2*x[0]) - K[4]*(1-2*x[1]) - K[5]*(1-2*x[2]);
  v[2] = -K[6]*(1-2*x[0]) - K[7]*(1-2*x[1]) - K[8]*(1-2*x[2]);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingQuadratic3D(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) { double K[4]; PermTest3D(x,K); (*f) = 2*(K[0]+K[3]+K[8]); PetscFunctionReturn(0);}

/*--- -paper Wheeler2006 -------------------------------------------------------*/

/* -problem 1 */

void PermWheeler2006_1(PetscReal *x,PetscReal *K) {
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

PetscErrorCode PressureWheeler2006_1(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  (*f)  = PetscPowReal(1-x[0],4);
  (*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
  (*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
  PetscFunctionReturn(0);
}
PetscErrorCode VelocityWheeler2006_1(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
  double vx,vy,K[4];
  PermWheeler2006_1(x,K);
  vx  = -4*PetscPowReal(1-x[0],3);
  vx += -PetscPowReal(1-x[1],3);
  vx += +PetscSinReal(x[1]-1)*PetscSinReal(x[0]-1);
  vy  = -3*PetscPowReal(1-x[1],2)*(1-x[0]);
  vy += -PetscCosReal(x[0]-1)*PetscCosReal(x[1]-1);
  v[0] = -(K[0]*vx+K[1]*vy);
  v[1] = -(K[2]*vx+K[3]*vy);
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingWheeler2006_1(TDy tdy,PetscReal *x,PetscReal *f, void *ctx) {
  double K[4];
  PermWheeler2006_1(x,K);
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

void PermWheeler2006_2(PetscReal *x,PetscReal *K) {
  K[0] = 4 + PetscSqr(x[0]+2) + PetscSqr(x[1]);
  K[1] = 1 + PetscSinReal(x[0]*x[1]);
  K[2] = K[1];
  K[3] = 2;
}
PetscErrorCode PressureWheeler2006_2(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  (*f) = sx*sx*sy*sy;
  PetscFunctionReturn(0);
}
PetscErrorCode VelocityWheeler2006_2(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  PetscReal cx = PetscCosReal(3*PETSC_PI*x[0]);
  PetscReal cy = PetscCosReal(3*PETSC_PI*x[1]);
  v[0] = -6*PETSC_PI*((sin(x[0]*x[1]) + 1)*sx*cy + (x[1]*x[1] + PetscSqr(x[0]+2) + 4)*sy*cx)*sx*sy;
  v[1] = -6*PETSC_PI*((sin(x[0]*x[1]) + 1)*sy*cx + 2*sx*cy)*sx*sy;
  PetscFunctionReturn(0);
}
PetscErrorCode ForcingWheeler2006_2(TDy tdy,PetscReal *x,PetscReal *f, void *ctx) {
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

void PermWheeler2012_1(PetscReal *x,PetscReal *K) {
  K[0] = 2   ; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
}

PetscErrorCode PressureWheeler2012_1(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  PetscReal sx = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal sy = PetscSinReal(3*PETSC_PI*x[1]);
  (*f) = sx*sx*sy*sy;
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityWheeler2012_1(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
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
PetscErrorCode ForcingWheeler2012_1(TDy tdy,PetscReal *x,PetscReal *f, void *ctx) {
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

void PermWheeler2012_2(PetscReal *x,PetscReal *K) {
  K[0] = alpha; K[1] = 1; K[2] = 1;
  K[3] = 1    ; K[4] = 2; K[5] = 1;
  K[6] = 1    ; K[7] = 1; K[8] = 2;
}

PetscErrorCode PressureWheeler2012_2(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  (*f) = x2*y2*z2*xm12*ym12*zm12;
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityWheeler2012_2(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);

  PetscReal a1 = 2*x[0]*(x[0]-1)*(2*x[0]-1);
  PetscReal b1 = 2*x[1]*(x[1]-1)*(2*x[1]-1);
  PetscReal c1 = 2*x[2]*(x[2]-1)*(2*x[2]-1);

  PetscReal px = a1     *y2*ym12 *z2*zm12;
  PetscReal py = x2*xm12*b1      *z2*zm12;
  PetscReal pz = x2*xm12*y2*ym12*c1     ;

  double K[9]; PermWheeler2012_2(x,K);
  v[0] = -K[0]*px - K[1]*py - K[2]*pz;
  v[1] = -K[3]*px - K[4]*py - K[5]*pz;
  v[2] = -K[6]*px - K[7]*py - K[8]*pz;
  PetscFunctionReturn(0);
}

PetscErrorCode ForcingWheeler2012_2(TDy tdy,PetscReal *x,PetscReal *f, void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  double K[9]; PermWheeler2012_2(x,K);

  PetscReal a1 = 2*x[0]*(x[0]-1)*(2*x[0]-1);
  PetscReal b1 = 2*x[1]*(x[1]-1)*(2*x[1]-1);
  PetscReal c1 = 2*x[2]*(x[2]-1)*(2*x[2]-1);

  PetscReal a2 = 12*x2 - 12*x[0] + 2;
  PetscReal b2 = 12*y2 - 12*x[1] + 2;
  PetscReal c2 = 12*z2 - 12*x[2] + 2;

  (*f) =
    -K[0]*a2*y2*ym12*z2*zm12 -K[1]*a1     *b1*z2*zm12 -K[2]*a1     *y2*ym12*c1 +
    -K[3]*a1*b1     *z2*zm12 -K[4]*x2*xm12*b2*z2*zm12 -K[5]*x2*xm12*b1     *c1 +
    -K[6]*a1*y2*ym12*c1      -K[7]*x2*xm12*b1*c1      -K[8]*x2*xm12*y2*ym12*c2;

  PetscFunctionReturn(0);
}

/*--- -dim 3 -problem 3 ---------------------------------------------------------------*/

PetscErrorCode Pressure3(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  (*f) = PetscCosReal(x[0])*PetscCosReal(x[1])*PetscCosReal(x[2]);
  PetscFunctionReturn(0);
}

PetscErrorCode Velocity3(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
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

PetscErrorCode Forcing3(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
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
  ierr = DMGetLabel(dm, "boundary", &label); CHKERRQ(ierr);
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
      coords[offset]   = x + 0.06*PetscSinReal(2.0*PETSC_PI*x)*PetscSinReal(2.0*PETSC_PI*y);
      coords[offset+1] = y - 0.05*PetscSinReal(2.0*PETSC_PI*x)*PetscSinReal(2.0*PETSC_PI*y);
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

PetscErrorCode SaveVertices(DM dm, char filename[256]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  Vec coordinates;
  PetscViewer viewer;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = VecView(coordinates,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SaveCentroids(DM dm, char filename[256]){
  PetscInt d,dim,c,cStart,cEnd;
  PetscReal cen[3];
  PetscScalar *centroids_p;
  Vec centroids;
  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,(cEnd-cStart)*dim,PETSC_DETERMINE,&centroids); CHKERRQ(ierr);

  ierr = VecGetArray(centroids,&centroids_p); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexComputeCellGeometryFVM(dm, c, PETSC_NULL, &cen[0],
                                        PETSC_NULL); CHKERRQ(ierr);
    for (d=0;d<dim;d++) centroids_p[c*dim+d] = cen[d];
  }
  ierr = VecRestoreArray(centroids,&centroids_p); CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = VecView(centroids,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SaveTrueSolution(TDy tdy, char filename[256]){
  DM dm = tdy->dm;
  Vec coordinates,pressure;
  PetscScalar *coords,*pres;
  PetscInt c,cStart,cEnd;
  PetscReal centroid[3];
  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&pressure); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  ierr = VecGetArray(pressure,&pres); CHKERRQ(ierr);

  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexComputeCellGeometryFVM(dm, c, PETSC_NULL, &centroid[0],
                                        PETSC_NULL); CHKERRQ(ierr);
    ierr = tdy->ops->compute_boundary_pressure(tdy,&centroid[0],&pres[c],NULL);
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  ierr = VecRestoreArray(pressure,&pres); CHKERRQ(ierr);
  ierr = VecView(pressure,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = VecDestroy(&pressure); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SaveForcing(TDy tdy, char filename[256]){
  DM dm = tdy->dm;
  Vec forcing;
  PetscScalar *f;
  PetscInt c,cStart,cEnd;
  PetscReal centroid[3];
  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&forcing); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = VecGetArray(forcing,&f); CHKERRQ(ierr);

  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexComputeCellGeometryFVM(dm, c, PETSC_NULL, &centroid[0],
                                        PETSC_NULL); CHKERRQ(ierr);
    ierr = tdy->ops->computeforcing(tdy,&centroid[0],&f[c],NULL);
  }
  ierr = VecRestoreArray(forcing,&f); CHKERRQ(ierr);
  ierr = VecView(forcing,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = VecDestroy(&forcing); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GeometryWheeler2006_2(DM dm, PetscInt n){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,dim;
  PetscReal    x,y;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMGetLabelByNum(dm,2,&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    x = coords[offset  ];
    y = coords[offset+1];
    if(x < 0.5){
      coords[offset+1] = (-4.8*x*x + 2.*x + 0.7)*y ;
    }else{
      coords[offset+1] = (0.75*x + 0.125)*y ;
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GeometryColumn(DM dm){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,dim;
  PetscReal    x,y;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMGetLabel(dm,"boundary",&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    x = coords[offset  ]; y = coords[offset+1];
    coords[offset+2] += 0.05*x+0.15*y;
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// This set of globals is used by CreateDM below to create a DM for this demo.
typedef struct DMOptions {
  PetscInt dim;      // Dimension of DM (2 or 3)
  PetscInt N;        // Number of cells on a side
  PetscBool perturb; // whether to perturb randomly (as opposed to smoothly)
  PetscBool exo;     // whether to load a named exodus file
  PetscBool column;  // column mesh?
  char exofile[256]; // name of the exodus file to load
} DMOptions;

static DMOptions dm_options_;

// This function creates a DM specifically for this demo. Overrides are applied
// to the resulting DM with TDySetFromOptions.
PetscErrorCode CreateDM(void* context, MPI_Comm comm, DM* dm) {
  int ierr;

  PetscInt N = dm_options_.N;
  if(dm_options_.exo) {
    ierr = DMPlexCreateExodusFromFile(PETSC_COMM_WORLD, dm_options_.exofile,
      PETSC_TRUE,dm); CHKERRQ(ierr);
  } else {
    PetscInt Nx=N,Ny=N,Nz=N;
    PetscReal Lx=1,Ly=1,Lz=1;
    if(dm_options_.column){
      Nx = 1; Ny = 1; Nz = N;
      Lx = 10; Ly = 10; Lz = 1;
    }
    const PetscInt  faces[3] = {Nx ,Ny ,Nz };
    const PetscReal lower[3] = {0.0,0.0,0.0};
    const PetscReal upper[3] = {Lx ,Ly ,Lz };
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dm_options_.dim,PETSC_FALSE,
             faces,lower,upper,NULL,PETSC_TRUE,dm); CHKERRQ(ierr);
  }
}

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code=0;
  char vertices_filename[256]="none";
  char centroids_filename[256]="none";
  char true_pres_filename[256]="none";
  char forcing_filename[256]="none";
  char paper[256];
  PetscBool wheeler2006, wheeler2012;

  // DM-related options.
  PetscInt N = 4, dim = 2, problem = 2;
  PetscBool perturb = PETSC_FALSE, exo = PETSC_FALSE, column;
  char exofile[256];

  ierr = TDyInit(argc, argv); CHKERRQ(ierr);
  MPI_Comm comm = PETSC_COMM_WORLD;
  TDy tdy;
  ierr = TDyCreate(comm, &tdy); CHKERRQ(ierr);
  ierr = TDySetDiscretization(tdy,MPFA_O); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Sample Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Problem dimension","",dim,&dim,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-problem","Problem number","",problem,&problem,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Permeability scaling","",alpha,&alpha,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsString("-exo","Mesh file in exodus format","",exofile,exofile,256,&exo); CHKERRQ(ierr);
  ierr = PetscOptionsString("-paper","Select paper","",paper,paper,256,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_vertices","Filename to save vertices","",vertices_filename,vertices_filename,256,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_centroids","Filename to save centroids","",centroids_filename,centroids_filename,256,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_true_pressure","Filename to save true pressure","",true_pres_filename,true_pres_filename,256,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_forcing","Filename to save forcing","",forcing_filename,forcing_filename,256,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = PetscStrcasecmp(paper,"wheeler2006",&wheeler2006); CHKERRQ(ierr);
  ierr = PetscStrcasecmp(paper,"wheeler2012",&wheeler2012); CHKERRQ(ierr);
  ierr = PetscStrcasecmp(paper,"column",&column); CHKERRQ(ierr);

  // Copy DM-related globals into place for use with CreateDM.
  dm_options_.N = N;
  dm_options_.dim = dim;
  dm_options_.perturb = perturb;
  dm_options_.column = column;
  dm_options_.exo = exo;
  if (exo) strcpy(dm_options_.exofile, exofile);

  // Specify a special DM to be constructed for this demo.
  ierr = TDySetDMConstructor(tdy, CreateDM); CHKERRQ(ierr);

  // Apply overrides.
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  // Setup problem parameters
  if(wheeler2006){
    if(dim != 2){
      SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2006 is only for -dim 2 problems");
    }
    switch(problem) {
        case 0: // not a problem in the paper, but want to check constants on the geometry
        ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFn(tdy,PressureConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFn(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
        break;
        case 1:
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2006_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2006_1,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2006_1,NULL); CHKERRQ(ierr);
        break;
        case 2:
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2006_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_2,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2006_2,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2006_2,NULL); CHKERRQ(ierr);
        break;
        default:
        SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2006 only valid for -problem {0,1,2}");
    }
  }else if(wheeler2012){
    switch(problem) {
        case 1:
        if(dim != 2){
          SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2012 -problem 1 is only for -dim 2");
        }
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2012_1,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2012_1,NULL); CHKERRQ(ierr);
        break;
        case 2:
        if(dim != 3){
          SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2012 -problem 2 is only for -dim 3");
        }
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2012_2,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2012_2,NULL); CHKERRQ(ierr);
        break;
        default:
        SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2012 only valid for -problem {1,2}");
    }
  }else{
    switch(problem) {
        case 1:
        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
        } else {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        }
        ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFn(tdy,PressureConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFn(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
        break;

        case 2:

        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingQuadratic2D,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFn(tdy,PressureQuadratic2D,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFn(tdy,VelocityQuadratic2D,NULL); CHKERRQ(ierr);
        } else {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2012_2,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2012_2,NULL); CHKERRQ(ierr);
        }
        break;

        case 3:
        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2006_1,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2006_1,NULL); CHKERRQ(ierr);
        } else {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,Forcing3,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFn(tdy,Pressure3,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFn(tdy,Velocity3,NULL); CHKERRQ(ierr);
        }
        break;

        case 4:
        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_1); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFn(tdy,PressureWheeler2012_1,NULL); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFn(tdy,VelocityWheeler2012_1,NULL); CHKERRQ(ierr);
        } else {

        }
        break;
    }
  }

  ierr = TDySetup(tdy); CHKERRQ(ierr);

  // Make adjustments to our DM based on the problem.
  DM dm;
  TDyGetDM(tdy, &dm);
  if(wheeler2006 && (problem==2 || problem==0)){
    ierr = GeometryWheeler2006_2(dm,N); CHKERRQ(ierr);
  }
  if(column) {
    ierr = GeometryColumn(dm); CHKERRQ(ierr);
  }
  if(perturb) {
    ierr = PerturbVerticesRandom(dm,1./N); CHKERRQ(ierr);
  }else{
    if(wheeler2012) {
      ierr = PerturbVerticesSmooth(dm); CHKERRQ(ierr);
    }
  }

  // View the configured DM.
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

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
  ierr = KSPCreate(comm,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  /* Output solution */
  PetscViewer viewer;
  PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),"sol.vtk",FILE_MODE_WRITE,&viewer);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer); CHKERRQ(ierr); // the approximate solution
  //ierr = OperatorApplicationResidual(tdy,Ue,K,tdy->ops->compute_boundary_pressure,F);
  ierr = VecView(F,viewer); CHKERRQ(ierr); // the residual K*Ue-F
  ierr = VecView(Ue,viewer); CHKERRQ(ierr);  // the exact solution
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  /* Evaluate error norms */
  PetscReal normp,normv;
  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv);
  ierr = PetscPrintf(comm,"%e %e\n",normp,normv); CHKERRQ(ierr);

  /* Save vertex coordinates */
  PetscBool file_not_specified;
  ierr = PetscStrcasecmp(vertices_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveVertices(dm,vertices_filename); CHKERRQ(ierr);
  }

  /* Save cell centroids */
  ierr = PetscStrcasecmp(centroids_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveCentroids(dm,centroids_filename); CHKERRQ(ierr);
  }

  /* Save true solution */
  ierr = PetscStrcasecmp(true_pres_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveTrueSolution(tdy,true_pres_filename); CHKERRQ(ierr);
  }

  /* Save forcing */
  ierr = PetscStrcasecmp(forcing_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveForcing(tdy,forcing_filename); CHKERRQ(ierr);
  }

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U);

  /* Cleanup */
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&Ue); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = TDyFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
