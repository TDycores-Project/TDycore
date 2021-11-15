#include "tdycore.h"
#include "private/tdycoreimpl.h"

PetscReal alpha = 1;

void PermTest2D(PetscInt n, PetscReal *x,PetscReal *K) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[4*i]);
    Ki[0] = 5; Ki[1] = 1;
    Ki[2] = 1; Ki[3] = 2;
  }
}

void PermTest3D(PetscInt n, PetscReal *x, PetscReal *K) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[9*i]);
    Ki[0] = 4.321; Ki[1] = 1;     Ki[2] = 0.5;
    Ki[3] = 1    ; Ki[4] = 1.234; Ki[5] = 1;
    Ki[6] = 0.5  ; Ki[7] = 1;     Ki[8] = 1.1;
  }
}

/*--- -dim {2|3} -problem 1 ---------------------------------------------------------------*/

void PressureConstant(PetscInt n, PetscReal *x, PetscReal *p) {
  for (PetscInt i = 0; i < n; ++i) {
    p[i] = 3.14;
  }
}

void VelocityConstant(PetscInt n, PetscReal *x, PetscReal *v) {
  for (PetscInt i = 0; i < n; ++i) {
    v[3*i] = 0;
    v[3*i+1] = 0;
    v[3*i+2] = 0;
  }
}

void ForcingConstant(PetscInt n, PetscReal *x, PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    f[i] = 0;
  }
}

/*--- -dim 2 -problem 2 ---------------------------------------------------------------*/

void PressureQuadratic2D(PetscInt n, PetscReal *x, PetscReal *p) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    p[i] = 3.14+X*(1-X)+Y*(1-Y);
  }
}

void VelocityQuadratic2D(PetscInt n, PetscReal *x, PetscReal *v) {
  PetscReal K[4*n];
  PermTest2D(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal *Ki = &(K[4*i]);
    v[2*i]   = -Ki[0]*(1-2*X) - Ki[1]*(1-2*Y);
    v[2*i+1] = -Ki[2]*(1-2*X) - Ki[3]*(1-2*Y);
  }
}

void ForcingQuadratic2D(PetscInt n, PetscReal *x, PetscReal *f) {
  PetscReal K[4*n];
  PermTest2D(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[4*i]);
    f[i] = 2*Ki[0]+2*Ki[3];
  }
}

void PressureQuadratic3D(PetscInt n, PetscReal *x, PetscReal *p) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    p[i] = 3.14+X*(1-X)+Y*(1-Y)+Z*(1-Z);
  }
}

void VelocityQuadratic3D(PetscInt n, PetscReal *x, PetscReal *v) {
  PetscReal K[9*n];
  PermTest3D(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    PetscReal *Ki = &(K[9*i]);
    v[3*i+0] = -Ki[0]*(1-2*X) - Ki[1]*(1-2*Y) - Ki[2]*(1-2*Z);
    v[3*i+1] = -Ki[3]*(1-2*X) - Ki[4]*(1-2*Y) - Ki[5]*(1-2*Z);
    v[3*i+2] = -Ki[6]*(1-2*X) - Ki[7]*(1-2*Y) - Ki[8]*(1-2*Z);
  }
}

void ForcingQuadratic3D(PetscInt n, PetscReal *x, PetscReal *f) {
  PetscReal K[9*n];
  PermTest3D(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[9*i]);
    f[i] = 2*(Ki[0]+Ki[3]+Ki[8]);
  }
}

/*--- -paper Wheeler2006 -------------------------------------------------------*/

/* -problem 1 */

void PermWheeler2006_1(PetscInt n, PetscReal *x, PetscReal *K) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[4*i]);
    Ki[0] = 5; Ki[1] = 1;
    Ki[2] = 1; Ki[3] = 2;
  }
}

void PressureWheeler2006_1(PetscInt n, PetscReal *x, PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    f[n]  = PetscPowReal(1-X,4);
    f[n] += PetscPowReal(1-Y,3)*(1-X);
    f[n] += PetscSinReal(1-Y)*PetscCosReal(1-X);
  }
}

void VelocityWheeler2006_1(PetscInt n, PetscReal *x, PetscReal *v) {
  PetscReal K[4*n];
  PermWheeler2006_1(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal vx, vy;
    PetscReal *Ki = &(K[4*i]);
    vx  = -4*PetscPowReal(1-X,3);
    vx += -PetscPowReal(1-Y,3);
    vx += +PetscSinReal(Y-1)*PetscSinReal(X-1);
    vy  = -3*PetscPowReal(1-Y,2)*(1-X);
    vy += -PetscCosReal(X-1)*PetscCosReal(Y-1);
    v[2*i] = -(Ki[0]*vx+Ki[1]*vy);
    v[2*i+1] = -(Ki[2]*vx+Ki[3]*vy);
  }
}

void ForcingWheeler2006_1(PetscInt n, PetscReal *x, PetscReal *f) {
  PetscReal K[4*n];
  PermWheeler2006_1(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal *Ki = &(K[4*i]);
    f[i]  = -Ki[0]*(12*PetscPowReal(1-X, 2)+PetscSinReal(Y-1)*PetscCosReal(X-1));
    f[i] += -Ki[1]*( 3*PetscPowReal(1-Y, 2)+PetscSinReal(X-1)*PetscCosReal(Y-1));
    f[i] += -Ki[2]*( 3*PetscPowReal(1-Y, 2)+PetscSinReal(X-1)*PetscCosReal(Y-1));
    f[i] += -Ki[3]*(-6*(1-X)*(Y-1)+PetscSinReal(Y-1)*PetscCosReal(X-1));
  }
}

/* -problem 2 */

void PermWheeler2006_2(PetscInt n, PetscReal *x, PetscReal *K) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal *Ki = &(K[4*i]);
    Ki[0] = 4 + PetscSqr(X+2) + PetscSqr(Y);
    Ki[1] = 1 + PetscSinReal(X*Y);
    Ki[2] = Ki[1];
    Ki[3] = 2;
  }
}

void PressureWheeler2006_2(PetscInt n, PetscReal *x, PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal sx = PetscSinReal(3*PETSC_PI*X);
    PetscReal sy = PetscSinReal(3*PETSC_PI*Y);
    f[i] = sx*sx*sy*sy;
  }
}

void VelocityWheeler2006_2(PetscInt n, PetscReal *x, PetscReal *v) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal sx = PetscSinReal(3*PETSC_PI*X);
    PetscReal sy = PetscSinReal(3*PETSC_PI*Y);
    PetscReal cx = PetscCosReal(3*PETSC_PI*X);
    PetscReal cy = PetscCosReal(3*PETSC_PI*Y);
    v[2*i]   = -6*PETSC_PI*((sin(X*Y) + 1)*sx*cy + (Y*Y + PetscSqr(X+2) + 4)*sy*cx)*sx*sy;
    v[2*i+1] = -6*PETSC_PI*((sin(X*Y) + 1)*sy*cx + 2*sx*cy)*sx*sy;
  }
}

void ForcingWheeler2006_2(PetscInt n, PetscReal *x,PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal sx = PetscSinReal(3*PETSC_PI*X);
    PetscReal sy = PetscSinReal(3*PETSC_PI*Y);
    PetscReal cx = PetscCosReal(3*PETSC_PI*X);
    PetscReal cy = PetscCosReal(3*PETSC_PI*Y);
    f[i]  = -12*PETSC_PI*(PetscSinReal(X*Y) + 1)*sx*sy*cx*cy;
    f[i] +=  3*PETSC_PI*(Y*Y + PetscSqr(X + 2) + 4)*sx*sx*sy*sy;
    f[i] -=  3*PETSC_PI*(Y*Y + PetscSqr(X + 2) + 4)*sy*sy*cx*cx;
    f[i] +=  6*PETSC_PI*sx*sx*sy*sy;
    f[i] -=  6*PETSC_PI*sx*sx*cy*cy;
    f[i] -=  X*sx*sy*sy*cx*PetscCosReal(X*Y);
    f[i] -=  Y*sx*sx*sy*cy*PetscCosReal(X*Y);
    f[i] -=  2*(X + 2)*sx*sy*sy*cx;
    f[i] *= 6*PETSC_PI;
  }
}

/*--- -paper Wheeler2012 -------------------------------------------------------*/

/* -problem 1 */

void PermWheeler2012_1(PetscInt n, PetscReal *x, PetscReal *K) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[4*i]);
    Ki[0] = 2   ; Ki[1] = 1.25;
    Ki[2] = 1.25; Ki[3] = 3;
  }
}

void PressureWheeler2012_1(PetscInt n, PetscReal *x, PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal sx = PetscSinReal(3*PETSC_PI*X);
    PetscReal sy = PetscSinReal(3*PETSC_PI*Y);
    f[i] = sx*sx*sy*sy;
  }
}

void VelocityWheeler2012_1(PetscInt n, PetscReal *x, PetscReal *v) {
  PetscReal K[4*n];
  PermWheeler2012_1(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal sx = PetscSinReal(3*PETSC_PI*X);
    PetscReal sy = PetscSinReal(3*PETSC_PI*Y);
    PetscReal cx = PetscCosReal(3*PETSC_PI*X);
    PetscReal cy = PetscCosReal(3*PETSC_PI*Y);
    PetscReal px,py;
    px = 6*PETSC_PI*sx*sy*sy*cx;
    py = 6*PETSC_PI*sx*sx*sy*cy;
    PetscReal *Ki = &(K[4*i]);
    v[0] = -(Ki[0]*px+Ki[1]*py);
    v[1] = -(Ki[2]*px+Ki[3]*py);
  }
}

void ForcingWheeler2012_1(PetscInt n, PetscReal *x, PetscReal *f) {
  PetscReal K[4*n];
  PermWheeler2012_1(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[2*i], Y = x[2*i+1];
    PetscReal sx = PetscSinReal(3*PETSC_PI*X);
    PetscReal sy = PetscSinReal(3*PETSC_PI*Y);
    PetscReal cx = PetscCosReal(3*PETSC_PI*X);
    PetscReal cy = PetscCosReal(3*PETSC_PI*Y);
    PetscReal *Ki = &(K[4*i]);
    f[i]  = Ki[0]*sx*sx*sy*sy;
    f[i] -= Ki[0]*sy*sy*cx*cx;
    f[i] -= Ki[1]*(PetscCosReal(6*PETSC_PI*(X-Y))-PetscCosReal(6*PETSC_PI*(X+Y)))*0.25;
    f[i] -= Ki[2]*(PetscCosReal(6*PETSC_PI*(X-Y))-PetscCosReal(6*PETSC_PI*(X+Y)))*0.25;
    f[i] += Ki[3]*sx*sx*sy*sy;
    f[i] -= Ki[3]*sx*sx*cy*cy;
    f[i] *= 18*PETSC_PI*PETSC_PI;
  }
}

/* -problem 2 */

void PermWheeler2012_2(PetscInt n, PetscReal *x, PetscReal *K) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal *Ki = &(K[9*i]);
    Ki[0] = alpha; Ki[1] = 1; Ki[2] = 1;
    Ki[3] = 1    ; Ki[4] = 2; Ki[5] = 1;
    Ki[6] = 1    ; Ki[7] = 1; Ki[8] = 2;
  }
}

void PressureWheeler2012_2(PetscInt n, PetscReal *x, PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    PetscReal X2 = X*X, Y2 = Y*Y, Z2 = Z*Z;
    PetscReal Xm12 = PetscSqr(X-1);
    PetscReal Ym12 = PetscSqr(Y-1);
    PetscReal Zm12 = PetscSqr(Z-1);
    f[i] = X2*Y2*Z2*Xm12*Ym12*Zm12;
  }
}

void VelocityWheeler2012_2(PetscInt n, PetscReal *x, PetscReal *v) {
  PetscReal K[9*n];
  PermWheeler2012_2(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    PetscReal X2 = X*X, Y2 = Y*Y, Z2 = Z*Z;
    PetscReal Xm12 = PetscSqr(X-1);
    PetscReal Ym12 = PetscSqr(Y-1);
    PetscReal Zm12 = PetscSqr(Z-1);

    PetscReal a1 = 2*X*(X-1)*(2*X-1);
    PetscReal b1 = 2*Y*(Y-1)*(2*Y-1);
    PetscReal c1 = 2*Z*(Z-1)*(2*Z-1);

    PetscReal px = a1     *Y2*Ym12 *Z2*Zm12;
    PetscReal py = X2*Xm12*b1      *Z2*Zm12;
    PetscReal pz = X2*Xm12*Y2*Ym12*c1     ;

    PetscReal *Ki = &(K[9*i]);
    v[3*i]   = -Ki[0]*px - Ki[1]*py - Ki[2]*pz;
    v[3*i+1] = -Ki[3]*px - Ki[4]*py - Ki[5]*pz;
    v[3*i+2] = -Ki[6]*px - Ki[7]*py - Ki[8]*pz;
  }
}

void ForcingWheeler2012_2(PetscInt n, PetscReal *x, PetscReal *f) {
  PetscReal K[9*n];
  PermWheeler2012_2(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    PetscReal X2 = X*X, Y2 = Y*Y, Z2 = Z*Z;
    PetscReal Xm12 = PetscSqr(X-1);
    PetscReal Ym12 = PetscSqr(Y-1);
    PetscReal Zm12 = PetscSqr(Z-1);

    PetscReal a1 = 2*X*(X-1)*(2*X-1);
    PetscReal b1 = 2*Y*(Y-1)*(2*Y-1);
    PetscReal c1 = 2*Z*(Z-1)*(2*Z-1);

    PetscReal a2 = 12*X2 - 12*X + 2;
    PetscReal b2 = 12*Y2 - 12*Y + 2;
    PetscReal c2 = 12*Z2 - 12*Z + 2;

    PetscReal *Ki = &(K[9*i]);
    f[i] =
      -Ki[0]*a2*Y2*Ym12*Z2*Zm12 -Ki[1]*a1     *b1*Z2*Zm12 -Ki[2]*a1     *Y2*Ym12*c1 +
      -Ki[3]*a1*b1     *Z2*Zm12 -Ki[4]*X2*Xm12*b2*Z2*Zm12 -Ki[5]*X2*Xm12*b1     *c1 +
      -Ki[6]*a1*Y2*Ym12*c1      -Ki[7]*X2*Xm12*b1*c1      -Ki[8]*X2*Xm12*Y2*Ym12*c2;
  }
}

/*--- -dim 3 -problem 3 ---------------------------------------------------------------*/

void Pressure3(PetscInt n, PetscReal *x, PetscReal *f) {
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    f[i] = PetscCosReal(X)*PetscCosReal(Y)*PetscCosReal(Z);
  }
}

void Velocity3(PetscInt n, PetscReal *x, PetscReal *v) {
  PetscReal K[9*n];
  PermWheeler2012_2(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    PetscReal *Ki = &(K[9*i]);
    v[3*i]   = Ki[0]*PetscSinReal(X)*PetscCosReal(Y)*PetscCosReal(Z) +
               Ki[1]*PetscSinReal(Y)*PetscCosReal(X)*PetscCosReal(Z) +
               Ki[2]*PetscSinReal(Z)*PetscCosReal(X)*PetscCosReal(Y);
    v[3*i+1] = Ki[3]*PetscSinReal(X)*PetscCosReal(Y)*PetscCosReal(Z) +
               Ki[4]*PetscSinReal(Y)*PetscCosReal(X)*PetscCosReal(Z) +
               Ki[5]*PetscSinReal(Z)*PetscCosReal(X)*PetscCosReal(Y);
    v[3*i+2] = Ki[6]*PetscSinReal(X)*PetscCosReal(Y)*PetscCosReal(Z) +
               Ki[7]*PetscSinReal(Y)*PetscCosReal(X)*PetscCosReal(Z) +
               Ki[8]*PetscSinReal(Z)*PetscCosReal(X)*PetscCosReal(Y);
  }
}

void Forcing3(PetscInt n, PetscReal *x, PetscReal *f) {
  PetscReal K[9*n];
  PermWheeler2012_2(n, x, K);
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal X = x[3*i], Y = x[3*i+1], Z = x[3*i+2];
    PetscReal *Ki = &(K[9*i]);
    f[i] = Ki[0]*PetscCosReal(X)*PetscCosReal(Y)*PetscCosReal(Z) -
           Ki[1]*PetscSinReal(X)*PetscSinReal(Y)*PetscCosReal(Z) -
           Ki[2]*PetscSinReal(X)*PetscSinReal(Z)*PetscCosReal(Y) -
           Ki[3]*PetscSinReal(X)*PetscSinReal(Y)*PetscCosReal(Z) +
           Ki[4]*PetscCosReal(X)*PetscCosReal(Y)*PetscCosReal(Z) -
           Ki[5]*PetscSinReal(Y)*PetscSinReal(Z)*PetscCosReal(X) -
           Ki[6]*PetscSinReal(X)*PetscSinReal(Z)*PetscCosReal(Y) -
           Ki[7]*PetscSinReal(Y)*PetscSinReal(Z)*PetscCosReal(X) +
           Ki[8]*PetscCosReal(X)*PetscCosReal(Y)*PetscCosReal(Z);
  }
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
      /* This is because 'boundary' is broken in 3D */
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
    ierr = ConditionsComputeBoundaryPressure(tdy->conditions, 1, &centroid[0],
                                             &pres[c]); CHKERRQ(ierr);
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

  // TODO: This kind of stuff doesn't belong in a demo. We need to hash out our
  // TODO: library interface to provide a higher-level way of accomplishing this.
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexComputeCellGeometryFVM(dm, c, PETSC_NULL, &centroid[0],
                                        PETSC_NULL); CHKERRQ(ierr);
    ierr = ConditionsComputeForcing(tdy->conditions, 1, &centroid[0],&f[c]);
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

// This data is used by CreateDM below to create a DM for this demo.
typedef struct DMOptions {
  PetscInt dim;        // Dimension of DM (2 or 3)
  PetscInt N;          // Number of cells on a side
  PetscBool exo;       // whether to load a named exodus file
  PetscBool column;    // column mesh?
  const char* exofile; // name of the exodus file to load
} DMOptions;

// This function creates a DM specifically for this demo. Overrides are applied
// to the resulting DM with TDySetFromOptions.
PetscErrorCode CreateDM(MPI_Comm comm, void* context, DM* dm) {
  int ierr;
  DMOptions* options = context;

  PetscInt N = options->N;
  if(options->exo) {
    ierr = DMPlexCreateExodusFromFile(PETSC_COMM_WORLD, options->exofile,
      PETSC_TRUE,dm); CHKERRQ(ierr);
  } else {
    PetscInt Nx=N,Ny=N,Nz=N;
    PetscReal Lx=1,Ly=1,Lz=1;
    if(options->column){
      Nx = 1; Ny = 1; Nz = N;
      Lx = 10; Ly = 10; Lz = 1;
    }
    const PetscInt  faces[3] = {Nx ,Ny ,Nz };
    const PetscReal lower[3] = {0.0,0.0,0.0};
    const PetscReal upper[3] = {Lx ,Ly ,Lz };
    ierr = DMPlexCreateBoxMesh(comm, options->dim, PETSC_FALSE,
      faces,lower,upper,NULL,PETSC_TRUE,dm); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  ierr = TDySetMode(tdy,RICHARDS); CHKERRQ(ierr);
  ierr = TDySetDiscretization(tdy,MPFA_O); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Sample Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Problem dimension","",dim,&dim,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D","",N,&N,NULL); CHKERRQ(ierr);
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

  // Specify a special DM to be constructed for this demo, and pass it the
  // relevant options.
  DMOptions dm_options = {.N = N, .dim = dim, .exo = exo, .exofile = exofile};
  ierr = TDySetDMConstructor(tdy, &dm_options, CreateDM); CHKERRQ(ierr);

  // Apply overrides.
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  // Setup problem parameters
  if(wheeler2006){
    if(dim != 2){
      SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2006 is only for -dim 2 problems");
    }
    switch(problem) {
        case 0: // not a problem in the paper, but want to check constants on the geometry
        ierr = TDySetTensorPermeabilityFunction(tdy,PermTest2D); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingConstant); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFunction(tdy,PressureConstant); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFunction(tdy,VelocityConstant); CHKERRQ(ierr);
        break;
        case 1:
        ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2006_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2006_1); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2006_1); CHKERRQ(ierr);
        break;
        case 2:
        ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2006_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_2); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2006_2); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2006_2); CHKERRQ(ierr);
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
        ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2012_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2012_1); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2012_1); CHKERRQ(ierr);
        break;
        case 2:
        if(dim != 3){
          SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2012 -problem 2 is only for -dim 3");
        }
        ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2012_2); CHKERRQ(ierr);
        break;
        default:
        SETERRQ(comm,PETSC_ERR_USER,"-paper wheeler2012 only valid for -problem {1,2}");
    }
  }else{
    switch(problem) {
        case 1:
        if (dim == 2) {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermTest2D); CHKERRQ(ierr);
        } else {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        }
        ierr = TDySetForcingFunction(tdy,ForcingConstant); CHKERRQ(ierr);
        ierr = TDySetBoundaryPressureFunction(tdy,PressureConstant); CHKERRQ(ierr);
        ierr = TDySetBoundaryVelocityFunction(tdy,VelocityConstant); CHKERRQ(ierr);
        break;

        case 2:

        if (dim == 2) {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermTest2D); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingQuadratic2D); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFunction(tdy,PressureQuadratic2D); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFunction(tdy,VelocityQuadratic2D); CHKERRQ(ierr);
        } else {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2012_2); CHKERRQ(ierr);
        }
        break;

        case 3:
        if (dim == 2) {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermTest2D); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2006_1); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2006_1); CHKERRQ(ierr);
        } else {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,Forcing3); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFunction(tdy,Pressure3); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFunction(tdy,Velocity3); CHKERRQ(ierr);
        }
        break;

        case 4:
        if (dim == 2) {
          ierr = TDySetTensorPermeabilityFunction(tdy,PermWheeler2012_1); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1); CHKERRQ(ierr);
          ierr = TDySetBoundaryPressureFunction(tdy,PressureWheeler2012_1); CHKERRQ(ierr);
          ierr = TDySetBoundaryVelocityFunction(tdy,VelocityWheeler2012_1); CHKERRQ(ierr);
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
  ierr = TDyWYComputeSystem(tdy,K,F); CHKERRQ(ierr);

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
