#include "tdycore.h"

PetscReal alpha = 1;

void Permeability(double *x,double *K) {
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

/*--- -dim {2|3} -problem 1 ---------------------------------------------------------------*/

PetscErrorCode PressureConstant(TDy tdy,double *x,double *p,void *ctx) { (*p) = 3.14; PetscFunctionReturn(0);}
void VelocityConstant(double *x,double *v) { v[0] = 0; v[1] = 0; v[2] = 0;}
PetscErrorCode ForcingConstant(TDy tdy,double *x,double *f,void *ctx) { (*f) = 0; PetscFunctionReturn(0);}

/*--- -dim 2 -problem 2 ---------------------------------------------------------------*/

PetscErrorCode PressureQuadratic(TDy tdy,double *x,double *p,void *ctx) { (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1]); PetscFunctionReturn(0);}
void VelocityQuadratic(double *x,double *v) {
  double K[4]; Permeability(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]);
  v[1] = -K[2]*(1-2*x[0]) - K[3]*(1-2*x[1]);
}
PetscErrorCode ForcingQuadratic(TDy tdy,double *x,double *f,void *ctx) { double K[4]; Permeability(x,K); (*f) = 2*K[0]+2*K[3]; PetscFunctionReturn(0);}

/*--- -dim 2 -problem 3 ---------------------------------------------------------------*/

PetscErrorCode Pressure(TDy tdy,double *x,double *f,void *ctx) {
  (*f)  = PetscPowReal(1-x[0],4);
  (*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
  (*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
  PetscFunctionReturn(0);
}
void Velocity(double *x,double *v) {
  double vx,vy,K[4];
  Permeability(x,K);
  vx  = -4*PetscPowReal(1-x[0],3);
  vx += -PetscPowReal(1-x[1],3);
  vx += +PetscSinReal(x[1]-1)*PetscSinReal(x[0]-1);
  vy  = -3*PetscPowReal(1-x[1],2)*(1-x[0]);
  vy += -PetscCosReal(x[0]-1)*PetscCosReal(x[1]-1);
  v[0] = -(K[0]*vx+K[1]*vy);
  v[1] = -(K[2]*vx+K[3]*vy);
}
PetscErrorCode Forcing(TDy tdy,double *x,double *f, void *ctx) {
  double K[4];
  Permeability(x,K);
  (*f)  = -K[0]*(12*PetscPowReal(1-x[0],
                                 2)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  (*f) += -K[1]*( 3*PetscPowReal(1-x[1],
                                 2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[2]*( 3*PetscPowReal(1-x[1],
                                 2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[3]*(-6*(1-x[0])*(x[1]-1)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  PetscFunctionReturn(0);
}

/*--- -dim 2 -problem 4 ---------------------------------------------------------------*/

void PermeabilitySine(double *x,double *K) {
  K[0] = 2   ; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
}
PetscErrorCode PressureSine(TDy tdy,double *x,double *p,void *ctx) {
  PetscReal s = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal t = PetscSinReal(3*PETSC_PI*x[1]);
  (*p) = s*s*t*t;
  PetscFunctionReturn(0);
}
void VelocitySine(double *x,double *v) {
  double K[4]; PermeabilitySine(x,K);
  double pi = PETSC_PI;
  double s3pX = PetscSinReal(3*pi*x[0]);
  double s3pY = PetscSinReal(3*pi*x[1]);
  double c3pX = PetscCosReal(3*pi*x[0]);
  double c3pY = PetscCosReal(3*pi*x[1]);
  v[0] = -6*K[0]*pi*s3pX*s3pY*s3pY*c3pX - 6*K[1]*pi*s3pX*s3pX*s3pY*c3pY;
  v[1] = -6*K[2]*pi*s3pX*s3pY*s3pY*c3pX - 6*K[3]*pi*s3pX*s3pX*s3pY*c3pY;
}
PetscErrorCode ForcingSine(TDy tdy,double *x,double *f,void *ctx) {
  double K[4]; PermeabilitySine(x,K);
  double pi = PETSC_PI;
  double s3pX = PetscSinReal(3*pi*x[0]);
  double s3pY = PetscSinReal(3*pi*x[1]);
  double c3pX = PetscCosReal(3*pi*x[0]);
  double c3pY = PetscCosReal(3*pi*x[1]);
  (*f) =  18*pi*pi*(K[0]*s3pX*s3pX*s3pY*s3pY - K[0]*s3pY*s3pY*c3pX*c3pX - K[1]*
                    (PetscCosReal(6*pi*(x[0] - x[1])) - PetscCosReal(6*pi*(x[0] + x[1])))/4 - K[2]*
                    (PetscCosReal(6*pi*(x[0] - x[1])) - PetscCosReal(6*pi*(x[0] + x[1])))/4 +
                    K[3]*s3pX*s3pX*s3pY*s3pY - K[3]*s3pX*s3pX*c3pY*c3pY);
 PetscFunctionReturn(0);

}

/*--- -dim 3 -problem 2 ---------------------------------------------------------------*/

void Permeability3D(double *x,double *K) {
  K[0] = alpha; K[1] = 1; K[2] = 1;
  K[3] = 1    ; K[4] = 2; K[5] = 1;
  K[6] = 1    ; K[7] = 1; K[8] = 2;
}

PetscErrorCode Pressure3D(TDy tdy,double *x,double *f,void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  (*f) = x2*y2*z2*xm12*ym12*zm12;
  PetscFunctionReturn(0);
}

void Velocity3D(double *x,double *v) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  PetscReal px = (x2*y2*z2*(2*x[0] - 2)*ym12*zm12 + 2*x[0]*y2*z2*xm12*ym12*zm12);
  PetscReal py = (x2*y2*z2*xm12*(2*x[1] - 2)*zm12 + 2*x2*x[1]*z2*xm12*ym12*zm12);
  PetscReal pz = (x2*y2*z2*xm12*ym12*(2*x[2] - 2) + 2*x2*y2*x[2]*xm12*ym12*zm12);
  double K[9]; Permeability3D(x,K);
  v[0] = -K[0]*px - K[1]*py - K[2]*pz;
  v[1] = -K[3]*px - K[4]*py - K[5]*pz;
  v[2] = -K[6]*px - K[7]*py - K[8]*pz;
}

PetscErrorCode Forcing3D(TDy tdy,double *x,double *f, void *ctx) {
  PetscReal x2 = x[0]*x[0], y2 = x[1]*x[1], z2 = x[2]*x[2];
  PetscReal xm1  = (x[0]-1);
  PetscReal ym1  = (x[1]-1);
  PetscReal zm1  = (x[2]-1);
  PetscReal xm12 = PetscSqr(x[0]-1);
  PetscReal ym12 = PetscSqr(x[1]-1);
  PetscReal zm12 = PetscSqr(x[2]-1);
  double K[9]; Permeability3D(x,K);
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

void Velocity3(double *x,double *v) {
  double K[9]; Permeability3D(x,K);
  v[0]  =  K[0]*PetscSinReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
             x[2]) + K[1]*PetscSinReal(x[1])*PetscCosReal(x[0])*PetscCosReal(
             x[2]) + K[2]*PetscSinReal(x[2])*PetscCosReal(x[0])*PetscCosReal(x[1]);
  v[1]  =  K[3]*PetscSinReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
             x[2]) + K[4]*PetscSinReal(x[1])*PetscCosReal(x[0])*PetscCosReal(
             x[2]) + K[5]*PetscSinReal(x[2])*PetscCosReal(x[0])*PetscCosReal(x[1]);
  v[2]  =  K[6]*PetscSinReal(x[0])*PetscCosReal(x[1])*PetscCosReal(
             x[2]) + K[7]*PetscSinReal(x[1])*PetscCosReal(x[0])*PetscCosReal(
             x[2]) + K[8]*PetscSinReal(x[2])*PetscCosReal(x[0])*PetscCosReal(x[1]);
}

PetscErrorCode Forcing3(TDy tdy,double *x,double *f,void *ctx) {
  double K[9]; Permeability3D(x,K);
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

PetscErrorCode PerturbInteriorVertices(DM dm,PetscReal h) {
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
      /* this is because 'marker' is broken in 3D */
      if(coords[offset] > 0 && coords[offset] < 1 &&
          coords[offset+1] > 0 && coords[offset+1] < 1 &&
          coords[offset+2] > 0 && coords[offset+2] < 1) {
        coords[offset+2] += (((PetscReal)rand())/((PetscReal)RAND_MAX)-0.5)*h*0.1;
      }
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2, problem = 2;
  PetscInt successful_exit_code=0;
  PetscBool perturb = PETSC_FALSE;
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
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  const PetscInt  faces[3] = {N,N,N  };
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
                             NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
  if(perturb) {
    ierr = PerturbInteriorVertices(dm,1./N); CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);
  if(dim == 2) {
    ierr = TDySetPermeabilityTensor(tdy,Permeability); CHKERRQ(ierr);
    switch(problem) {
    case 1:
      ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocityConstant); CHKERRQ(ierr);
      break;
    case 2:
      ierr = TDySetForcingFunction(tdy,ForcingQuadratic,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureQuadratic,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocityQuadratic); CHKERRQ(ierr);
      break;
    case 3:
      ierr = TDySetForcingFunction(tdy,Forcing,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,Pressure,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,Velocity); CHKERRQ(ierr);
      break;
    case 4:
      ierr = TDySetPermeabilityTensor(tdy,PermeabilitySine); CHKERRQ(ierr);
      ierr = TDySetForcingFunction(tdy,ForcingSine,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureSine,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocitySine); CHKERRQ(ierr);
      break;
    }
  } else {
    ierr = TDySetPermeabilityTensor(tdy,Permeability3D); CHKERRQ(ierr);
    switch(problem) {
    case 1:
      ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocityConstant); CHKERRQ(ierr);
      break;
    case 2:
      ierr = TDySetForcingFunction(tdy,Forcing3D,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,Pressure3D,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,Velocity3D); CHKERRQ(ierr);
      break;
    case 3:
      ierr = TDySetForcingFunction(tdy,Forcing3,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,Pressure3,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,Velocity3); CHKERRQ(ierr);
      break;
    }
  }
  ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Compute system */
  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F); CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K); CHKERRQ(ierr);
  ierr = TDyComputeSystem(tdy,K,F); CHKERRQ(ierr);

  /* Solve system */
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  /* Evaluate error norms */
  PetscReal normp,normv;
  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%e %e\n",normp,normv); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U);

  /* Cleanup */
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
