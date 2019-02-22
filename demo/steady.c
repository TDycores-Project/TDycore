#include "tdycore.h"

PetscReal alpha = 1;

void Permeability(double *x,double *K){
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

/*--- -dim 2 -problem 1 ---------------------------------------------------------------*/

void PressureConstant(double *x,double *p){ (*p) = 1; }
void VelocityConstant(double *x,double *v){ v[0] = 0; v[1] = 0; }
void ForcingConstant(double *x,double *f){ (*f) = 0; }

/*--- -dim 2 -problem 2 ---------------------------------------------------------------*/

void PressureQuadratic(double *x,double *p){ (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1]); }
void VelocityQuadratic(double *x,double *v){
  double K[4]; Permeability(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]);
  v[1] = -K[2]*(1-2*x[0]) - K[3]*(1-2*x[1]);
}
void ForcingQuadratic(double *x,double *f){ double K[4]; Permeability(x,K); (*f) = 2*K[0]+2*K[3]; }

/*--- -dim 2 -problem 3 ---------------------------------------------------------------*/

void Pressure(double *x,double *f){
  (*f)  = PetscPowReal(1-x[0],4);
  (*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
  (*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
}
void Velocity(double *x,double *v){
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
void Forcing(double *x,double *f){
  double K[4];
  Permeability(x,K);
  (*f)  = -K[0]*(12*PetscPowReal(1-x[0],2)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  (*f) += -K[1]*( 3*PetscPowReal(1-x[1],2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[2]*( 3*PetscPowReal(1-x[1],2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[3]*(-6*(1-x[0])*(x[1]-1)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
}

/*--- -dim 2 -problem 4 ---------------------------------------------------------------*/
void PermeabilitySine(double *x,double *K){
  K[0] = 2   ; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
}
void PressureSine(double *x,double *p){
  PetscReal s = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal t = PetscSinReal(3*PETSC_PI*x[1]);
  (*p) = s*s*t*t;
}
void VelocitySine(double *x,double *v){
  double K[4]; PermeabilitySine(x,K);
  double pi = PETSC_PI;
  double s3pX = PetscSinReal(3*pi*x[0]);
  double s3pY = PetscSinReal(3*pi*x[1]);
  double c3pX = PetscCosReal(3*pi*x[0]);
  double c3pY = PetscCosReal(3*pi*x[1]);
  v[0] = -6*K[0]*pi*s3pX*s3pY*s3pY*c3pX - 6*K[1]*pi*s3pX*s3pX*s3pY*c3pY;
  v[1] = -6*K[2]*pi*s3pX*s3pY*s3pY*c3pX - 6*K[3]*pi*s3pX*s3pX*s3pY*c3pY;
}
void ForcingSine(double *x,double *f){
  double K[4]; PermeabilitySine(x,K);
  double pi = PETSC_PI;
  double s3pX = PetscSinReal(3*pi*x[0]);
  double s3pY = PetscSinReal(3*pi*x[1]);
  double c3pX = PetscCosReal(3*pi*x[0]);
  double c3pY = PetscCosReal(3*pi*x[1]);
  (*f) =  18*pi*pi*(K[0]*s3pX*s3pX*s3pY*s3pY - K[0]*s3pY*s3pY*c3pX*c3pX - K[1]*(PetscCosReal(6*pi*(x[0] - x[1])) - PetscCosReal(6*pi*(x[0] + x[1])))/4 - K[2]*(PetscCosReal(6*pi*(x[0] - x[1])) - PetscCosReal(6*pi*(x[0] + x[1])))/4 + K[3]*s3pX*s3pX*s3pY*s3pY - K[3]*s3pX*s3pX*c3pY*c3pY);
}

/*-------------------------------------------------------------------------------------*/

void Permeability3D(double *x,double *K){
  K[0] = alpha; K[1] = 1; K[2] = 1;
  K[3] = 1    ; K[4] = 2; K[5] = 1;
  K[6] = 1    ; K[7] = 1; K[8] = 2;
}

void Pressure3D(double *x,double *f){
  (*f)  = x[0]*(-x[0]+1);
  (*f) += x[1]*(-x[1]+1);
  (*f) += x[2]*(-x[2]+1);
  //(*f) = x[0] + x[1] + x[2];
}

void Velocity3D(double *x,double *v){
  double K[9]; Permeability3D(x,K);
  v[0] = -K[0]*(-2*x[0]+1) - K[1]*(-2*x[1]+1) - K[2]*(-2*x[2]+1);
  v[1] = -K[3]*(-2*x[0]+1) - K[4]*(-2*x[1]+1) - K[5]*(-2*x[2]+1);
  v[2] = -K[6]*(-2*x[0]+1) - K[7]*(-2*x[1]+1) - K[8]*(-2*x[2]+1);
  //v[0] = -(K[0]+K[1]+K[2]);
  //v[1] = -(K[3]+K[4]+K[5]);
  //v[2] = -(K[6]+K[7]+K[8]);
}

void Forcing3D(double *x,double *f){
  double K[9]; Permeability3D(x,K);
  (*f) = 2*(K[0] + K[4] + K[8]);
  //(*f) = 0;
}

PetscErrorCode PerturbInteriorVertices(DM dm,PetscReal h){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value;
  ierr = DMGetLabelByNum(dm,2,&label);CHKERRQ(ierr); // this is the 'marker' label which marks boundary entities
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){
    ierr = PetscSectionGetOffset(coordSection,v,&offset);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value);CHKERRQ(ierr);
    if(value==-1){
      PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*(h*0.471404); // h*sqrt(2)/3
      PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
      coords[offset  ] += r*PetscCosReal(t);
      coords[offset+1] += r*PetscSinReal(t);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2, problem = 3;
  PetscBool perturb = PETSC_FALSE;
  ierr = PetscInitialize(&argc,&argv,(char*)0,0);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim"    ,"Problem dimension"        ,"",dim    ,&dim    ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N"      ,"Number of elements in 1D" ,"",N      ,&N      ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-problem","Problem number"           ,"",problem,&problem,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-perturb","Perturb interior vertices","",perturb,&perturb,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha"  ,"Permeability scaling"     ,"",alpha  ,&alpha  ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  const PetscInt  faces[3] = {N  ,N  ,N  };
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  if(perturb){
    ierr = PerturbInteriorVertices(dm,1./N);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);  
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy);CHKERRQ(ierr);
  if(dim == 2){
    ierr = TDySetPermeabilityTensor(tdy,Permeability);CHKERRQ(ierr);
    switch(problem){
    case 1:
      ierr = TDySetForcingFunction(tdy,ForcingConstant);CHKERRQ(ierr);
      ierr = TDySetDirichletFunction(tdy,PressureConstant);CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocityConstant);CHKERRQ(ierr);
      break;
    case 2:
      ierr = TDySetForcingFunction(tdy,ForcingQuadratic);CHKERRQ(ierr);
      ierr = TDySetDirichletFunction(tdy,PressureQuadratic);CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocityQuadratic);CHKERRQ(ierr);
      break;
    case 3:
      ierr = TDySetForcingFunction(tdy,Forcing);CHKERRQ(ierr);
      ierr = TDySetDirichletFunction(tdy,Pressure);CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,Velocity);CHKERRQ(ierr);
      break;
    case 4:
      ierr = TDySetPermeabilityTensor(tdy,PermeabilitySine);CHKERRQ(ierr);
      ierr = TDySetForcingFunction(tdy,ForcingSine);CHKERRQ(ierr);
      ierr = TDySetDirichletFunction(tdy,PressureSine);CHKERRQ(ierr);
      ierr = TDySetDirichletFlux(tdy,VelocitySine);CHKERRQ(ierr);
      break;
    }      
  }else{
    ierr = TDySetPermeabilityTensor(tdy,Permeability3D);CHKERRQ(ierr);
    ierr = TDySetForcingFunction(tdy,Forcing3D);CHKERRQ(ierr);
    ierr = TDySetDirichletFunction(tdy,Pressure3D);CHKERRQ(ierr);
    ierr = TDySetDirichletFlux(tdy,Velocity3D);CHKERRQ(ierr);
  }
  ierr = TDySetDiscretizationMethod(tdy,WY);CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy);CHKERRQ(ierr);
  
  /* Compute system */
  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K);CHKERRQ(ierr);
  ierr = TDyComputeSystem(tdy,K,F);CHKERRQ(ierr);

  /* Solve system */
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U);CHKERRQ(ierr);

  /* Evaluate error norms */
  PetscReal normp,normv,normd;
  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv,&normd);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%e %e %e\n",normp,normv,normd);CHKERRQ(ierr);
    
  /* Cleanup */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy);CHKERRQ(ierr); 
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
