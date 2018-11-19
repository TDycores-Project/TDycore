#include "tdycore.h"

void Permeability(double *x,double *K){
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

void Pressure(double *x,double *f){
  (*f)  = PetscPowReal(1-x[0],4);
  (*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
  (*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
}

void Forcing(double *x,double *f){
  double K[4];
  Permeability(x,K);
  (*f)  = -K[0]*(12*PetscPowReal(1-x[0],2)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  (*f) += -K[1]*( 3*PetscPowReal(1-x[1],2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[2]*( 3*PetscPowReal(1-x[1],2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[3]*(-6*(1-x[0])*(x[1]-1)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
}

int main(int argc, char **argv)
{
  /* Initialize */
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,0);CHKERRQ(ierr);
  
  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  const PetscInt  faces[2] = {8  ,8  };
  const PetscReal lower[2] = {0.0,0.0};
  const PetscReal upper[2] = {1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,2,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);  
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy);CHKERRQ(ierr);
  ierr = TDySetPermeabilityTensor(dm,tdy,Permeability);CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(dm,tdy,WHEELER_YOTOV);CHKERRQ(ierr);
  ierr = TDySetForcingFunction(tdy,Forcing);CHKERRQ(ierr);
  ierr = TDySetDirichletFunction(tdy,Pressure);CHKERRQ(ierr);
  
  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K);CHKERRQ(ierr);
  ierr = TDyComputeSystem(dm,tdy,K,F);CHKERRQ(ierr);
    
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy);CHKERRQ(ierr); 
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
