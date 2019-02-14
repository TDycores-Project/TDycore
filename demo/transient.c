#include "tdycore.h"

void Porosity(double *x,double *theta){
  (*theta) = 0.5;
}

void Permeability(double *x,double *K){
  (*K) = 1;
}

void Pressure(double *x,double *p){
  (*p) = 1;
}

void Forcing(double *x,double *f){
  (*f) = 0;
}

int main(int argc, char **argv)
{
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2;
  ierr = PetscInitialize(&argc,&argv,(char*)0,0);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Transient Options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D","",N,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  const PetscInt  faces[3] = {N  ,N  };
  const PetscReal lower[3] = {0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);  
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy);CHKERRQ(ierr);
  ierr = TDySetPorosity(tdy,Porosity);CHKERRQ(ierr);
  ierr = TDySetPermeabilityScalar(tdy,Permeability);CHKERRQ(ierr);
  ierr = TDySetForcingFunction(tdy,Forcing);CHKERRQ(ierr);
  ierr = TDySetDirichletFunction(tdy,Pressure);CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,WHEELER_YOTOV);CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy);CHKERRQ(ierr);
  
  /* Setup initial condition */
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);

  /* Create time stepping and solve */
  TS  ts;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,TDyWYResidual,tdy);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* Cleanup */
  ierr = TDyDestroy(&tdy);CHKERRQ(ierr); 
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
