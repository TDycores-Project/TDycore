#include "tdycore.h"
#include "tdydmperturbation.h"

void Porosity(double *x,double *theta) {
  (*theta) = 0.5;
}

void Permeability(double *x,double *K) {
  (*K) = 1e-10;
}

PetscErrorCode Pressure(TDy tdy,double *x,double *p,void *ctx) {
  (*p) = 91325;
  PetscFunctionReturn(0);
}

PetscErrorCode Forcing(TDy tdy,double *x,double *f,void *ctx) {
  (*f) = 0;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code=0;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
			   "Transient Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = SetVertexPerturbationFunction(PerturbVerticesRandom); CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  DM dm;
  ierr = TDyGetDM(tdy,&dm); CHKERRQ(ierr);
  ierr = TDySetPorosity(tdy,Porosity); CHKERRQ(ierr);
  ierr = TDySetPermeabilityScalar(tdy,Permeability); CHKERRQ(ierr);
  ierr = TDySetForcingFunction(tdy,Forcing,NULL); CHKERRQ(ierr);
  ierr = TDySetDirichletValueFunction(tdy,Pressure,NULL); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Setup initial condition */
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = VecSet(U,91325); CHKERRQ(ierr);
  
  /* Create time stepping and solve */
  TS  ts;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,TDyWYResidual,tdy); CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm); CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U); CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  ierr = TSSetUp(ts); CHKERRQ(ierr);
  ierr = TSSolve(ts,U); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U); CHKERRQ(ierr);

  /* Cleanup */
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}
