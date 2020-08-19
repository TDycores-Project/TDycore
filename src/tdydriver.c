#include <tdydriver.h>

PetscErrorCode TDyDriverInitializeTDy(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal gravity[3] = {0.,0.,0.};

  DM dm;
  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr); 
  gravity[dim-1] = 9.8068;
  ierr = TDySetGravityVector(tdy,gravity);
  ierr = TDySetPorosityFunction(tdy,TDyPorosityFunctionDefault,PETSC_NULL);
         CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(tdy,TDyPermeabilityFunctionDefault,PETSC_NULL);
         CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  ierr = TDyTimestepperCreate(&tdy->timestepper); CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&tdy->timestepper->snes); CHKERRQ(ierr);
  ierr = TDySetSNESFunction(tdy->timestepper->snes,tdy); CHKERRQ(ierr);
  ierr = TDySetSNESJacobian(tdy->timestepper->snes,tdy); CHKERRQ(ierr);
  SNESLineSearch linesearch;
  ierr = SNESGetLineSearch(tdy->timestepper->snes,&linesearch); CHKERRQ(ierr);
  ierr = SNESLineSearchSetPostCheck(linesearch,TDyRichardsSNESPostCheck,&tdy);
         CHKERRQ(ierr);
  ierr = SNESSetFromOptions(tdy->timestepper->snes); CHKERRQ(ierr);

  switch (tdy->mode) {
    case RICHARDS:
      ierr = TDyRichardsInitialize(tdy); CHKERRQ(ierr);
      break;
    case TH:
      break;
  }
  ierr = TDySetInitialSolutionForSNESSolver(tdy,tdy->solution); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
