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

  ierr = TimestepperCreate(&tdy->ts); CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&tdy->ts->snes); CHKERRQ(ierr);
  ierr = TDySetSNESFunction(tdy->ts->snes,tdy); CHKERRQ(ierr);
  ierr = TDySetSNESJacobian(tdy->ts->snes,tdy); CHKERRQ(ierr);
  SNESLineSearch linesearch;
  ierr = SNESGetLineSearch(tdy->ts->snes,&linesearch); CHKERRQ(ierr);
  ierr = SNESLineSearchSetPostCheck(linesearch,TDyRichardsSNESPostCheck,&tdy);
         CHKERRQ(ierr);
  ierr = SNESSetFromOptions(tdy->ts->snes); CHKERRQ(ierr);

  switch (tdy->mode) {
    case RICHARDS:
      ierr = TDyRichardsInitialize(tdy); CHKERRQ(ierr);
      break;
    case TH:
      break;
  }
  ierr = TDySetInitialSolutionForSNESSolver(tdy,tdy->U); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
