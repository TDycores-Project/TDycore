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

  ierr = TDyTimeIntegratorCreate(&tdy->ti); CHKERRQ(ierr);
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ierr = SNESCreate(PETSC_COMM_WORLD,&tdy->ti->snes); 
             CHKERRQ(ierr);
      ierr = TDySetSNESFunction(tdy->ti->snes,tdy); CHKERRQ(ierr);
      ierr = TDySetSNESJacobian(tdy->ti->snes,tdy); CHKERRQ(ierr);
      SNESLineSearch linesearch;
      ierr = SNESGetLineSearch(tdy->ti->snes,&linesearch); 
             CHKERRQ(ierr);
      ierr = SNESLineSearchSetPostCheck(linesearch,TDyRichardsSNESPostCheck,
                                        &tdy); CHKERRQ(ierr);
      ierr = SNESSetFromOptions(tdy->ti->snes); CHKERRQ(ierr);
      break;
    case TDyTS:
      break;
    default:
      if (tdy->io->io_process) {
        printf("Unrecognized time integration method.\n"); 
        exit(1);
      }
  }

  switch (tdy->mode) {
    case RICHARDS:
      ierr = TDyRichardsInitialize(tdy); CHKERRQ(ierr);
      break;
    case TH:
      break;
    default:
      if (tdy->io->io_process) {
        printf("Unrecognized flow mode.\n"); 
        exit(1);
      }
  }
  ierr = TDySetInitialSolutionForSNESSolver(tdy,tdy->solution); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
