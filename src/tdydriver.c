#include <private/tdycoreimpl.h>
#include <tdydriver.h>
#include <tdypermeability.h>
#include <tdyporosity.h>
#include <tdyrichards.h>
#include <tdytimers.h>

PetscErrorCode TDyDriverInitializeTDy(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyEnterProfilingStage("TDycore Setup");
  TDY_START_FUNCTION_TIMER()
  PetscReal gravity[3] = {0.,0.,0.};
  SNESLineSearch linesearch;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  if (dim != 3) {
    PetscPrintf(PETSC_COMM_WORLD,"Richards mode currently only supports 3D\n");
    exit(0);
  }
  gravity[dim-1] = 9.8068;
  ierr = TDySetGravityVector(tdy,gravity);
  ierr = TDySetPorosityFunction(tdy,TDyPorosityFunctionDefault,PETSC_NULL);
         CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(tdy,TDyPermeabilityFunctionDefault,PETSC_NULL);
         CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  ierr = TDyTimeIntegratorCreate(&tdy->ti); CHKERRQ(ierr);
  ierr = TDyCreateVectors(tdy); CHKERRQ(ierr);
  ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);

  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ierr = SNESCreate(PETSC_COMM_WORLD,&tdy->ti->snes);
             CHKERRQ(ierr);
      ierr = TDySetSNESFunction(tdy->ti->snes,tdy); CHKERRQ(ierr);
      ierr = TDySetSNESJacobian(tdy->ti->snes,tdy); CHKERRQ(ierr);
      ierr = SNESGetLineSearch(tdy->ti->snes,&linesearch);
             CHKERRQ(ierr);
/*
      ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);
             CHKERRQ(ierr);
*/
      ierr = SNESLineSearchSetPostCheck(linesearch,TDyRichardsSNESPostCheck,
                                        &tdy); CHKERRQ(ierr);
      ierr = SNESSetConvergenceTest(tdy->ti->snes,TDyRichardsConvergenceTest,
                                    &tdy,NULL); CHKERRQ(ierr);
      ierr = SNESSetFromOptions(tdy->ti->snes); CHKERRQ(ierr);
      break;
    case TDyTS:
      ierr = TSCreate(PETSC_COMM_WORLD,&tdy->ti->ts); CHKERRQ(ierr);
//      ierr = TSSetType(tdy->ti->ts,TSBEULER); CHKERRQ(ierr);
//      ierr = TSSetType(tdy->ti->ts,TSPSEUDO); CHKERRQ(ierr);
//      ierr = TSPseudoSetTimeStep(tdy->ti->ts,TSPseudoTimeStepDefault,NULL); CHKERRQ(ierr);
      ierr = TSSetEquationType(tdy->ti->ts,TS_EQ_IMPLICIT); CHKERRQ(ierr);
      ierr = TSSetProblemType(tdy->ti->ts,TS_NONLINEAR); CHKERRQ(ierr);
      ierr = TSSetDM(tdy->ti->ts,tdy->dm); CHKERRQ(ierr);
      ierr = TSSetSolution(tdy->ti->ts,tdy->solution); CHKERRQ(ierr);
      ierr = TDySetIFunction(tdy->ti->ts,tdy); CHKERRQ(ierr);
      ierr = TDySetIJacobian(tdy->ti->ts,tdy); CHKERRQ(ierr);
      ierr = TSSetPostStep(tdy->ti->ts,TDyRichardsTSPostStep); CHKERRQ(ierr);
      SNES snes;
      ierr = TSGetSNES(tdy->ti->ts,&snes); CHKERRQ(ierr);
      ierr = SNESGetLineSearch(snes,&linesearch);
             CHKERRQ(ierr);
/*
      ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);
             CHKERRQ(ierr);
*/
      ierr = SNESLineSearchSetPostCheck(linesearch,TDyRichardsSNESPostCheck,
                                        &tdy); CHKERRQ(ierr);
      ierr = SNESSetConvergenceTest(snes,TDyRichardsConvergenceTest,&tdy,NULL); 
             CHKERRQ(ierr);
      ierr = TSSetExactFinalTime(tdy->ti->ts,TS_EXACTFINALTIME_MATCHSTEP);
             CHKERRQ(ierr);
      ierr = TSSetFromOptions(tdy->ti->ts); CHKERRQ(ierr);
      if (tdy->io->io_process) {
        printf("TS time integration method not implemented.\n");
        exit(1);
      }
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
      if (tdy->io->io_process) {
        printf("TH flow mode not implemented.\n");
        exit(1);
      }
      break;
    default:
      if (tdy->io->io_process) {
        printf("Unrecognized flow mode.\n");
        exit(1);
      }
  }
  printf("tdy->ti->time_integration_method = %d\n",tdy->ti->time_integration_method);
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ierr = TDySetInitialSolutionForSNESSolver(tdy,tdy->solution);
             CHKERRQ(ierr);
      break;
    case TDyTS:
      ierr = TDySetInitialSolutionForSNESSolver(tdy,tdy->solution);
             CHKERRQ(ierr);
      break;
    default:
      printf("Unable to set initial condition for the time integration method.\n");
      exit(1);
  }
  TDY_STOP_FUNCTION_TIMER()
  TDyExitProfilingStage("TDycore Setup");
  PetscFunctionReturn(0);
}
