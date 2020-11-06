#include <private/tdycoreimpl.h>
#include <private/tdydmimpl.h>
#include <private/tdyrichardsimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdythimpl.h>
#include <tdytimers.h>
#include <private/tdypermeabilityimpl.h>

PetscErrorCode TDyDriverInitializeTDy(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyEnterProfilingStage("TDycore Setup");
  TDY_START_FUNCTION_TIMER()
  PetscReal gravity[3] = {0.,0.,0.};
  TS ts;
  SNES snes;
  SNESLineSearch linesearch;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  if (dim != 3) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Driver currently only supports 3D");
  }
  gravity[dim-1] = 9.8068;
  ierr = TDySetGravityVector(tdy,gravity);

  switch(tdy->method) {
    case TPF:
    case MPFA_O:
      break;
    case MPFA_O_DAE:
    case MPFA_O_TRANSIENTVAR:
    case BDM:
    case WY:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Driver not supported for specified method.");
      break;
  }

  ierr = TDySetPorosityFunction(tdy,TDyPorosityFunctionDefault,PETSC_NULL);
         CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(tdy,TDyPermeabilityFunctionDefault,
                                    PETSC_NULL); CHKERRQ(ierr);
  if (tdy->mode == TH) {
    ierr = TDySetThermalConductivityFunction(tdy,
                                         TDyThermalConductivityFunctionDefault,
                                         PETSC_NULL); CHKERRQ(ierr);
    ierr = TDySetSoilDensity(tdy,TDySoilDensityFunctionDefault); CHKERRQ(ierr);
    ierr = TDySetSoilSpecificHeatCapacity(tdy,TDySpecificSoilHeatCapacityFunctionDefault); CHKERRQ(ierr);
  }

  ierr = TDySetupNumericalMethods(tdy); CHKERRQ(ierr);

  ierr = TDyTimeIntegratorCreate(&tdy->ti); CHKERRQ(ierr);
  ierr = TDyCreateVectors(tdy); CHKERRQ(ierr);
  ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);

  // check for unsupported modes
  switch (tdy->mode) {
    case RICHARDS:
    case TH:
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unrecognized flow mode.");
  }
  // check for unsupported time integration methods
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      switch (tdy->mode) {
        case RICHARDS:
          break;
        case TH:
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"SNES not supported for TH mode.");
          break;
      }
      break;
    case TDyTS:
      break;
    default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unrecognized time integration method.");
  }

  // create time integrator 
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ierr = SNESCreate(PETSC_COMM_WORLD,&snes);
             CHKERRQ(ierr);
      ierr = TDySetSNESFunction(snes,tdy); CHKERRQ(ierr);
      ierr = TDySetSNESJacobian(snes,tdy); CHKERRQ(ierr);
      ierr = SNESGetLineSearch(snes,&linesearch);
             CHKERRQ(ierr);
      tdy->ti->snes = snes;
      break;
    case TDyTS:
      ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
//      ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
//      ierr = TSSetType(ts,TSPSEUDO); CHKERRQ(ierr);
//      ierr = TSPseudoSetTimeStep(ts,TSPseudoTimeStepDefault,NULL); CHKERRQ(ierr);
      ierr = TSSetEquationType(ts,TS_EQ_IMPLICIT); CHKERRQ(ierr);
      ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
      ierr = TSSetDM(ts,tdy->dm); CHKERRQ(ierr);
      ierr = TSSetSolution(ts,tdy->solution); CHKERRQ(ierr);
      ierr = TDySetIFunction(ts,tdy); CHKERRQ(ierr);
      ierr = TDySetIJacobian(ts,tdy); CHKERRQ(ierr);
      ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
      ierr = SNESGetLineSearch(snes,&linesearch);
             CHKERRQ(ierr);
      ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);
             CHKERRQ(ierr);
      tdy->ti->ts = ts;
      break;
  }
  // time integrator settings for all modes
//  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);
//         CHKERRQ(ierr);
  // mode specific time integrator settings
  switch (tdy->mode) {
    case RICHARDS:
      ierr = SNESLineSearchSetPostCheck(linesearch,TDyRichardsSNESPostCheck,
                                        &tdy); CHKERRQ(ierr);
      ierr = SNESSetConvergenceTest(snes,TDyRichardsConvergenceTest,
                                    &tdy,NULL); CHKERRQ(ierr);
      switch(tdy->ti->time_integration_method) {
        case TDySNES:
          break;
        case TDyTS:
          ierr = TSSetPostStep(ts,TDyRichardsTSPostStep); CHKERRQ(ierr);
          break;
      }
      ierr = TDyRichardsInitialize(tdy); CHKERRQ(ierr);
      break;
    case TH:
      ierr = SNESLineSearchSetPostCheck(linesearch,TDyTHSNESPostCheck,
                                        &tdy); CHKERRQ(ierr);
      ierr = SNESSetConvergenceTest(snes,TDyTHConvergenceTest,
                                    &tdy,NULL); CHKERRQ(ierr);
      switch(tdy->ti->time_integration_method) {
        case TDySNES:
          break;
        case TDyTS:
          ierr = TSSetPostStep(ts,TDyTHTSPostStep); CHKERRQ(ierr);
      }
      ierr = TDyTHInitialize(tdy); CHKERRQ(ierr);
      break;
  }
  PetscPrintf(PETSC_COMM_WORLD,"tdy->ti->time_integration_method = %d\n",
              tdy->ti->time_integration_method);
  // finish set of time integrators
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
      ierr = TDySetPreviousSolutionForSNESSolver(tdy,tdy->solution);
             CHKERRQ(ierr);
      break;
    case TDyTS:
      ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
      break;
  }
  TDY_STOP_FUNCTION_TIMER()
  TDyExitProfilingStage("TDycore Setup");
  PetscFunctionReturn(0);
}
