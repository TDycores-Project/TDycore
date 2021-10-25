#include <private/tdycoreimpl.h>
#include <private/tdydmimpl.h>
#include <private/tdyrichardsimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdythimpl.h>
#include <private/tdyioimpl.h>
#include <tdytimers.h>
#include <private/tdycharacteristiccurvesimpl.h>

PetscErrorCode TDyDriverInitializeTDy(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyEnterProfilingStage("TDycore Setup");
  TDY_START_FUNCTION_TIMER()
  TS ts;
  SNES snes;
  SNESLineSearch linesearch;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  DM dm;
  ierr = TDyGetDM(tdy,&dm); CHKERRQ(ierr);

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  if (dim != 3) {
    SETERRQ(comm,PETSC_ERR_USER,"Driver currently only supports 3D");
  }

  switch(tdy->options.discretization) {
    case MPFA_O:
      break;
    case MPFA_O_DAE:
    case MPFA_O_TRANSIENTVAR:
    case BDM:
    case WY:
      SETERRQ(comm,PETSC_ERR_USER,"Driver not supported for specified discretization.");
      break;
  }

  if (!TDyIsPorositySet(tdy)) {
    size_t len;
    ierr = PetscStrlen(tdy->io->porosity_filename, &len); CHKERRQ(ierr);
    if (!len){
      ierr = TDySetPorosityFunction(tdy,TDyConstantPorosityFunction,PETSC_NULL);CHKERRQ(ierr);
    } else {
      ierr = TDyIOReadPorosity(tdy);CHKERRQ(ierr);
    }
  }

  if (!TDyIsPermeabilitySet(tdy)){
    size_t len;
    ierr = PetscStrlen(tdy->io->permeability_filename, &len); CHKERRQ(ierr);
    if (!len){
      ierr = TDySetPermeabilityFunction(tdy,TDyConstantPermeabilityFunction,PETSC_NULL); CHKERRQ(ierr);     
    } else {
      ierr = TDyIOReadPermeability(tdy);CHKERRQ(ierr);
    }
  }

  if (tdy->options.mode == TH) {

    if (!TDyIsThermalConductivytSet(tdy)) {
      ierr = TDySetThermalConductivityFunction(tdy,TDyConstantThermalConductivityFunction,PETSC_NULL); CHKERRQ(ierr);
    }

    if (!TDyIsSoilDensitySet(tdy)) {
      //ierr = TDySetSoilDensity(tdy,TDyConstantSoilDensityFunction); CHKERRQ(ierr);
      ierr = TDySetSoilDensityFunction(tdy,TDyConstantSoilDensityFunction,PETSC_NULL); CHKERRQ(ierr);
    }

    if (!TDyIsSoilSpecificHeatSet(tdy)) {
      ierr = TDySetSoilSpecificHeatFunction(tdy,TDyConstantSoilSpecificHeatFunction,PETSC_NULL); CHKERRQ(ierr);
    }
  }

  ierr = TDySetup(tdy); CHKERRQ(ierr);

  ierr = TDyTimeIntegratorCreate(&tdy->ti); CHKERRQ(ierr);
  ierr = TDyCreateVectors(tdy); CHKERRQ(ierr);
  ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);

  // check for unsupported modes
  switch (tdy->options.mode) {
    case RICHARDS:
    case TH:
      break;
    default:
      SETERRQ(comm,PETSC_ERR_USER,"Unrecognized flow mode.");
  }
  // check for unsupported time integration methods
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      switch (tdy->options.mode) {
        case RICHARDS:
          break;
        case TH:
          SETERRQ(comm,PETSC_ERR_USER,"SNES not supported for TH mode.");
          break;
      }
      break;
    case TDyTS:
      break;
    default:
        SETERRQ(comm,PETSC_ERR_USER,"Unrecognized time integration method.");
  }

  // create time integrator
  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ierr = SNESCreate(comm,&snes);
             CHKERRQ(ierr);
      ierr = TDySetSNESFunction(snes,tdy); CHKERRQ(ierr);
      ierr = TDySetSNESJacobian(snes,tdy); CHKERRQ(ierr);
      ierr = SNESGetLineSearch(snes,&linesearch);
             CHKERRQ(ierr);
      tdy->ti->snes = snes;
      break;
    case TDyTS:
      ierr = TSCreate(comm,&ts); CHKERRQ(ierr);
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
  switch (tdy->options.mode) {
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
      // FIXME: This is a different path from the one used by TDycore proper.
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
      // FIXME: This is a different path from the one used by TDycore proper.
      ierr = TDyTHInitialize(tdy); CHKERRQ(ierr);
      break;
  }
  PetscPrintf(comm,"tdy->ti->time_integration_method = %d\n",
              tdy->ti->time_integration_method);
   size_t len;
   ierr = PetscStrlen(tdy->io->ic_filename, &len); CHKERRQ(ierr);
   if (len){
     TDyIOReadIC(tdy); CHKERRQ(ierr);
   }
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
