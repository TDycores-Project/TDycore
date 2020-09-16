#include <private/tdycoreimpl.h>
#include <tdycore.h>
#include <tdyti.h>
#include <tdyio.h>

PetscErrorCode TDyTimeIntegratorCreate(TDyTimeIntegrator *_ti) {
  TDyTimeIntegrator ti;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ti = (TDyTimeIntegrator)malloc(sizeof(struct TDyTimeIntegrator));
  *_ti =ti;

  ti->time_integration_method = TDySNES;
  ti->dt_init = 1.;
  ti->dt_max = 1.e20;
  ti->dt_growth_factor = 1.25;
  ti->dt_reduction_factor = 0.5;
  ti->dt = ti->dt_init;
  ti->time = 0.;
  PetscScalar final_time_in_years = 0.0001;
  ti->final_time = final_time_in_years*365.*24.*3600.;
  ti->istep = 0;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Time Integration Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsReal("-final_time","Final Time","",ti->final_time,
                          &ti->final_time,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_max","Maximum Timestep Size","",ti->dt_max,
                          &ti->dt_max,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_init","Initial Timestep Size","",ti->dt_init,
                          &ti->dt_init,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_growth_factor","Timestep Growth Factor",
                          "",ti->dt_growth_factor,
                          &ti->dt_growth_factor,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_reduction_factor","Timestep Reduction Factor",
                          "",ti->dt_reduction_factor,
                          &ti->dt_reduction_factor,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorUpdateDT(TDyTimeIntegrator ti, PetscReal sync_time) {
  PetscFunctionBegin;
  ti->dt *= ti->dt_growth_factor;
  if (ti->dt > ti->dt_max) ti->dt = ti->dt_max;
  PetscInt dt_for_sync = sync_time-ti->time;
  if (ti->dt > dt_for_sync) {
    ti->dt = dt_for_sync; 
    ti->time = sync_time;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorRunToTime(TDy tdy,PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyTimeIntegrator ti;
  SNESConvergedReason reason;

  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ti = tdy->ti;
      while (ti->time < sync_time) {
        ierr = TDySetDtimeForSNESSolver(tdy,ti->dt); CHKERRQ(ierr);
        ierr = TDyPreSolveSNESSolver(tdy); CHKERRQ(ierr);
        ierr = SNESSolve(ti->snes,PETSC_NULL,tdy->solution); 
               CHKERRQ(ierr);
        ierr = TDyPostSolveSNESSolver(tdy,tdy->solution); CHKERRQ(ierr);
        ti->time += ti->dt;
        ti->istep++;
        PetscInt nit, lit;
        ierr = SNESGetIterationNumber(ti->snes,&nit); CHKERRQ(ierr);
        ierr = SNESGetLinearSolveIterations(ti->snes,&lit); 
               CHKERRQ(ierr);
        if (tdy->io->io_process)
        if (tdy->io->io_process)
          printf("Time step %d: time = %f dt = %f ni = %d li = %d rsn = %s\n",
                 ti->istep,ti->time,ti->dt,nit,lit,
                 SNESConvergedReasons[reason]);
        if (tdy->io->print_intermediate) {
          ierr = TDyIOPrintVec(tdy->solution,"soln",ti->istep); 
          CHKERRQ(ierr);
        }
        ierr = TDyTimeIntegratorUpdateDT(ti,sync_time); CHKERRQ(ierr);
      }
      break;
    case TDyTS:
      break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorDestroy(TDyTimeIntegrator *ti) {
  PetscFunctionBegin;
  free(*ti);
  ti = PETSC_NULL;
  PetscFunctionReturn(0);
}

