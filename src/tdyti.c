#include <private/tdycoreimpl.h>
#include <tdycore.h>
#include <private/tdyioimpl.h>
#include <tdytimers.h>
#include <private/tdytiimpl.h>

PetscErrorCode TDyTimeIntegratorCreate(TDyTimeIntegrator *_ti) {
  TDyTimeIntegrator ti;
  char time_integration_method[32];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ti = (TDyTimeIntegrator)malloc(sizeof(struct _p_TDyTimeIntegrator));
  *_ti =ti;

  ti->time_integration_method = TDySNES;
  ti->dt_init = 1.;
  ti->dt_max = 1.e20;
  ti->dt_growth_factor = 1.25;
  ti->dt_reduction_factor = 0.5;
  ti->time = 0.;
  ti->dt_save = 0.;
  PetscReal final_time_in_years = 0.0001;
  ti->final_time = final_time_in_years*365.*24.*3600.;
  ti->istep = 0;

  time_integration_method[0] = '\0';
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
  ierr = PetscOptionsString("-time_integration_method",
                            "Time Integration Method","",
                            time_integration_method,time_integration_method,32,
                            NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  size_t len;
  ierr = PetscStrlen(time_integration_method,&len); CHKERRQ(ierr);
  if (len) {
    if (!strcmp(time_integration_method,"SNES"))
      ti->time_integration_method = TDySNES;
    else if (!strcmp(time_integration_method,"TS"))
      ti->time_integration_method = TDyTS;
    else
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unrecognized Time Integration Method");
  }
  ti->dt = ti->dt_init;

  switch(ti->time_integration_method) {
    case TDySNES:
      PetscPrintf(PETSC_COMM_WORLD,"Using TDycore backward Euler (and PETSc SNES) for time integration.\n");
      break;
    case TDyTS:
      PetscPrintf(PETSC_COMM_WORLD,"Using PETSc TS for time integration.\n");
      break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorUpdateDT(TDyTimeIntegrator ti, PetscReal sync_time) {
  PetscFunctionBegin;
  if (ti->dt_save > 0.) {
    ti->time = sync_time;
    ti->dt = ti->dt_save;
    ti->dt_save = 0.;
  }
  ti->dt *= ti->dt_growth_factor;
  if (ti->dt > ti->dt_max) ti->dt = ti->dt_max;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorSetTargetTime(TDyTimeIntegrator ti, PetscReal sync_time) {
  PetscFunctionBegin;
  PetscReal dt_for_sync = sync_time-ti->time;
  if (ti->dt > dt_for_sync) {
    ti->dt_save = ti->dt;
    ti->dt = dt_for_sync; 
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorSetTimeStep(TDy tdy, PetscReal dt) {
  PetscFunctionBegin;
  tdy->ti->dt_init = dt;
  tdy->ti->dt = dt;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorRunToTime(TDy tdy,PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  TDyTimeIntegrator ti;
  SNESConvergedReason reason;

  switch(tdy->ti->time_integration_method) {
    case TDySNES:
      ti = tdy->ti;

      MPI_Comm comm;
      PetscMPIInt rank;
      ierr = PetscObjectGetComm((PetscObject)ti->snes, &comm); CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

      while (ti->time < sync_time) {
        ierr = TDyTimeIntegratorSetTargetTime(ti,sync_time); CHKERRQ(ierr);
        ierr = TDySetDtimeForSNESSolver(tdy,ti->dt); CHKERRQ(ierr);
        if (!rank){
          switch (tdy->mode){
            case RICHARDS:
              printf("===== RICHARDS MODE ==============================\n");
              break;
            case TH:
              printf("===== TH MODE ====================================\n");
              break;
          }
        }
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
        ierr = SNESGetConvergedReason(ti->snes,&reason); CHKERRQ(ierr);
        if (tdy->io->io_process)
          printf("\nTime step %d: time = %f dt = %f ni = %d li = %d rsn = %s\n\n",
                 ti->istep,ti->time,ti->dt,nit,lit,
                 SNESConvergedReasons[reason]);
        if (tdy->io->print_intermediate) {
          ierr = TDyIOWriteVec(tdy); 
          CHKERRQ(ierr);
        }
        ierr = TDyTimeIntegratorUpdateDT(ti,sync_time); CHKERRQ(ierr);
      }
      break;
    case TDyTS:
      ti = tdy->ti;
      PetscReal delta_time = sync_time - ti->time;
      ierr = TSSetTimeStep(ti->ts,ti->dt_init); CHKERRQ(ierr);
      ierr = TSSetMaxTime(ti->ts,delta_time); CHKERRQ(ierr);
      ierr = TSSolve(ti->ts,tdy->solution); CHKERRQ(ierr);
      ierr = TSGetTime(ti->ts,&ti->time); CHKERRQ(ierr);
      break;
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimeIntegratorOutputRegression(TDy tdy) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  ierr = TDyOutputRegression(tdy,tdy->solution); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode TDyTimeIntegratorDestroy(TDyTimeIntegrator *ti) {
  PetscFunctionBegin;
  free(*ti);
  ti = PETSC_NULL;
  PetscFunctionReturn(0);
}

