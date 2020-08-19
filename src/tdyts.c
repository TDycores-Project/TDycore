#include <private/tdycoreimpl.h>
#include <tdycore.h>
#include <tdyts.h>
#include <tdyio.h>

PetscErrorCode TDyTimestepperCreate(TDyTimestepper *_timestepper) {
  TDyTimestepper timestepper;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  timestepper = (TDyTimestepper)malloc(sizeof(struct TDyTimestepper));
  *_timestepper =timestepper;

  timestepper->dt_init = 1.;
  timestepper->dt_max = 1.e20;
  timestepper->dt_growth_factor = 1.25;
  timestepper->dt_reduction_factor = 0.5;
  timestepper->dt = timestepper->dt_init;
  timestepper->time = 0.;
  PetscScalar final_time_in_years = 0.0001;
  timestepper->final_time = final_time_in_years*365.*24.*3600.;
  timestepper->istep = 0;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Timestepper Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsReal("-final_time","Final Time","",timestepper->final_time,
                          &timestepper->final_time,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_max","Maximum Timestep Size","",timestepper->dt_max,
                          &timestepper->dt_max,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_init","Initial Timestep Size","",timestepper->dt_init,
                          &timestepper->dt_init,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_growth_factor","Timestep Growth Factor",
                          "",timestepper->dt_growth_factor,
                          &timestepper->dt_growth_factor,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_reduction_factor","Timestep Reduction Factor",
                          "",timestepper->dt_reduction_factor,
                          &timestepper->dt_reduction_factor,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimestepperUpdateDT(TDyTimestepper timestepper,
                                      PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  timestepper->dt *= timestepper->dt_growth_factor;
  if (timestepper->dt > timestepper->dt_max) timestepper->dt = timestepper->dt_max;
  PetscInt dt_for_sync = sync_time-timestepper->time;
  if (timestepper->dt > dt_for_sync) {
    timestepper->dt = dt_for_sync; 
    timestepper->time = sync_time;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimestepperRunToTime(TDy tdy,PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscViewer viewer;
  PetscScalar *a;

  TDyTimestepper timestepper = tdy->timestepper;

  while (timestepper->time < sync_time) {
    ierr = TDySetDtimeForSNESSolver(tdy,timestepper->dt); CHKERRQ(ierr);
    ierr = TDyPreSolveSNESSolver(tdy); CHKERRQ(ierr);
    ierr = SNESSolve(timestepper->snes,PETSC_NULL,tdy->solution); CHKERRQ(ierr);
    ierr = TDyPostSolveSNESSolver(tdy,tdy->solution); CHKERRQ(ierr);
    timestepper->time += timestepper->dt;
    timestepper->istep++;
    PetscInt nit;
    ierr = SNESGetLinearSolveIterations(timestepper->snes,&nit); CHKERRQ(ierr);
    if (tdy->io->io_process)
      printf("Time step %d: time = %f dt = %f ni=%d\n",
             timestepper->istep,timestepper->time,timestepper->dt,nit);
    if (tdy->io->print_intermediate)
      ierr = TDyIOPrintVec(tdy->solution,"soln",timestepper->istep); CHKERRQ(ierr);
    ierr = TDyTimestepperUpdateDT(timestepper,sync_time); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyTimestepperDestroy(TDyTimestepper *timestepper) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  free(*timestepper);
  timestepper = PETSC_NULL;
  PetscFunctionReturn(0);
}

