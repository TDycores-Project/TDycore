#include <private/tdycoreimpl.h>
#include <tdycore.h>
#include <tdyts.h>
#include <tdyio.h>

PetscErrorCode TimestepperCreate(Timestepper *_ts) {
  Timestepper ts;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ts = (Timestepper)malloc(sizeof(struct Timestepper));
  *_ts = ts;

  ts->dt_init = 1.;
  ts->dt_max = 1.e20;
  ts->dt_growth_factor = 1.25;
  ts->dt_reduction_factor = 0.5;
  ts->dt = ts->dt_init;
  ts->time = 0.;
  PetscScalar final_time_in_years = 0.0001;
  ts->final_time = final_time_in_years*365.*24.*3600.;
  ts->istep = 0;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Timestepper Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsReal("-final_time","Final Time","",ts->final_time,
                          &ts->final_time,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_max","Maximum Timestep Size","",ts->dt_max,
                          &ts->dt_max,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_init","Initial Timestep Size","",ts->dt_init,
                          &ts->dt_init,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_growth_factor","Timestep Growth Factor",
                          "",ts->dt_growth_factor,
                          &ts->dt_growth_factor,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt_reduction_factor","Timestep Reduction Factor",
                          "",ts->dt_reduction_factor,
                          &ts->dt_reduction_factor,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TimestepperUpdateDT(Timestepper ts) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode TimestepperRunToTime(TDy tdy,PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscViewer viewer;
  PetscScalar *a;

  Timestepper ts = tdy->ts;

  while (ts->time < sync_time) {
    ierr = TDySetDtimeForSNESSolver(tdy,ts->dt); CHKERRQ(ierr);
    ierr = TDyPreSolveSNESSolver(tdy); CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,PETSC_NULL,tdy->U); CHKERRQ(ierr);
    ierr = TDyPostSolveSNESSolver(tdy,tdy->U); CHKERRQ(ierr);
    ts->time += ts->dt;
    ts->dt *= ts->dt_growth_factor;
    if (ts->dt > ts->dt_max) ts->dt = ts->dt_max;
    PetscInt dt_for_sync = sync_time-ts->time;
    if (ts->dt > dt_for_sync) {
      ts->dt = dt_for_sync; 
      ts->time = sync_time;
    }
    ts->istep++;
    PetscInt nit;
    ierr = SNESGetLinearSolveIterations(ts->snes,&nit); CHKERRQ(ierr);
    if (tdy->io->io_process)
      printf("Time step %d: time = %f dt = %f ni=%d\n",
             ts->istep,ts->time,ts->dt,nit);
    if (tdy->io->print_intermediate)
      ierr = PrintVec(tdy->U,"soln",ts->istep); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TimestepperDestroy(Timestepper *ts) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  free(*ts);
  ts = PETSC_NULL;
  PetscFunctionReturn(0);
}

