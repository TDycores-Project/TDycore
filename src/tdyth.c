#include <private/tdycoreimpl.h>
#include <private/tdythimpl.h>

PetscErrorCode TDyTHInitialize(TDy tdy) {
  PetscErrorCode ierr;
  Vec temp_vec;
  PetscReal *soln_p;
  PetscReal *temp_p;
  PetscInt local_size;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"Running TH mode.\n");

  if (tdy->init_with_random_field) {
    PetscRandom rand;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand); CHKERRQ(ierr);
    ierr = VecGetLocalSize(tdy->solution,&local_size); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(tdy->dm,&temp_vec); CHKERRQ(ierr);
    // pressure
    ierr = PetscRandomSetInterval(rand,1.e4,1.e5); CHKERRQ(ierr);
    ierr = VecSetRandom(temp_vec,rand); CHKERRQ(ierr);
    ierr = VecGetArray(tdy->solution,&soln_p); CHKERRQ(ierr);
    ierr = VecGetArray(temp_vec,&temp_p); CHKERRQ(ierr);
    for (int i=0; i<local_size; i+=2) soln_p[i] = temp_p[i];
    ierr = VecRestoreArray(temp_vec,&temp_p); CHKERRQ(ierr);
    // temperature
    ierr = PetscRandomSetInterval(rand,15.,35.); CHKERRQ(ierr);
    ierr = VecSetRandom(temp_vec,rand); CHKERRQ(ierr);
    ierr = VecGetArray(temp_vec,&temp_p); CHKERRQ(ierr);
    for (int i=1; i<local_size; i+=2) soln_p[i] = temp_p[i];
    ierr = VecRestoreArray(temp_vec,&temp_p); CHKERRQ(ierr);
    ierr = VecRestoreArray(tdy->solution,&soln_p); CHKERRQ(ierr);
    ierr = VecDestroy(&temp_vec); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand); CHKERRQ(ierr);
  }
  else {
    ierr = VecSet(tdy->solution,101325.); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyTHTSPostStep(TS ts) {
  PetscErrorCode ierr;
  PetscReal dt;
  PetscReal time;
  PetscInt istep;
  PetscInt nit;
  PetscInt lit;
  SNES snes;
  SNESConvergedReason reason;
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr); 
  ierr = TSGetTime(ts,&time); CHKERRQ(ierr); 
  ierr = TSGetStepNumber(ts,&istep); CHKERRQ(ierr); 
  ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&nit); CHKERRQ(ierr); ierr = SNESGetLinearSolveIterations(snes,&lit); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason); CHKERRQ(ierr);
  if (!rank)
    printf("Time step %d: time = %f dt = %f ni = %d li = %d rsn = %s\n",
           istep,time,dt,nit,lit,SNESConvergedReasons[reason]);
//           ti->istep,ti->time,ti->dt,nit,lit);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTHSNESPostCheck(SNESLineSearch linesearch,
                                        Vec X, Vec Y, Vec W,
                                        PetscBool *changed_Y,
                                        PetscBool *changed_W,void *ctx) {
  //PetscErrorCode ierr;
  PetscFunctionBegin;

//#define DEBUG
#if defined(DEBUG)
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"post_update.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(Y,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"post_solution.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(W,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode TDyTHConvergenceTest(SNES snes, PetscInt it,
                 PetscReal xnorm, PetscReal unorm, PetscReal fnorm,
                 SNESConvergedReason *reason, void *ctx) {
  //TDy tdy = (TDy)ctx;
  PetscErrorCode ierr;
  Vec r;
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
  ierr = SNESConvergedDefault(snes,it,xnorm,unorm,fnorm,reason,ctx);
         CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&r,NULL,NULL); CHKERRQ(ierr);
#if defined(DEBUG)
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"residual.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(r,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif
  if (!rank) 
    printf("%3d: %9.2e %9.2e %9.2e\n",it,fnorm,xnorm,unorm);
    
  PetscFunctionReturn(0);
}
