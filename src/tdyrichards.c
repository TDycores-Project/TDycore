#include <private/tdycoreimpl.h>
#include <tdyrichards.h>

PetscErrorCode TDyRichardsInitialize(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"Running Richards mode.\n");
  PetscRandom rand;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand); CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,1.e4,1.e6); CHKERRQ(ierr);
  ierr = VecSetRandom(tdy->solution,rand); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand); CHKERRQ(ierr);

#if 0
  PetscReal values[8] = {722832., 71176., 19922.1, 153193., 403800., 82306.2, 112828., 258273.};
  PetscInt indices[8] = {0,1,2,3,4,5,6,7};
  ierr = VecSetValues(tdy->solution,8,indices,values,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(tdy->solution); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(tdy->solution); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRichardsTSPostStep(TS ts) {
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

PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch linesearch,
                                        Vec X, Vec Y, Vec W,
                                        PetscBool *changed_Y,
                                        PetscBool *changed_W,void *ctx) {
  PetscErrorCode ierr;
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

PetscErrorCode TDyRichardsConvergenceTest(SNES snes, PetscInt it,
                 PetscReal xnorm, PetscReal unorm, PetscReal fnorm,
                 SNESConvergedReason *reason, void *ctx) {
  TDy tdy = (TDy)ctx;
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