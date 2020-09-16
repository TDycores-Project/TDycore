#include <tdyrichards.h>

PetscErrorCode TDyRichardsInitialize(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscRandom rand;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand); CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,1.e4,1.e6); CHKERRQ(ierr);
  ierr = VecSetRandom(tdy->solution,rand); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch linesearch,
                                        Vec X, Vec Y, Vec W,
                                        PetscBool *changed_Y,
                                        PetscBool *changed_W,void *ctx) {
  PetscFunctionBegin;
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
