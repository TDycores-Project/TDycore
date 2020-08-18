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
