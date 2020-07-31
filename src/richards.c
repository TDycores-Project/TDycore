#include "richards.h"

PetscErrorCode RichardsCreate(Richards **r) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
#if 0
  *r = (int*)malloc(sizeof(int));
  **r = 9;
#else
  *r = (Richards*) malloc(sizeof(Richards));
  (*r)->i = 9;
  ierr = TDyCreate(&((*r)->tdy)); CHKERRQ(ierr);
  DM dm;
  ierr = TDyGetDM((*r)->tdy,&dm); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&((*r)->U)); CHKERRQ(ierr);
  ierr = VecDuplicate((*r)->U,&((*r)->F)); CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,&((*r)->J)); CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode RichardsRunToTime(Richards r,PetscReal time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}


PetscErrorCode RichardsDestroy(Richards **r) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy(&((*r)->U)); CHKERRQ(ierr);
  ierr = VecDestroy(&((*r)->F)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*r)->J)); CHKERRQ(ierr);
  ierr = TDyDestroy(&((*r)->tdy)); CHKERRQ(ierr);
  printf("%d\n",(*(r))->i);
  free(*r);
  *r = NULL;
  PetscFunctionReturn(0);
}
