#include "richards.h"

PetscErrorCode RichardsCreate(Richards *r) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *r = (Richards) malloc(sizeof(struct Richards));
  ierr = TDyCreate(&((*r)->tdy)); CHKERRQ(ierr);
  DM dm;
  ierr = TDyGetDM((*r)->tdy,&dm); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&((*r)->U)); CHKERRQ(ierr);
  ierr = VecDuplicate((*r)->U,&((*r)->F)); CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,&((*r)->J)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RichardsRunToTime(Richards r,PetscReal time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"J.mat",&viewer); 
         CHKERRQ(ierr);
  ierr = MatView(r->J,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  printf("here\n");
  PetscFunctionReturn(0);
}


PetscErrorCode RichardsDestroy(Richards *r) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy(&((*r)->U)); CHKERRQ(ierr);
  ierr = VecDestroy(&((*r)->F)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*r)->J)); CHKERRQ(ierr);
  ierr = TDyDestroy(&((*r)->tdy)); CHKERRQ(ierr);
  free(*r);
  *r = NULL;
  PetscFunctionReturn(0);
}
