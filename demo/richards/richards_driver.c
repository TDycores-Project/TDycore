#include "richards.h"

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code;
  Richards *r = NULL;

  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options",""); 
                           CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code",
                         "Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = RichardsCreate(&r); CHKERRQ(ierr);
  ierr = RichardsDestroy(&r); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
