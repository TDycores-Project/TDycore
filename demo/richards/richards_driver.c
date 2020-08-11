#include "richards.h"

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code;
  PetscBool print_intermediate = PETSC_FALSE;
  PetscMPIInt rank;
  Richards r = NULL;

  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
  if (!rank) printf("begin\n");
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options",""); 
                           CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code",
                         "Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_intermediate_solutions",
                          "Print intermediate solutions","",
                          print_intermediate,&print_intermediate,NULL);
                          CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = RichardsCreate(&r); CHKERRQ(ierr);
  ierr = RichardsInitialize(r); CHKERRQ(ierr);
  if (!rank) r->io_process = PETSC_TRUE;
  r->print_intermediate = print_intermediate;
  ierr = RichardsPrintVec(r->U,"initial_solution",-1); CHKERRQ(ierr);
  ierr = RichardsRunToTime(r,r->final_time); CHKERRQ(ierr);
  ierr = RichardsPrintVec(r->U,"final_solution",-1); CHKERRQ(ierr);
  ierr = TDyOutputRegression(r->tdy,r->U); CHKERRQ(ierr);
  ierr = RichardsDestroy(&r); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  if (!rank) printf("done\n");
  return(successful_exit_code);
}
