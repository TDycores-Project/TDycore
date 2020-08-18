#include <tdycore.h>
#include <private/tdycoreimpl.h>
#include <tdydriver.h>
#include <tdyts.h>
#include <tdyio.h>

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code;
  PetscBool print_intermediate = PETSC_FALSE;
  PetscMPIInt rank;
  TDy tdy = PETSC_NULL;

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

  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  ierr = TDyDriverInitializeTDy(tdy); CHKERRQ(ierr);
  if (!rank) tdy->io->io_process = PETSC_TRUE;
  tdy->io->print_intermediate = print_intermediate;
  ierr = PrintVec(tdy->U,"initial_solution",-1); CHKERRQ(ierr);
  ierr = TimestepperRunToTime(tdy,tdy->ts->final_time); CHKERRQ(ierr);
  ierr = PrintVec(tdy->U,"final_solution",-1); CHKERRQ(ierr);
  ierr = TDyOutputRegression(tdy,tdy->U); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  if (!rank) printf("done\n");
  return(successful_exit_code);
}
