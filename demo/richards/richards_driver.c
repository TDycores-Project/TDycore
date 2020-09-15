#include <tdycore.h>
#include <private/tdycoreimpl.h>
#include <tdydriver.h>
#include <tdyti.h>
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
  PetscPrintf(PETSC_COMM_WORLD,"Beginning Richards Driver simulation.\n");
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
  PetscPrintf(PETSC_COMM_WORLD,"--\n");
  ierr = TDyIOPrintVec(tdy->solution,"initial_solution",-1); CHKERRQ(ierr);
  ierr = TDyTimeIntegratorRunToTime(tdy,tdy->ti->final_time); 
         CHKERRQ(ierr);
  ierr = TDyIOPrintVec(tdy->solution,"final_solution",-1); CHKERRQ(ierr);
  ierr = TDyOutputRegression(tdy,tdy->solution); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"--\n");
  PetscPrintf(PETSC_COMM_WORLD,"Simulation complete.\n");
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}
