#include <tdycore.h>
#include <private/tdycoreimpl.h>
#include <tdyio.h>

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code = 0;
  PetscMPIInt rank, size;
  MPI_Comm comm = PETSC_COMM_WORLD;
  TDy tdy = PETSC_NULL;
  TDyIOFormat format = HDF5Format;

  ierr = TDyInit(argc, argv); CHKERRQ(ierr);
  ierr = TDyCreate(comm, &tdy); CHKERRQ(ierr);
  ierr = TDySetMode(tdy,RICHARDS); CHKERRQ(ierr);
  ierr = TDySetDiscretization(tdy,MPFA_O); CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  PetscPrintf(comm,"Beginning Richards Driver simulation.\n");
  ierr = PetscOptionsBegin(comm,NULL,"Sample Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code",
                         "Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // default mode and method must be set prior to TDySetFromOptions()
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  ierr = TDyDriverInitializeTDy(tdy); CHKERRQ(ierr);
  if (!rank) {
    ierr = TDyIOSetIOProcess(tdy->io, PETSC_TRUE); CHKERRQ(ierr);
  }
  PetscPrintf(comm,"--\n");
  if (size == 1) {
    ierr = TDyIOSetMode(tdy,format);CHKERRQ(ierr);
    ierr = TDyIOWriteVec(tdy); CHKERRQ(ierr);
  }
  ierr = TDyTimeIntegratorRunToTime(tdy,tdy->ti->final_time);
         CHKERRQ(ierr);
  if (size == 1) {ierr = TDyIOWriteVec(tdy); CHKERRQ(ierr);}
  ierr = TDyOutputRegression(tdy,tdy->solution); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);

  PetscPrintf(comm,"--\n");
  PetscPrintf(comm,"Simulation complete.\n");
  ierr = TDyFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}
