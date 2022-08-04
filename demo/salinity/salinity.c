#include <tdycore.h>
#include <private/tdycoreimpl.h>
#include <tdyio.h>

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code;
  PetscMPIInt rank, size;
  MPI_Comm comm = PETSC_COMM_WORLD;
  TDy tdy = PETSC_NULL;
  TDyIOFormat format = PetscViewerASCIIFormat;
  Vec U;
  PetscReal *soln;
  PetscInt ncells,c;
  PetscInt local_size;

  ierr = TDyInit(argc, argv); CHKERRQ(ierr);
  ierr = TDyCreate(comm, &tdy); CHKERRQ(ierr);
  ierr = TDySetMode(tdy,SALINITY); CHKERRQ(ierr);
  ierr = TDySetDiscretization(tdy,MPFA_O); CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Beginning salinity simulation.\n");
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code",
                         "Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = TDySetWaterDensityType(tdy,WATER_DENSITY_BATZLE_AND_WANG);
  ierr = TDySetWaterViscosityType(tdy,WATER_VISCOSITY_BATZLE_AND_WANG);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  ierr = TDyDriverInitializeTDy(tdy); CHKERRQ(ierr);
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);

  // default mode and method must be set prior to TDySetFromOptions()
  ncells = 5;
  //ierr = VecGetLocalSize(tdy->solution,&ncells); CHKERRQ(ierr);
  PetscRandom rand;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = VecGetArray(U,&soln);
  for (c=0; c<(ncells*2); c+=2){
    soln[c] = 201325.;
  }
  soln[1] = 1.0;
  for (c=3; c<=(ncells*2); c+=2){
  soln[c] = 0.000001;
  // printf("soln  %f",soln[c]);
  }
  ierr = VecRestoreArray(U,&soln);

  ierr = TDySetInitialCondition(tdy,U);

  if (!rank) {
    ierr = TDyIOSetIOProcess(tdy->io, PETSC_TRUE); CHKERRQ(ierr);
  }
  PetscPrintf(PETSC_COMM_WORLD,"--\n");
  if (size == 1) {
    ierr = TDyIOSetMode(tdy,format);CHKERRQ(ierr);
    ierr = TDyIOWriteVec(tdy); CHKERRQ(ierr);
  }
  ierr = TDyTimeIntegratorRunToTime(tdy,tdy->ti->final_time); 
         CHKERRQ(ierr);
  if (size == 1) {ierr = TDyIOWriteVec(tdy); CHKERRQ(ierr);}
  ierr = TDyOutputRegression(tdy,tdy->soln); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"--\n");
  PetscPrintf(PETSC_COMM_WORLD,"Simulation complete.\n");
  ierr = TDyFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}
