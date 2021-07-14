// This demo creates a box mesh and attaches labels to the faces on its six
// boundaries that allow a user to experiment with boundary conditions using
// a selection of simple flow problems.

#include "tdycore.h"
#include "private/tdycoreimpl.h"

#include <petscviewerhdf5.h>

int ParseOptions(PetscInt* dim,
                 PetscInt faces[3],
                 PetscReal lower[3],
                 PetscReal upper[3],
                 char mesh_file[]) {
  // Configure it with command line options.
  int ierr;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options",""); CHKERRQ(ierr);

  ierr = PetscOptionsInt("-dim","Problem dimension","",*dim,dim,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx","Number of x cells","",faces[0],&faces[0],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny","Number of y cells","",faces[1],&faces[1],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nz","Number of z cells (dim=3 only)","",faces[2],&faces[2],NULL); CHKERRQ(ierr);
  if ((faces[2] > 1) && (*dim == 2)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-Nz is only for 3D problems");
  }

  lower[0] = lower[1] = lower[2] = 0.0;
  ierr = PetscOptionsReal("-Lx","Length of domain in x","",upper[0],&upper[0],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Ly","Length of domain in y","",upper[1],&upper[1],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Lz","Length of domain in z","",upper[2],&upper[2],NULL); CHKERRQ(ierr);
  if ((upper[2] != 1.0) && (*dim == 2)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-Lz is only for 3D problems");
  }

  PetscBool mesh = PETSC_FALSE;
  mesh_file[0] = '\0';
  ierr = PetscOptionsString("-mesh","Read a mesh file in HDF5 format","",
    mesh_file, mesh_file, 256, &mesh); CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  return ierr;
}

int main(int argc, char **argv) {

  // Initialize our environment.
  PetscErrorCode ierr;
  ierr = TDyInit(argc, argv); CHKERRQ(ierr);

  MPI_Comm comm = PETSC_COMM_WORLD;

  // Parse command line options.
  PetscInt dim = 2;
  PetscInt Nx = 4;
  PetscInt faces[3] = {Nx, Nx, Nx};
  PetscReal lower[3] = {0.0,0.0,0.0};
  PetscReal Lx = 1.0;
  PetscReal upper[3] = {Lx, Lx, Lx};
  char mesh_file[256];
  ierr = ParseOptions(&dim, faces, lower, upper, mesh_file); CHKERRQ(ierr);

  // Initialize the model in MPFOA mode.
  TDy  tdy;
  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);

  // Create or read in a mesh.
  DM dm;
  if (strlen(mesh_file) > 0) { // read an HDF5 mesh file
    PetscViewer v;
    ierr = PetscViewerHDF5Open(comm, mesh_file, FILE_MODE_READ, &v); CHKERRQ(ierr);
    ierr = DMCreate(comm, &dm); CHKERRQ(ierr);
    ierr = DMSetType(dm, DMPLEX); CHKERRQ(ierr);
    ierr = DMLoad(dm, v); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&v); CHKERRQ(ierr);
  } else { // create a new box mesh with boundary faces tagged
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces,
             lower, upper, NULL, PETSC_TRUE, &dm); CHKERRQ(ierr);
  }

  // Distribute the mesh, including 1 cell of overlap.
  {
    DM dm_dist;
    ierr = DMPlexDistribute(dm, 1, NULL, &dm_dist);
    if (dm_dist) {
      DMDestroy(&dm);
      dm = dm_dist;
    }
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  }

  // Set up the dycore.
  ierr = TDySetDM(tdy,dm); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  // Set problem parameters.
  //ierr = TDySetPermeabilityTensor(tdy,permFunc); CHKERRQ(ierr);
  //ierr = TDySetForcingFunction(tdy,forcingFunc,NULL); CHKERRQ(ierr);
  //ierr = TDySetDirichletValueFunction(tdy,pressureFunc,NULL); CHKERRQ(ierr);
  //ierr = TDySetDirichletFluxFunction(tdy,velocityFunc,NULL); CHKERRQ(ierr);

  ierr = TDySetupNumericalMethods(tdy); CHKERRQ(ierr);

  // Do more stuff here.
  // ...

  // Clean up our mess.
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = TDyFinalize(); CHKERRQ(ierr);

  return 0;
}
