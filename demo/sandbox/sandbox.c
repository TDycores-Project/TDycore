// This demo creates a box mesh and attaches labels to the faces on its six
// boundaries that allow a user to experiment with boundary conditions using
// a selection of simple flow problems.

#include "tdycore.h"
#include "private/tdycoreimpl.h"

#include <petscviewerhdf5.h>

void ParseOptions(PetscInt* dim,
                  PetscInt faces[3],
                  PetscReal lower[3],
                  PetscReal upper[3],
                  char mesh_file[]) {
  // Configure it with command line options.
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options",""); CHKERRQ(ierr);

  ierr = PetscOptionsInt("-dim","Problem dimension","",dim,&dim,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx","Number of x cells","",faces[0],&faces[0],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny","Number of y cells","",faces[1],&faces[1],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nz","Number of z cells (dim=3 only)","",faces[2],&faces[2],NULL); CHKERRQ(ierr);
  if ((faces[2] > 1) && (dim == 2)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-Nz is only for 3D problems");
  }

  lower[0] = lower[1] = lower[2] = 0.0;
  ierr = PetscOptionsReal("-Lx","Length of domain in x","",upper[0],&upper[0],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Ly","Length of domain in y","",upper[1],&upper[1],NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Lz","Length of domain in z","",upper[2],&upper[2],NULL); CHKERRQ(ierr);
  if ((upper[2] != 1.0) && (dim == 2)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-Lz is only for 3D problems");
  }

  PetscBool mesh = PETSC_FALSE;
  mesh_file[0] = '\0';
  ierr = PetscOptionsString("-mesh","Read a mesh file in HDF5 format","",
    mesh_file, mesh_file, 256, &mesh); CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
}

int main(int argc, char **argv) {

  // Initialize our environment.
  PetscErrorCode ierr;
  ierr = TDyInit(argc, argv); CHKERRQ(ierr);

  // Parse command line options.
  PetscInt dim = 2;
  PetscInt faces[3] = {Nx, Ny, Nz};
  PetscReal lower[3] = {0.0,0.0,0.0};
  PetscReal upper[3] = {Lx, Ly, Lz};
  char mesh_file[256];
  ParseOptions(&dim, faces, lower, upper, mesh_file);

  // Initialize the model in MPFOA mode.
  TDy  tdy;
  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);

  // Create or read in a mesh.
  DM dm;
  if (mesh) { // read an HDF5 mesh file
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

  /* Setup problem parameters */
  if(wheeler2006){
    if(dim != 2){
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-paper wheeler2006 is only for -dim 2 problems");
    }
    switch(problem) {
        case 0: // not a problem in the paper, but want to check constants on the geometry
        ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
        break;
        case 1:
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2006_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2006_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2006_1,NULL); CHKERRQ(ierr);
        break;
        case 2:
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2006_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_2,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2006_2,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2006_2,NULL); CHKERRQ(ierr);
        break;
        default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-paper wheeler2006 only valid for -problem {0,1,2}");
    }
  }else if(wheeler2012){
    switch(problem) {
        case 1:
        if(dim != 2){
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-paper wheeler2012 -problem 1 is only for -dim 2");
        }
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_1); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2012_1,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2012_1,NULL); CHKERRQ(ierr);
        break;
        case 2:
        if(dim != 3){
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-paper wheeler2012 -problem 2 is only for -dim 3");
        }
        ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2012_2,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2012_2,NULL); CHKERRQ(ierr);
        break;
        default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-paper wheeler2012 only valid for -problem {1,2}");
    }
  }else{
    switch(problem) {
        case 1:
        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
        } else {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
        }
        ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
        ierr = TDySetDirichletFluxFunction(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
        break;

        case 2:

        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingQuadratic2D,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletValueFunction(tdy,PressureQuadratic2D,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletFluxFunction(tdy,VelocityQuadratic2D,NULL); CHKERRQ(ierr);
        } else {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_2,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2012_2,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2012_2,NULL); CHKERRQ(ierr);
        }
        break;

        case 3:
        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermTest2D); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2006_1,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2006_1,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2006_1,NULL); CHKERRQ(ierr);
        } else {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_2); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,Forcing3,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletValueFunction(tdy,Pressure3,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletFluxFunction(tdy,Velocity3,NULL); CHKERRQ(ierr);
        }
        break;

        case 4:
        if (dim == 2) {
          ierr = TDySetPermeabilityTensor(tdy,PermWheeler2012_1); CHKERRQ(ierr);
          ierr = TDySetForcingFunction(tdy,ForcingWheeler2012_1,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletValueFunction(tdy,PressureWheeler2012_1,NULL); CHKERRQ(ierr);
          ierr = TDySetDirichletFluxFunction(tdy,VelocityWheeler2012_1,NULL); CHKERRQ(ierr);
        } else {

        }
        break;
    }
  }

  ierr = TDySetupNumericalMethods(tdy); CHKERRQ(ierr);

  /* Compute system */
  Mat K;
  Vec U,Ue,F;
  ierr = DMCreateGlobalVector(dm,&U ); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&Ue); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F ); CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K ); CHKERRQ(ierr);
  ierr = TDyComputeSystem(tdy,K,F); CHKERRQ(ierr);

  /* Solve system */
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  /* Output solution */
  PetscViewer viewer;
  PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),"sol.vtk",FILE_MODE_WRITE,&viewer);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer); CHKERRQ(ierr); // the approximate solution
  //ierr = OperatorApplicationResidual(tdy,Ue,K,tdy->ops->computedirichletvalue,F);
  ierr = VecView(F,viewer); CHKERRQ(ierr); // the residual K*Ue-F
  ierr = VecView(Ue,viewer); CHKERRQ(ierr);  // the exact solution
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  /* Evaluate error norms */
  PetscReal normp,normv;
  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%e %e\n",normp,normv); CHKERRQ(ierr);

  /* Save vertex coordinates */
  PetscBool file_not_specified;
  ierr = PetscStrcasecmp(vertices_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveVertices(dm,vertices_filename); CHKERRQ(ierr);
  }

  /* Save cell centroids */
  ierr = PetscStrcasecmp(centroids_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveCentroids(dm,centroids_filename); CHKERRQ(ierr);
  }

  /* Save true solution */
  ierr = PetscStrcasecmp(true_pres_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveTrueSolution(tdy,true_pres_filename); CHKERRQ(ierr);
  }

  /* Save forcing */
  ierr = PetscStrcasecmp(forcing_filename,"none",&file_not_specified); CHKERRQ(ierr);
  if (file_not_specified == 0) {
    ierr = SaveForcing(tdy,forcing_filename); CHKERRQ(ierr);
  }

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U);

  /* Cleanup */
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&Ue); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = TDyFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
