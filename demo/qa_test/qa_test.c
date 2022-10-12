#include "tdycore.h"
#include "private/tdycoreimpl.h"

PetscReal alpha = 1;
PetscReal density = 1000.0;
PetscReal visocity = 1.;

/*--- -dim {2|3} -problem 1 ---------------------------------------------------------------*/

void Permeability(PetscReal *x,PetscReal *K) {
  K[0] = 10e-12; K[1] = 0;
  K[2] = 0     ; K[3] = K[0];
}

PetscErrorCode LinearProblem_Pressure(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) {
  PetscReal p_left = 3.e6, p_right = 1.e6;
  PetscReal x_min = 0., x_max = 1.;

  (*p) = p_left - (p_left-p_right)/(x_max - x_min)*x[0];

  PetscFunctionReturn(0);
}

PetscErrorCode LinearProblem_Velocity(TDy tdy,PetscReal *x,PetscReal *v,void *ctx) {
  v[0] = 0; v[1] = 0; v[2] = 0;
  PetscFunctionReturn(0);
}
PetscErrorCode LinearProblem_Forcing(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  (*f) = 0; PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2, problem = 2;
  PetscInt successful_exit_code=0;
  PetscBool perturb = PETSC_FALSE;
  char exofile[256];
  PetscBool exo = PETSC_FALSE;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
  ierr = PetscOptionsInt ("-dim","Problem dimension","",dim,&dim,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N","Number of elements in 1D","",N,&N,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-problem","Problem number","",problem,&problem,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  PetscOptionsEnd();

  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  DMLabel marker;

  const PetscInt  faces[3] = {N,N,N  };
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
	       NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);

  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);
  if(dim == 2) {
    switch(problem) {
    case 1:
      ierr = TDySetPermeabilityTensor(tdy,Permeability); CHKERRQ(ierr);
      ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetBoundaryPressureFn(tdy,PressureConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetBoundaryVelocityFn(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
      break;
    }
  }
  ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Compute system */
  Mat K;
  Vec U,Ue,F;
  ierr = TDyCreatePrognosticVector(tdy,&U ); CHKERRQ(ierr);
  ierr = TDyCreatePrognosticVector(tdy,&Ue); CHKERRQ(ierr);
  ierr = TDyCreatePrognosticVector(tdy,&F ); CHKERRQ(ierr);
  ierr = TDyCreateMatrix          (tdy,&K ); CHKERRQ(ierr);
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
  ierr = OperatorApplicationResidual(tdy,Ue,K,tdy->ops->compute_boundary_pressure,F);
  ierr = VecView(F,viewer); CHKERRQ(ierr); // the residual K*Ue-F
  ierr = VecView(Ue,viewer); CHKERRQ(ierr);  // the exact solution
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy);

  /* Cleanup */
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&Ue); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
