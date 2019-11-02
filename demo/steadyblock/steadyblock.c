#include "tdycore.h"
#include <stdio.h>

void PermeabilityBlockX(PetscInt dim,PetscReal *x,PetscReal *K) {
  const PetscReal xbegin = 0.0;
  const PetscReal xend = 0.5;
  const PetscReal high_perm = 1.e-12;
  const PetscReal low_perm = 1.e-13;
//  const PetscReal perm_to_K = 9.8068*1000./1.e-3;
  const PetscReal perm_to_K = 1.;
  PetscReal high_K = high_perm*perm_to_K;
  PetscReal low_K = low_perm*perm_to_K;
  PetscInt flag = 0;
  for(int i=0; i<dim; i++)
    if (x[i] >= xbegin && x[i] <= xend) flag++;
  for(int i=0; i<dim*dim; i++)
    K[i] = 0.;
  if (flag%2 != 0) {
    for (int i=0; i<dim*dim; i+=(dim+1))
      K[i] = high_K;
  } else {
    for (int i=0; i<dim*dim; i+=(dim+1))
      K[i] = low_K;
  }

  K[0] = high_perm; K[1] = 0.0;
  K[2] = 0.0     ; K[3] = high_perm;
  if (x[0]>0.0 && x[0]<0.5 && x[1]>0.0 && x[1]<0.5) {
    K[0] = low_perm;
    K[3] = low_perm;
  }
  if (x[0]>0.5 && x[0]<1.0 && x[1]>0.5 && x[1]<1.0) {
    K[0] = low_perm;
    K[3] = low_perm;
  }
}

void PermeabilityBlock1(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 1;
  PermeabilityBlockX(dim,x,K);
}

void PermeabilityBlock2(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 2;
  PermeabilityBlockX(dim,x,K);
}

void PermeabilityBlock3(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 3;
  PermeabilityBlockX(dim,x,K);
}

/*--- -dim {2} -problem 1 ---------------------------------------------------------------*/

PetscErrorCode PressureConstant(TDy tdy,double *x,double *p,void *ctx) {
  const PetscInt begin = 0.0;
  const PetscInt end = 1.;
  PetscReal xx=x[0], yy=x[1], zz=x[2];

  (*p) = -999.;
  if (xx < 1.e-40) {
    //if (yy <= end)
    // south
    (*p) = 3.e6;
  }
  else if (yy < 1.e-40) {
    //if (xx <= end)
    // west
    (*p) = 3.e6;
  }
  else if (fabs(end-xx) < 1.e-40) {
    //if (yy >= end)
    // north
    (*p) = 1.e6;
  }
  else if (fabs(end-yy) < 1.e-40) {
    //if (xx >= end)
    // east
    (*p) = 1.e6;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityConstant(TDy tdy,double *x,double *v,void *ctx) { v[0] = 0; v[1] = 0; v[2] = 0; PetscFunctionReturn(0);}
PetscErrorCode ForcingConstant(TDy tdy,double *x,double *f,void *ctx) { (*f) = 0; PetscFunctionReturn(0);}

int main(int argc, char **argv) {

  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2, problem = 2;
  PetscInt successful_exit_code=0;
  FILE *fp;
  char string[128];

  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);

  strcpy(string,"tdycore.in");

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",dim,&dim,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",dim,&dim,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N","Number of elements in 1D","",N,&N,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-problem","Problem number","",problem,&problem,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsString("-input","Input filename","",string,string,128,NULL);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  printf("Reading input.\n");
  fp = fopen(string,"r");
  fscanf(fp,"%d",&problem);
  fscanf(fp,"%d",&dim);
  fscanf(fp,"%d",&N);
  fclose(fp);

  printf("\n");
  printf("Problem        : %d\n",problem);
  printf("Dimension      : %d\n",dim);
  PetscInt temp_int = N;
  for (int i=0; i<dim-1; i++)
    temp_int *= N;
  printf("Number of Cells: %d\n",temp_int);
  printf("\n");

  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
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
  printf("Creating TDycore.\n");
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);
  if (dim == 1) {
    ierr = TDySetPermeabilityTensor(tdy,PermeabilityBlock1); CHKERRQ(ierr);
    switch(problem) {
    case 1:
//      ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
//      ierr = TDySetDirichletFluxFunction(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
      break;
    }
  } else if (dim == 2) {
    ierr = TDySetPermeabilityTensor(tdy,PermeabilityBlock2); CHKERRQ(ierr);
    switch(problem) {
    case 1:
//      ierr = TDySetForcingFunction(tdy,ForcingConstant,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureConstant,NULL); CHKERRQ(ierr);
//      ierr = TDySetDirichletFluxFunction(tdy,VelocityConstant,NULL); CHKERRQ(ierr);
      break;
      break;
    }
  }

  ierr = TDySetDiscretizationMethod(tdy,TPF); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Compute system */
  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F); CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K); CHKERRQ(ierr);
  ierr = MatSetOption(K,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);
  printf("Creating system.\n");
  ierr = TDyComputeSystem(tdy,K,F); CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"tdycore.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(K,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"tdycore.rhs",&viewer); CHKERRQ(ierr);
  ierr = VecView(F,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  /* Solve system */
  printf("Solving system.\n");
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  printf("Outputing results.\n");
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"tdycore.sol",&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  
  /* Output solution */
  PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),"tdycore.vtk",FILE_MODE_WRITE,&viewer);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer); CHKERRQ(ierr); // the approximate solution
  ierr = VecView(F,viewer); CHKERRQ(ierr); // the residual K*Ue-F
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


  /* Evaluate error norms */
//  PetscReal normp,normv;
//  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv);
//  ierr = PetscPrintf(PETSC_COMM_WORLD,"%e %e\n",normp,normv); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U);

  /* Cleanup */
  printf("Cleaning up.\n");
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
 // ierr = VecDestroy(&Ue); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  printf("Done.\n");
  return(successful_exit_code);
}
