#include "tdycore.h"
#include <stdio.h>

void PermeabilityBlockX(PetscInt dim,PetscReal *x,PetscReal *K,
                        PetscReal high_perm,PetscReal low_perm) {
  const PetscReal xbegin = 0.0;
  const PetscReal xend = 0.5;
  const PetscReal density = 1000.;
  const PetscReal viscosity = 1.e-3;
  PetscReal conversion = -999.;
  // to get PFLOTRAN equations
  conversion  = 1./viscosity;
  // to get exact PFLOTRAN coefficients, uncomment
  //perm_to_K = density/18.01534;
  PetscReal high_coef = high_perm*conversion;
  PetscReal low_coef = low_perm*conversion;
  PetscInt flag = 0;
  for(int i=0; i<dim; i++)
    if (x[i] >= xbegin && x[i] <= xend) flag++;
  for(int i=0; i<dim*dim; i++)
    K[i] = 0.;
  if (flag%2 != 0) {
    for (int i=0; i<dim*dim; i+=(dim+1))
      K[i] = high_coef;
  } else {
    for (int i=0; i<dim*dim; i+=(dim+1))
      K[i] = low_coef;
  }
}

void PermeabilityBlock1(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 1;
  const PetscReal high_perm = 1.e-12;
  const PetscReal low_perm = 1.e-13;
  PermeabilityBlockX(dim,x,K,high_perm,low_perm);
}

void PermeabilityBlock2(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 2;
  const PetscReal high_perm = 1.e-12;
  const PetscReal low_perm = 1.e-13;
  PermeabilityBlockX(dim,x,K,high_perm,low_perm);
}

void PermeabilityBlock3(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 3;
  const PetscReal high_perm = 1.e-12;
  const PetscReal low_perm = 1.e-13;
  PermeabilityBlockX(dim,x,K,high_perm,low_perm);
}

void PermeabilityUni2(PetscReal *x,PetscReal *K) {
  const PetscInt dim = 2;
  const PetscReal perm = 1.e-12;
  PermeabilityBlockX(dim,x,K,perm,perm);
}

/*--- -dim {2} -problem 1 ---------------------------------------------------------------*/

PetscErrorCode PressureBlock(TDy tdy,double *x,double *p,void *ctx) {
  const PetscInt begin = 0.0;
  const PetscInt end = 1.;
  PetscReal xx=x[0], yy=x[1], zz=x[2];

  (*p) = -999.;
  // west
  if (xx < 1.e-40) {
    if (yy <= end/2.) (*p) = 3.e6; 
  }
  // south
  else if (yy < 1.e-40) {
    if (xx <= end/2.) (*p) = 3.e6; 
  }
  // east
  else if (fabs(end-xx) < 1.e-40) {
    if (yy >= end/2.) (*p) = 1.e6; 
  }
  // north
  else if (fabs(end-yy) < 1.e-40) {
    if (xx >= end/2.) (*p) = 1.e6; 
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PressureGradual(TDy tdy,double *x,double *p,void *ctx) {
  const PetscInt begin = 0.0;
  const PetscInt end = 1.;
  const PetscReal high = 3.e6;
  const PetscReal low = 1.e6;
  
  PetscReal xx=x[0], yy=x[1], zz=x[2];
  PetscReal range = high-low;
  PetscReal mid = 0.5*(high+low);

  // west
  if (xx < 1.e-40) {
    (*p) = high-yy*0.5*range;
  }
  // south
  else if (yy < 1.e-40) {
    (*p) = high-xx*0.5*range;
  }
  // east
  else if (fabs(end-xx) < 1.e-40) {
    (*p) = low+(1.-yy)*0.5*range;
  }
  // north
  else if (fabs(end-yy) < 1.e-40) {
    (*p) = low+(1.-xx)*0.5*range;
  }
  else {
    printf("Unknown location in PressureGradual: %f %f %f\n",xx,yy,zz);
    exit(1);
  }
//  printf("%f %f %f : %f\n",xx,yy,zz,*p);
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityConstant(TDy tdy,double *x,double *v,void *ctx) { v[0] = 0; v[1] = 0; v[2] = 0; PetscFunctionReturn(0);}
PetscErrorCode ForcingConstant(TDy tdy,double *x,double *f,void *ctx) { (*f) = 0; PetscFunctionReturn(0);}

int main(int argc, char **argv) {

  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2, problem = -999;
  PetscInt successful_exit_code=0;
  PetscBool verbose;
  PetscBool qa;
  PetscInt i;
  FILE *fp;
  char input_filename[128];
  char output_filename[128];
  char algorithm[32];
  char filename[32];
  char prefix[32];

  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);

  strcpy(input_filename,"tdycore.in");
  strcpy(algorithm,"TPF");

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",dim,&dim,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool ("-verbose","Verbose output","",verbose,&verbose,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool ("-qa","QA output","",qa,&qa,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N","Number of elements in 1D","",N,&N,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-problem","Problem number","",problem,&problem,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsString ("-algorithm","Algorithm","",algorithm,algorithm,32,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsString("-input","Input filename","",input_filename,input_filename,128,NULL);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  strcpy(prefix,"");

  if (problem > 0) {
    // nullify input_filename so that command line options override the input file
    strcpy(input_filename,"");
  }

  if (strlen(input_filename) > 1) {
    printf("Reading input.\n");
    fp = fopen(input_filename,"r");
    fscanf(fp,"%d",&problem);
    fscanf(fp,"%d",&dim);
    fscanf(fp,"%d",&N);
    fscanf(fp,"%s",algorithm);
    fclose(fp);
    for (i=strlen(input_filename); i > -1; i--) 
      if (input_filename[i] == '.') break;
    printf("prefix1 = '%s'\n",prefix);
    printf("i = '%d'\n",i);
    printf("input_filename = '%s'\n",input_filename);
    if (i > -1) {//{strncpy(prefix,input_filename,i);
      //      prefix[i+1] = '\0';
    int ii;
    for (ii=0; ii<i; ii++) {
      prefix[ii] = input_filename[ii];}
     prefix[ii+1] = '\0';
      }
  }
  
  printf("prefix2 = '%s'\n",prefix);
  printf("\n");
  printf("Problem        : %d\n",problem);
  printf("Dimension      : %d\n",dim);
  PetscInt temp_int = N;
  for (int i=0; i<dim-1; i++)
    temp_int *= N;
  printf("Number of Cells: %d\n",temp_int);
  printf("      Algorithm: %s\n",algorithm);
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
        ierr = TDySetDirichletValueFunction(tdy,PressureBlock,NULL); CHKERRQ(ierr);
        break;
    }
  } else if (dim == 2) {
    switch(problem) {
      case 1:
        // block boundary pressures
        ierr = TDySetPermeabilityTensor(tdy,PermeabilityBlock2); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureBlock,NULL); CHKERRQ(ierr);
        break;
      case 2:
        // gradual boundary pressures
        ierr = TDySetPermeabilityTensor(tdy,PermeabilityBlock2); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureGradual,NULL); CHKERRQ(ierr);
        break;
      case 3:
        // gradual boundary pressure + uniform perm
        ierr = TDySetPermeabilityTensor(tdy,PermeabilityUni2); CHKERRQ(ierr);
        ierr = TDySetDirichletValueFunction(tdy,PressureGradual,NULL); CHKERRQ(ierr);
        break;
    }
  }

  if (!strcmp(algorithm,"TPF")) {
    ierr = TDySetDiscretizationMethod(tdy,TPF); CHKERRQ(ierr);
  } else if (!strcmp(algorithm,"WY")) {
    ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);
  } else if (!strcmp(algorithm,"MPF")) {
    ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);
  } else {
    printf("Unrecognized algorithm for TDySetDiscretizationMethod: %s\n",algorithm);
    exit(1);
  }

  if (strlen(prefix) == 0) {
    sprintf(prefix,"tdycore_p%d_%s",problem,algorithm);
  }


  printf("Output prefix: %s\n",prefix);
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
  if (verbose) {
    strcpy(filename,prefix);
    strcat(filename,".mat");
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewer); CHKERRQ(ierr);
    ierr = MatView(K,viewer);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    strcpy(filename,prefix);

    strcat(filename,".rhs");
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewer); CHKERRQ(ierr);
    ierr = VecView(F,viewer);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }

  /* Solve system */
  printf("Solving system.\n");
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  if (verbose || qa) {
    strcpy(filename,prefix);

    strcat(filename,".sol");
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewer); CHKERRQ(ierr);
    ierr = VecView(U,viewer);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }
  
  /* Output solution */
  strcpy(filename,prefix);

  strcat(filename,".vtk");
  PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),filename,FILE_MODE_WRITE,&viewer);
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
