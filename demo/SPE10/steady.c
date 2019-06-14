#include "tdycore.h"
#include <petscblaslapack.h>

PetscErrorCode Rotation(PetscReal *R, PetscReal ang, PetscInt axis, PetscInt dim){
  PetscFunctionBegin;
  PetscInt i,j,row,col;
  ang = ang/180*PETSC_PI;
  if(dim==2) axis = 2;
  for(i=0;i<dim;i++){
    for(j=0;j<dim;j++){
      if((i == axis) || (j == axis)){
	if(i==j){
	  R[j*dim+i] = 1;
	}else{
	  R[j*dim+i] = 0;
	}
      }else{
	row = i; if(row > axis) row -= 1;
	col = j; if(col > axis) col -= 1;
	if(row == 0 && col == 0) R[j*dim+i] =  PetscCosReal(ang); 
	if(row == 0 && col == 1) R[j*dim+i] = -PetscSinReal(ang); 
	if(row == 1 && col == 0) R[j*dim+i] =  PetscSinReal(ang); 
	if(row == 1 && col == 1) R[j*dim+i] =  PetscCosReal(ang); 
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ReadSPE10Permeability(TDy tdy,PetscReal ang){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt i,x,y,z,d,nx=60,ny=220,nz=85,dim;
  const char filename[] = "spe_perm.dat";
  FILE *f = fopen(filename,"r");
  PetscReal *buffer,*X;
  DM dm;
  ierr = TDyGetDimension(tdy,&dim); CHKERRQ(ierr);
  ierr = TDyGetDM(tdy,&dm); CHKERRQ(ierr);
  ierr = TDyGetCentroidArray(tdy,&X); CHKERRQ(ierr);
  
  // Rotation matrix
  PetscBLASInt n = dim;
  PetscReal one = 1, zero = 0;
  PetscReal R[9],R1[9],T[9],K[9];
  if(dim == 2){
    Rotation(R,ang,2,dim);
  }else{
    Rotation(R1,    ang,1,dim);
    Rotation(T ,0.5*ang,2,dim);
    BLASgemm_("N","N",&n,&n,&n,&one,R1,&n,T,&n,&zero,R,&n);
  }
      
  // Read data
  ierr = PetscMalloc(3*nz*ny*nx*sizeof(PetscReal),&buffer); CHKERRQ(ierr);
  if(f){
    i=0;
    while(fscanf(f,"%lf",&buffer[i]) != EOF) { i += 1; }
  }

  PetscInt c,cStart,cEnd;
  PetscReal dx = 20, dy = 10, dz = 2;
  PetscReal xL = -600, yL = -1100, zL = -170;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  for(c=cStart;c<cEnd;c++){
    K[0] = K[1] = K[2] = K[3] = K[4] = K[5] = K[6] = K[7] = K[8] = 0;
    if(dim==2){ /* in 2D we use the x-z plane */
      x = (X[c*dim  ]-xL)/dx;
      y = 0;
      z = (X[c*dim+1]-zL)/dz;
      z = nz - 1 - z;
      K[0*dim+0] = buffer[0*(nz*ny*nx)+z*(ny*nx)+y*(nx)+x];
      K[1*dim+1] = buffer[2*(nz*ny*nx)+z*(ny*nx)+y*(nx)+x];
    }else{
      x = (X[c*dim  ]-xL)/dx;
      y = (X[c*dim+1]-yL)/dy;
      z = (X[c*dim+2]-zL)/dz;
      z = nz - 1 - z;
      for(d=0;d<dim;d++){
	K[d*dim+d] = buffer[d*(nz*ny*nx)+z*(ny*nx)+y*(nx)+x];
      }
    }
    /* K = R * K * R.T */
    BLASgemm_("N","T",&n,&n,&n,&one,K,&n,R,&n,&zero,T,&n);
    BLASgemm_("N","N",&n,&n,&n,&one,R,&n,T,&n,&zero,K,&n);
    ierr = TDySetCellPermeability(tdy,c,K); CHKERRQ(ierr);
  }
  ierr = PetscFree(buffer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Pressure(TDy tdy,double *x,double *f,void *ctx){
  PetscReal xL = 1200, yL = 2200, zL = -170;
  (*f)  = (0.5*xL+x[0])/xL;
  (*f) += (0.5*yL+x[1])/yL;
  (*f) += (0.5*zL+x[2])/zL;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscInt dim = 2, N = 0;
  PetscReal ang = 0;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"SPE Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",
			  dim,&dim,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N","Number of cells in each dimension","",
			  N,&N,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-angle","Permeability angle","",
			  ang,&ang,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    
  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  PetscInt  faces[3] = {  60,  220,  85};
  PetscReal lower[3] = {-600,-1100,-170};
  PetscReal upper[3] = {+600,+1100,   0};
  if(dim==2){ /* if 2D we do the x-z plane */
    faces[1] = faces[2];
    lower[1] = lower[2];
    upper[1] = upper[2];
  }
  if(N>0){
    PetscReal fct = (PetscReal)N/(PetscReal)faces[0];
    faces[0] = N;
    faces[1] = (PetscInt)(fct*faces[1]);
    faces[2] = (PetscInt)(fct*faces[2]);
  }
  printf("grid: %d %d %d = %d\n",faces[0],faces[1],faces[2],faces[0]*faces[1]*faces[2]);
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
                             NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);
  ierr = ReadSPE10Permeability(tdy,ang); CHKERRQ(ierr);
  ierr = TDySetDirichletValueFunction(tdy,Pressure,NULL); CHKERRQ(ierr);

  /* Compute system */
  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F); CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K); CHKERRQ(ierr);
  ierr = TDyComputeSystem(tdy,K,F); CHKERRQ(ierr);

  /* Solve system */
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  /* Cleanup */
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(0);
}
