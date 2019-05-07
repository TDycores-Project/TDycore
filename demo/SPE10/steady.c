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
  PetscInt i,x,y,z,d,nx=60,ny=220,nz=85,dim,dim2;
  const char filename[] = "spe_perm.dat";
  FILE *f = fopen(filename,"r");
  PetscReal *buffer;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;

  // Rotation matrix
  PetscReal R[9],T[9];
  Rotation(R,ang,2,dim);
  
  // Read data
  ierr = PetscMalloc(3*nz*ny*nx*sizeof(PetscReal),&buffer); CHKERRQ(ierr);
  if(f){
    i=0;
    while(fscanf(f,"%lf",&buffer[i]) != EOF) { i += 1; }
  }

  PetscInt c,cStart,cEnd;
  PetscReal dx = 20, dy = 10, dz = 2;
  PetscReal xL = -600, yL = -1100, zL = -170;
  PetscBLASInt n = dim;
  PetscReal one = 1, zero = 0;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    if(dim==2){ /* in 2D we use the x-y plane */
      x = (tdy->X[c*dim  ]-xL)/dx;
      y = 0;
      z = (tdy->X[c*dim+1]-zL)/dz;
      z = nz - 1 - z;
      tdy->K[dim2*(c-cStart)+0*dim+0] = buffer[0*(nz*ny*nx)+z*(ny*nx)+y*(nx)+x];
      tdy->K[dim2*(c-cStart)+1*dim+1] = buffer[2*(nz*ny*nx)+z*(ny*nx)+y*(nx)+x];
    }else{
      x = (tdy->X[c*dim  ]-xL)/dx;
      y = (tdy->X[c*dim+1]-yL)/dy;
      z = (tdy->X[c*dim+2]-zL)/dz;
      z = nz - 1 - z;
      for(d=0;d<dim;d++){
	tdy->K[dim2*(c-cStart)+d*dim+d] = buffer[d*(nz*ny*nx)+z*(ny*nx)+y*(nx)+x];
      }
    }
    /* K = R * K * R.T */
    BLASgemm_("N","T",&n,&n,&n,&one,&(tdy->K[dim2*(c-cStart)]),&n,R,&n,&zero,T,&n);
    BLASgemm_("N","N",&n,&n,&n,&one,R,&n,T,&n,&zero,&(tdy->K[dim2*(c-cStart)]),&n);
  }
  ierr = PetscFree(buffer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void Pressure(double *x,double *f){
  PetscReal xL = 1200; //, yL = 2200, zL = 170;
  (*f) = (0.5*xL+x[0])/xL;
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscInt dim = 2;
  PetscReal ang = 0;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"SPE Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",
			  dim,&dim,NULL); CHKERRQ(ierr);
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
  ierr = TDySetDirichletFunction(tdy,Pressure); CHKERRQ(ierr);

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
