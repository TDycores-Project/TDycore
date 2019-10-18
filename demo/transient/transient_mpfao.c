#include "tdycore.h"

void Porosity(double *x,double *theta) {
  (*theta) = 0.115;
}

void Permeability3D(double *x,double *K) {
  K[0] = 1.0e-10; K[1] = 0.0    ; K[2] = 0.0    ;
  K[3] = 0.0    ; K[4] = 1.0e-10; K[5] = 0.0    ;
  K[6] = 0.0    ; K[7] = 0.0    ; K[8] = 1.0e-10;
}

PetscErrorCode PermeabilityFunction3D(TDy tdy, double *x, double *K, void *ctx){
  Permeability3D(x, K);
  return 0;
}

PetscErrorCode Pressure(TDy tdy,double *x,double *p,void *ctx) {
  (*p) = 91325;
  PetscFunctionReturn(0);
}

PetscErrorCode Forcing(TDy tdy,double *x,double *f,void *ctx) {
  (*f) = 0;
  PetscFunctionReturn(0);
}


PetscErrorCode PerturbInteriorVertices(DM dm,PetscReal h) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value,dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  /* this is the 'marker' label which marks boundary entities */
  ierr = DMGetLabelByNum(dm,2,&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value); CHKERRQ(ierr);
    if(dim==2) {
      if(value==-1) {
        /* perturb randomly O(h*sqrt(2)/3) */
        PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*(h*0.471404);
        PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
        coords[offset  ] += r*PetscCosReal(t);
        coords[offset+1] += r*PetscSinReal(t);
      }
    } else {
      /* this is because 'marker' is broken in 3D */
      if(coords[offset] > -0.5 && coords[offset] < 0.5 &&
	 coords[offset+1] > -0.5 && coords[offset+1] < 0.5 &&
	 coords[offset+2] > -0.5 && coords[offset+2] < 0.5) {
        coords[offset+2] += (((PetscReal)rand())/((PetscReal)RAND_MAX)-0.5)*h*0.8;
      }
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt nx = 3, ny = 3, nz = 3, dim = 3;
  PetscInt successful_exit_code=0;
  char exofile[256];
  PetscBool exo = PETSC_FALSE;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
			   "Transient Options",""); CHKERRQ(ierr);
  //ierr = PetscOptionsInt("-N","Number of elements in 1D",
	//		 "",N,&N,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsString("-exo","Mesh file in exodus format","",
			    exofile,exofile,256,&exo); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  DMLabel marker;
  if(exo){
    ierr = DMPlexCreateExodusFromFile(PETSC_COMM_WORLD,exofile,
				      PETSC_TRUE,&dm); CHKERRQ(ierr);
    //ierr = DMPlexOrient(dm); CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = DMCreateLabel(dm,"marker"); CHKERRQ(ierr);
    ierr = DMGetLabel(dm,"marker",&marker); CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm,1,marker); CHKERRQ(ierr);
  }else{
    const PetscInt  faces[3] = {nx,ny,nz};
    const PetscReal lower[3] = {0.0,0.0,0.0};
    const PetscReal upper[3] = {1.0,1.0,1.0};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
			       NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
    //ierr = PerturbInteriorVertices(dm,1./nx); CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  PetscInt c,cStart,cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscReal residualSat[cEnd-cStart];
  PetscInt index[cEnd-cStart];
  for (c=0;c<cEnd-cStart;c++) {
    index[c] = c;
    residualSat[c] = 0.115;
  }

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);
  ierr = TDySetPorosity(tdy,Porosity); CHKERRQ(ierr);
  //ierr = TDySetPermeabilityScalar(tdy,Permeability); CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(tdy,PermeabilityFunction3D,NULL); CHKERRQ(ierr);
  ierr = TDySetResidualSaturationValuesLocal(tdy,cEnd-cStart,index,residualSat);
  ierr = TDySetForcingFunction(tdy,Forcing,NULL); CHKERRQ(ierr);
  //ierr = TDySetDirichletValueFunction(tdy,Pressure,NULL); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Setup initial condition */
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = VecSet(U,91325); CHKERRQ(ierr);
  
  /* Create time stepping and solve */
  TS  ts;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_IMPLICIT); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TDySetIFunction(ts,tdy); CHKERRQ(ierr);
  ierr = TDySetIJacobian(ts,tdy); CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm); CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U); CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  ierr = TSSetUp(ts); CHKERRQ(ierr);
  ierr = TSSolve(ts,U); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U); CHKERRQ(ierr);

  /* Cleanup */
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}
