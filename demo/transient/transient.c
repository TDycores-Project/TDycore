#include "tdycore.h"

void Porosity(PetscReal *x,PetscReal *theta) {
  (*theta) = 0.5;
}

void Permeability(PetscReal *x,PetscReal *K) {
  (*K) = 1e-10;
}

PetscErrorCode Pressure(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) {
  (*p) = 91325;
  PetscFunctionReturn(0);
}

PetscErrorCode Forcing(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
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
  PetscInt N = 4, dim = 3;
  PetscInt successful_exit_code=0;
  char exofile[256];
  PetscBool exo = PETSC_FALSE;

  ierr = TDyInit(argc, argv); CHKERRQ(ierr);
  TDy  tdy;
  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,WY); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
			   "Transient Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D",
			 "",N,&N,NULL); CHKERRQ(ierr);
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
    const PetscInt  faces[3] = {N,N,N  };
    const PetscReal lower[3] = {-0.5,-0.5,-0.5};
    const PetscReal upper[3] = {+0.5,+0.5,+0.5};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
			       NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
    ierr = PerturbInteriorVertices(dm,1./N); CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  ierr = TDySetDM(tdy,dm); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Setup problem parameters */
  ierr = TDySetPorosity(tdy,Porosity); CHKERRQ(ierr);
  ierr = TDySetPermeabilityScalar(tdy,Permeability); CHKERRQ(ierr);
  ierr = TDySetForcingFunction(tdy,Forcing,NULL); CHKERRQ(ierr);
  ierr = TDySetDirichletValueFunction(tdy,Pressure,NULL); CHKERRQ(ierr);

  ierr = TDySetupNumericalMethods(tdy); CHKERRQ(ierr);

  /* Setup initial condition */
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = VecSet(U,91325); CHKERRQ(ierr);
  
  /* Create time stepping and solve */
  TS  ts;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,TDyWYResidual,tdy); CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm); CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U); CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  ierr = TSSetUp(ts); CHKERRQ(ierr);
  ierr = TSSolve(ts,U); CHKERRQ(ierr);

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U); CHKERRQ(ierr);

  /* Cleanup */
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = TDyFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}
