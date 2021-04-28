#include "tdycore.h"

void Porosity(PetscReal *x,PetscReal *theta) {
  (*theta) = 0.115;
}

void SpecificHeat(PetscReal *x,PetscReal *theta) {
  (*theta) = 1000.0;
}

void RockDensity(PetscReal *x,PetscReal *theta) {
  (*theta) = 2650.0;
}

void Permeability3D(PetscReal *x,PetscReal *K) {
  K[0] = 1.0e-10; K[1] = 0.0    ; K[2] = 0.0    ;
  K[3] = 0.0    ; K[4] = 1.0e-10; K[5] = 0.0    ;
  K[6] = 0.0    ; K[7] = 0.0    ; K[8] = 1.0e-10;
}

PetscErrorCode PermeabilityFunction3D(TDy tdy, PetscReal *x, PetscReal *K, void *ctx){
  Permeability3D(x, K);
  return 0;
}

void ThermalConductivity3D(PetscReal *x,PetscReal *K) {
  K[0] = 1.0    ; K[1] = 0.0    ; K[2] = 0.0    ;
  K[3] = 0.0    ; K[4] = 1.0    ; K[5] = 0.0    ;
  K[6] = 0.0    ; K[7] = 0.0    ; K[8] = 1.0    ;
}

PetscErrorCode ThermalConductivityFunction3D(TDy tdy, PetscReal *x, PetscReal *K, void *ctx){
  ThermalConductivity3D(x, K);
  return 0;
}

PetscErrorCode Pressure(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) {
  (*p) = 91325;
  PetscFunctionReturn(0);
}

PetscErrorCode Temperature(TDy tdy,PetscReal *x,PetscReal *p,void *ctx) {
  (*p) = 25;
  PetscFunctionReturn(0);
}

PetscErrorCode Forcing(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
  (*f) = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode EnergyForcing(TDy tdy,PetscReal *x,PetscReal *f,void *ctx) {
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

PetscErrorCode PostProcess(TS ts)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  PetscInt       stepi;
  Vec            sol;
  DM             dm;
  char           filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&sol); CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&stepi); CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof filename,"%s-%03D.vtk","solution",stepi); CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  //ierr = VecView(sol,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);

}


int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt nx = 3, ny = 3, nz = 3, dim = 3;
  PetscInt successful_exit_code=0;
  char exofile[256];
  PetscBool exo = PETSC_FALSE;

  ierr = TDyInit(argc, argv); CHKERRQ(ierr);
  TDy  tdy;
  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  TDyMode mode = TH;
  ierr = TDySetMode(tdy,mode); CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);

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

  ierr = TDySetDM(tdy,dm); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  /* Setup problem parameters */
  ierr = TDySetPorosity(tdy,Porosity); CHKERRQ(ierr);
  ierr = TDySetSoilSpecificHeat(tdy,SpecificHeat); CHKERRQ(ierr);
  ierr = TDySetSoilDensity(tdy,RockDensity); CHKERRQ(ierr);
  //ierr = TDySetPermeabilityScalar(tdy,Permeability); CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(tdy,PermeabilityFunction3D,NULL); CHKERRQ(ierr);
  ierr = TDySetThermalConductivityFunction(tdy,ThermalConductivityFunction3D,NULL); CHKERRQ(ierr);
  ierr = TDySetResidualSaturationValuesLocal(tdy,cEnd-cStart,index,residualSat);
  ierr = TDySetForcingFunction(tdy,Forcing,NULL); CHKERRQ(ierr);
  ierr = TDySetEnergyForcingFunction(tdy,EnergyForcing,NULL); CHKERRQ(ierr);
  //ierr = TDySetDirichletValueFunction(tdy,Pressure,NULL); CHKERRQ(ierr);
  //ierr = TDySetTemperatureDirichletValueFunction(tdy,Temperature,NULL); CHKERRQ(ierr);

  ierr = TDySetupNumericalMethods(tdy); CHKERRQ(ierr);

  PetscSection   sec;
  PetscInt num_fields;
  PetscReal total_mass_beg, total_mass_end;
  PetscInt gref, junkInt;
  ierr = DMGetSection(dm,&sec);
  ierr = PetscSectionGetNumFields(sec,&num_fields);

  PetscReal *mass_p,*pres_p, *u_p;
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&pres_p);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&mass_p);CHKERRQ(ierr);

  /* Setup initial condition */
  Vec U;
  PetscReal initial_temperature = 25;
  PetscReal initial_pressure = 91325;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  if (mode == TH && num_fields == 2) {
    ierr = VecGetArray(U, &u_p); CHKERRQ(ierr);
    for (c=0;c<cEnd-cStart;c++) {
      u_p[c*2]   = initial_pressure;
      u_p[c*2+1] = initial_temperature;
    }
    ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
  }
  //VecView(U,PETSC_VIEWER_STDOUT_WORLD);

  if (mode == RICHARDS){
    if (num_fields == 2) {
      ierr = VecGetArray(U,&u_p); CHKERRQ(ierr);
      for (c=0;c<cEnd-cStart;c++) pres_p[c] = u_p[c*2];
      ierr = TDyUpdateState(tdy,pres_p); CHKERRQ(ierr);
      ierr = TDyGetLiquidMassValuesLocal(tdy,&c,mass_p);
      total_mass_beg = 0.0;
      for (c=0;c<cEnd-cStart;c++) {
        u_p[c*2+1] = mass_p[c];
        ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
        if (gref>=0) total_mass_beg += mass_p[c];
      }
      ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(U,&pres_p); CHKERRQ(ierr);
      ierr = TDyUpdateState(tdy,pres_p); CHKERRQ(ierr);
      ierr = VecRestoreArray(U,&pres_p); CHKERRQ(ierr);
      ierr = TDyGetLiquidMassValuesLocal(tdy,&c,mass_p);
      total_mass_beg = 0.0;
      for (c=0;c<cEnd-cStart;c++) {
        ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
        if (gref>=0) total_mass_beg += mass_p[c];
      }
      ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
    }
  }

  if (mode == TH && num_fields == 2){
    ierr = VecGetArray(U,&u_p); CHKERRQ(ierr);
    for (c=0;c<cEnd-cStart;c++) pres_p[c] = u_p[c*2];
    ierr = TDyUpdateState(tdy,u_p); CHKERRQ(ierr);
    ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
    ierr = TDyGetLiquidMassValuesLocal(tdy,&c,mass_p);
    total_mass_beg = 0.0;
    for (c=0;c<cEnd-cStart;c++) {
      ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
      if (gref>=0) total_mass_beg += mass_p[c];
    }

  }

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

  ierr = TSSetPostStep(ts,PostProcess);CHKERRQ(ierr);
  ierr = PostProcess(ts);CHKERRQ(ierr); /* print the initial state */

  ierr = TSSetUp(ts); CHKERRQ(ierr);

  ierr = PetscPrintf(MPI_COMM_SELF,"Solving.\n");CHKERRQ(ierr);
  ierr = TSSolve(ts,U); CHKERRQ(ierr);

  ierr = VecGetArray(U,&u_p); CHKERRQ(ierr);
  for (c=0;c<cEnd-cStart;c++) pres_p[c] = u_p[c*2];
  ierr = TDyUpdateState(tdy,pres_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
  ierr = TDyGetLiquidMassValuesLocal(tdy,&c,mass_p);
  total_mass_end = 0.0;
  for (c=0;c<cEnd-cStart;c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      total_mass_end += mass_p[c];
    }
  }

  PetscReal total_mass_beg_glb, total_mass_end_glb;
  PetscInt rank;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = MPI_Reduce(&total_mass_beg,&total_mass_beg_glb,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);
  ierr = MPI_Reduce(&total_mass_end,&total_mass_end_glb,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);

  if (rank == 0) {
    if (num_fields == 1) printf("TS ODE ");
    else printf("TS DAE ");
    printf("Mass balance: beg = %e; end = %e; change %e\n",total_mass_beg_glb,total_mass_end_glb,total_mass_end_glb-total_mass_beg_glb);
  }

  /* Save regression file */
  ierr = TDyOutputRegression(tdy,U); CHKERRQ(ierr);

  /* Cleanup */
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = PetscFree(mass_p); CHKERRQ(ierr);
  ierr = PetscFree(u_p); CHKERRQ(ierr);
  ierr = PetscFree(pres_p); CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_SELF,"Done!\n");CHKERRQ(ierr);
  ierr = TDyFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}

