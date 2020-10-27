#include "tdycore.h"

void Porosity(double *x,double *theta) {
  (*theta) = 0.115;
}

void SpecificHeatCapacity(double *x,double *theta) {
  (*theta) = 1.0;
}

void RockDensity(double *x,double *theta) {
  (*theta) = 2650.0;
}

void Permeability3D(double *x,double *K) {
  K[0] = 1.0e-9; K[1] = 0.0    ; K[2] = 0.0    ;
  K[3] = 0.0    ; K[4] = 1.0e-9; K[5] = 0.0    ;
  K[6] = 0.0    ; K[7] = 0.0    ; K[8] = 1.0e-9;

  PetscReal s = PetscSinReal(PETSC_PI*x[0]);
  PetscReal t = PetscSinReal(PETSC_PI*x[2]);
  K[0] = K[0] + 1e-10*s*t;
  K[4] = K[4] + 1e-10*s*t;
  K[8] = K[8] + 1e-10*s*t;

  if (x[0] > 0.0 && x[0] < 4.0) {
    if (x[2] > 6.0 && x[2] < 8.0) {
      K[0] = 1.e-12; K[4] = 1.e-12; K[8] = 1.e-12;
    }
  }

}

PetscErrorCode PermeabilityFunction3D(TDy tdy, double *x, double *K, void *ctx){
  Permeability3D(x, K);
  return 0;
}

void ThermalConductivity3D(double *x,double *K) {
  K[0] = 1.e-12   ; K[1] = 0.0    ; K[2] = 0.0    ;
  K[3] = 0.0    ; K[4] = 1.e-12    ; K[5] = 0.0    ;
  K[6] = 0.0    ; K[7] = 0.0    ; K[8] = 1.e-12    ;
}

PetscErrorCode ThermalConductivityFunction3D(TDy tdy, double *x, double *K, void *ctx){
  ThermalConductivity3D(x, K);
  return 0;
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
  ierr = VecView(sol,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);

}


int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt nx = 31, ny = 1, nz = 31, dim = 3;
  PetscInt successful_exit_code=0;
  char exofile[256];
  PetscBool exo = PETSC_FALSE;
  ierr = TDyInit(argc, argv); CHKERRQ(ierr);
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
    const PetscReal upper[3] = {10.0,1.0,10.0};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
			       NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
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
  TDyMode mode = TH;
  ierr = TDyCreateWithDM(dm,&tdy); CHKERRQ(ierr);
  ierr = TDySetMode(tdy,mode); CHKERRQ(ierr);
  ierr = TDySetPorosity(tdy,Porosity); CHKERRQ(ierr);
  ierr = TDySetSpecificHeatCapacity(tdy,SpecificHeatCapacity); CHKERRQ(ierr);
  ierr = TDySetRockDensity(tdy,RockDensity); CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(tdy,PermeabilityFunction3D,NULL); CHKERRQ(ierr);
  ierr = TDySetThermalConductivityFunction(tdy,ThermalConductivityFunction3D,NULL); CHKERRQ(ierr);
  ierr = TDySetResidualSaturationValuesLocal(tdy,cEnd-cStart,index,residualSat);
  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  PetscInt ncells = nx*ny*nz;
  PetscReal forcing_vec[ncells], energy_forcing_vec[ncells];
  for (c=0;c<ncells;c++) {
    index[c] = c;
    forcing_vec[c] = 0.0;
    energy_forcing_vec[c] = 0.0;
  }
 

  for (c=24;c<26;c++){
    forcing_vec[ncells-1-c]   = 1.e1;
    energy_forcing_vec[ncells-1-c]   = forcing_vec[ncells-1-c]*1.e1;

  }

  ierr = TDySetSourceSinkValuesLocal(tdy,ncells,index,forcing_vec);
  ierr = TDySetEnergySourceSinkValuesLocal(tdy,ncells,index,energy_forcing_vec);

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
  PetscReal initial_temperature = 5;
  PetscReal initial_pressure = 91235;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  FILE *initpress_file;
  initpress_file = fopen("init_press.dat", "r");
  if (mode == TH && num_fields == 2) {
    ierr = VecGetArray(U, &u_p); CHKERRQ(ierr);
    for (c=0;c<cEnd-cStart;c++) {
      fscanf(initpress_file, "%lf", &u_p[c*2]);
      u_p[c*2+1] = initial_temperature;
    }
    ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
  }
  VecView(U,PETSC_VIEWER_STDOUT_WORLD);

  /* Create time stepping and solve */
  TS  ts;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_IMPLICIT); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TDySetIFunction(ts,tdy); CHKERRQ(ierr);
  ierr = TDySetIJacobian(ts,tdy); CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm); CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U); CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,100); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

  ierr = TSSetPostStep(ts,PostProcess);CHKERRQ(ierr);
  ierr = PostProcess(ts);CHKERRQ(ierr); /* print the initial state */

  ierr = TSSetUp(ts); CHKERRQ(ierr);

  ierr = PetscPrintf(MPI_COMM_SELF,"Solving.\n");CHKERRQ(ierr);
  ierr = TSSolve(ts,U); CHKERRQ(ierr);


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

/*
./th_source_sink -ts_view -ts_monitor -ts_final_time 1e5 -ts_adapt_type basic -ts_max_snes_failures -1 -ts_max_steps 300
*/
