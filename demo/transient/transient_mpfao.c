#include "tdycore.h"
#include "tdydmperturbation.h"

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

int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt successful_exit_code=0;
  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
			   "Transient Options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  /* Setup problem parameters */
  TDy  tdy;
  ierr = TDyCreate(&tdy); CHKERRQ(ierr);
  DM dm;
  ierr = TDyGetDM(tdy,&dm); CHKERRQ(ierr);
  PetscInt c,cStart,cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscReal residualSat[cEnd-cStart];
  PetscInt index[cEnd-cStart];
  for (c=0;c<cEnd-cStart;c++) {
    index[c] = c;
    residualSat[c] = 0.115;
  }
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
  //VecView(U,PETSC_VIEWER_STDOUT_WORLD);

  PetscSection   sec;
  PetscInt num_fields;
  PetscReal total_mass_beg, total_mass_end;
  PetscInt gref, junkInt;
  ierr = DMGetSection(dm, &sec);
  ierr = PetscSectionGetNumFields(sec, &num_fields);

  PetscReal *mass_p,*pres_p, *u_p;
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&pres_p);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&mass_p);CHKERRQ(ierr);

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
  ierr = TSSetUp(ts); CHKERRQ(ierr);
  ierr = TSSolve(ts,U); CHKERRQ(ierr);

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
  ierr = PetscFree(pres_p); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);
}

