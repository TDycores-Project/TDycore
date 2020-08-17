#include "richards.h"
#include "private/tdydriverimpl.h"

//#define DEBUG
#define R (*r)

PetscErrorCode RichardsCreate(TDyDriver tdydriver) {
  Richards R;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  R = (Richards) malloc(sizeof(struct Richards));
  tdydriver->driverctx = (void*)malloc(sizeof(struct Richards));
  ierr = TDyCreate(&(R->tdy)); CHKERRQ(ierr);
  rr = (void*)r; 
  PetscFunctionReturn(0);
}

PetscErrorCode RichardsPrintVec(Vec v,char *prefix, int print_count) {
  char word[32];
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (print_count >= 0)
    sprintf(word,"%s_%d.txt",prefix,print_count);
  else
    sprintf(word,"%s.txt",prefix);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); 
         CHKERRQ(ierr);
  ierr = VecView(v,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PorosityFunction(TDy tdy, double *x, double *por, void *ctx) {
  PetscErrorCode ierr;
  PetscInt dim;
  PetscFunctionBegin;
  TDy tdya;
  *por = 0.25;
  PetscFunctionReturn(0);
}

PetscErrorCode PermeabilityFunction(TDy tdy, double *x, double *K, void *ctx) {
  PetscErrorCode ierr;
  PetscInt dim;
  PetscFunctionBegin;
  TDy tdya;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  for (int j=0; j<dim; j++) {
    for (int i=0; i<dim; i++) {
      K[j*dim+i] = 0.;
    }
  }
  for (int i=0; i<dim; i++)
    K[i*dim+i] = 1.e-12;
  return 0;
  PetscFunctionReturn(0);
}

PetscErrorCode RichardsSNESPostCheck(SNESLineSearch linesearch, 
                                     Vec X, Vec Y, Vec W,
                                     PetscBool *changed_Y, 
                                     PetscBool *changed_W,void *ctx) {
  PetscFunctionBegin;
  Richards r = (Richards)ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode RichardsInitialize(void *rr) {
  PetscRandom rand;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Richards R;
  r = (Richards*)rr;

  R->U = PETSC_NULL;
  R->snes = PETSC_NULL;
  R->io_process = PETSC_FALSE;
  R->print_intermediate = PETSC_FALSE;
  R->time = 0.;
  R->dtime = 1.;
  R->istep = 0;
  PetscScalar final_time_in_years = 0.0001;
  R->final_time = final_time_in_years*365.*24.*3600.;

  PetscReal gravity[3];
  gravity[0] = 0.0; gravity[1] = 0.0; gravity[2] = 9.8068;
  ierr = TDySetGravityVector(R->tdy,gravity);
  ierr = TDySetPorosityFunction(R->tdy,PorosityFunction,PETSC_NULL); 
         CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(R->tdy,PermeabilityFunction,PETSC_NULL); 
         CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(R->tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(R->tdy); CHKERRQ(ierr); 

  DM dm;
  ierr = TDyGetDM(R->tdy,&dm); CHKERRQ(ierr);
  PetscViewer viewer;
#if defined(DEBUG)
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"dm.txt",&viewer); 
         CHKERRQ(ierr);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif
  
  ierr = DMCreateGlobalVector(dm,&(R->U)); CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand); CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,1.e4,1.e6); CHKERRQ(ierr);
  ierr = VecSetRandom(R->U,rand); CHKERRQ(ierr);
//  ierr = VecSet(R->U,2.e5); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&(R->snes)); CHKERRQ(ierr);
  ierr = TDySetSNESFunction(R->snes,R->tdy); CHKERRQ(ierr);
  ierr = TDySetSNESJacobian(R->snes,R->tdy); CHKERRQ(ierr);
  SNESLineSearch linesearch;
  ierr = SNESGetLineSearch(R->snes,&linesearch); CHKERRQ(ierr);
  ierr = SNESLineSearchSetPostCheck(linesearch,RichardsSNESPostCheck,&r);
         CHKERRQ(ierr);
  ierr = SNESSetFromOptions(R->snes); CHKERRQ(ierr);
  ierr = TDySetInitialSolutionForSNESSolver(R->tdy,R->U); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode RichardsRunToTime(void *rr,PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscViewer viewer;
  PetscScalar *a;

  Richards *r;
  r = (Richards*)rr;

  while (R->time < sync_time) {
    ierr = TDySetDtimeForSNESSolver(R->tdy,R->dtime); CHKERRQ(ierr);
    ierr = TDyPreSolveSNESSolver(R->tdy); CHKERRQ(ierr);
    ierr = SNESSolve(R->snes,PETSC_NULL,R->U); CHKERRQ(ierr);
    ierr = TDyPostSolveSNESSolver(R->tdy,R->U); CHKERRQ(ierr);
    R->time += R->dtime;
    R->dtime *= 1.25;
    PetscInt dt_for_sync = sync_time-R->time;
    if (R->dtime > dt_for_sync) {R->dtime = dt_for_sync; R->time = sync_time;}
    R->istep++;
    PetscInt nit;
    ierr = SNESGetLinearSolveIterations(R->snes,&nit); CHKERRQ(ierr);
    if (R->io_process)
      printf("Time step %d: time = %f dt = %f ni=%d\n",
             R->istep,R->time,R->dtime,nit);
    if (R->print_intermediate)
      ierr = RichardsPrintVec(R->U,"soln",R->istep); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode RichardsDestroy(void *rr) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Richards *r;
  r = (Richards*)rr;

  ierr = VecDestroy(&(R->U)); CHKERRQ(ierr);
  ierr = TDyDestroy(&(R->tdy)); CHKERRQ(ierr);
  free(R);
  rr = PETSC_NULL;
  PetscFunctionReturn(0);
}
#undef R
