#include "richards.h"

//#define DEBUG

PetscErrorCode RichardsCreate(Richards *r) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *r = (Richards) malloc(sizeof(struct Richards));
  ierr = TDyCreate(&((*r)->tdy)); CHKERRQ(ierr);
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

PetscErrorCode RichardsInitialize(Richards r) {
  PetscRandom rand;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  r->U = PETSC_NULL;
  r->snes = PETSC_NULL;
  r->io_process = PETSC_FALSE;
  r->print_intermediate = PETSC_FALSE;
  r->time = 0.;
  r->dtime = 1.;
  r->istep = 0;
  PetscScalar final_time_in_years = 0.0001;
  r->final_time = final_time_in_years*365.*24.*3600.;

  PetscReal gravity[3];
  gravity[0] = 0.0; gravity[1] = 0.0; gravity[2] = 9.8068;
  ierr = TDySetGravityVector(r->tdy,gravity);
  ierr = TDySetPorosityFunction(r->tdy,PorosityFunction,PETSC_NULL); 
         CHKERRQ(ierr);
  ierr = TDySetPermeabilityFunction(r->tdy,PermeabilityFunction,PETSC_NULL); 
         CHKERRQ(ierr);
  ierr = TDySetDiscretizationMethod(r->tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(r->tdy); CHKERRQ(ierr); 

  DM dm;
  ierr = TDyGetDM(r->tdy,&dm); CHKERRQ(ierr);
  PetscViewer viewer;
#if defined(DEBUG)
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"dm.txt",&viewer); 
         CHKERRQ(ierr);
  ierr = DMView(dm,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif
  
  ierr = DMCreateGlobalVector(dm,&(r->U)); CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand); CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,1.e4,1.e6); CHKERRQ(ierr);
  ierr = VecSetRandom(r->U,rand); CHKERRQ(ierr);
//  ierr = VecSet(r->U,2.e5); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&(r->snes)); CHKERRQ(ierr);
  ierr = TDySetSNESFunction(r->snes,r->tdy); CHKERRQ(ierr);
  ierr = TDySetSNESJacobian(r->snes,r->tdy); CHKERRQ(ierr);
  SNESLineSearch linesearch;
  ierr = SNESGetLineSearch(r->snes,&linesearch); CHKERRQ(ierr);
  ierr = SNESLineSearchSetPostCheck(linesearch,RichardsSNESPostCheck,&r);
         CHKERRQ(ierr);
  ierr = SNESSetFromOptions(r->snes); CHKERRQ(ierr);
  ierr = TDySetInitialSolutionForSNESSolver(r->tdy,r->U); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode RichardsRunToTime(Richards r,PetscReal sync_time) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscViewer viewer;
  PetscScalar *a;

  while (r->time < sync_time) {
    ierr = TDySetDtimeForSNESSolver(r->tdy,r->dtime); CHKERRQ(ierr);
    ierr = TDyPreSolveSNESSolver(r->tdy); CHKERRQ(ierr);
    ierr = SNESSolve(r->snes,PETSC_NULL,r->U); CHKERRQ(ierr);
    ierr = TDyPostSolveSNESSolver(r->tdy,r->U); CHKERRQ(ierr);
    r->time += r->dtime;
    r->dtime *= 1.25;
    PetscInt dt_for_sync = sync_time-r->time;
    if (r->dtime > dt_for_sync) {r->dtime = dt_for_sync; r->time = sync_time;}
    r->istep++;
    PetscInt nit;
    ierr = SNESGetLinearSolveIterations(r->snes,&nit); CHKERRQ(ierr);
    if (r->io_process)
      printf("Time step %d: time = %f dt = %f ni=%d\n",
             r->istep,r->time,r->dtime,nit);
    if (r->print_intermediate)
      ierr = RichardsPrintVec(r->U,"soln",r->istep); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode RichardsDestroy(Richards *r) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy(&((*r)->U)); CHKERRQ(ierr);
  ierr = TDyDestroy(&((*r)->tdy)); CHKERRQ(ierr);
  free(*r);
  *r = PETSC_NULL;
  PetscFunctionReturn(0);
}
