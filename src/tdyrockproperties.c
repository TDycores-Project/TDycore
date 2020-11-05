#include <private/tdycoreimpl.h>
#include <private/tdyrockpropertiesimpl.h>
#include <tdytimers.h>

PetscErrorCode TDySetSpecificHeatCapacity(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(tdy->Cr,sizeof(PetscReal)*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(tdy->Cr[c-cStart]));
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetRockDensity(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(tdy->rhor,sizeof(PetscReal)*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(tdy->rhor[c-cStart]));
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

void TDyRockDensityFunctionDefault(PetscReal *x, PetscReal *den) {
  *den = 2650.;
}

void TDySpecificHeatCapacityFunctionDefault(PetscReal *x, PetscReal *cr) {
  *cr = 1000.;
}

