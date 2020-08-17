#include "tdydriver.h"

PetscErrorCode TDyDriverCreate(TDyDriver *_tdydriver) {
  TDyDriver tdydriver;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscHeaderCreate(tdydriver,TDYDRIVER_CLASSID,"TDyDriver",
                           "TDyDriver","TDyDriver",PETSC_COMM_WORLD,
                           TDyDriverDestroy,TDyDriverView); CHKERRQ(ierr);
  _tdydriver = &tdydriver;
  Richards r = NULL;
  ierr = RichardsCreate(&r); CHKERRQ(ierr);
  tdydriver->driverctx = &r;
  tdydriver->ops->create = &RichardsCreate;
  tdydriver->ops->destroy = &RichardsDestroy;
  tdydriver->ops->setup = &RichardsInitialize;
  tdydriver->ops->runtotime = &RichardsRunToTime;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyDriverView(TDyDriver tdydriver) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyDriverDestroy(TDyDriver *tdydriver) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = (*tdydriver)->ops->destroy((*tdydriver)->driverctx); CHKERRQ(ierr);
  free(*tdydriver);
  *tdydriver = PETSC_NULL;
  PetscFunctionReturn(0);
}
