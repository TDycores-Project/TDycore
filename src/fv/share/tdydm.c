#include <private/tdydmimpl.h>

PetscErrorCode TDyDMCreate(TDyDM *tdydm) {

  PetscErrorCode ierr;

  tdydm = malloc(sizeof(TDyDM));

  tdydm->dm = NULL;

  ierr = TDyUGDMCreate(&tdydm->ugdm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
