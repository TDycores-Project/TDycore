#include <private/tdydmimpl.h>
#include <private/tdyugridimpl.h>

PetscErrorCode TDyDMCreate(TDyDM *tdydm) {

  PetscErrorCode ierr;

  tdydm = malloc(sizeof(TDyDM));

  tdydm->dm = NULL;

  ierr = TDyUGDMCreate(&tdydm->ugdm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

PetscErrorCode TDyDMCreateFromUGrid(PetscInt ndof, TDyUGrid *ugrid, TDyDM *tdydm) {

  PetscErrorCode ierr;

  ierr = TDyUGDMCreateFromUGrid(ndof, ugrid, &tdydm->ugdm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

