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

PetscErrorCode TDyDMCreateFromPFLOTRANMesh(TDyUGrid *ugrid, TDyDM *tdydm, const char *mesh_file) {

  PetscErrorCode ierr;

  //ierr = TDyUGDMCreateFromPFLOTRANMesh(ugrid, &tdydm->ugdm, mesh_file); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

