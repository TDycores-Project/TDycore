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
PetscErrorCode TDyDMDestroy(TDyDM *tdydm) {

  PetscErrorCode ierr;

  ierr = DMDestroy(&tdydm->dm); CHKERRQ(ierr);
  ierr = TDyUGDMDestroy(&tdydm->ugdm); CHKERRQ(ierr);

  free (tdydm);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

PetscErrorCode TDyDMCreateFromUGrid(PetscInt ndof, TDyUGrid *ugrid, TDyDM *tdydm) {

  PetscErrorCode ierr;

  tdydm->dmtype = TDYCORE_DM_TYPE;

  ierr = TDyUGDMCreateFromUGrid(ndof, ugrid, &tdydm->ugdm); CHKERRQ(ierr);

  ierr = DMShellCreate(PETSC_COMM_WORLD, &tdydm->dm); CHKERRQ(ierr);
  ierr = DMShellSetGlobalToLocalVecScatter(tdydm->dm, (&tdydm->ugdm)->Scatter_GlobalCells_to_LocalCells); CHKERRQ(ierr);
  ierr = DMShellSetLocalToGlobalVecScatter(tdydm->dm, (&tdydm->ugdm)->Scatter_LocalCells_to_GlobalCells); CHKERRQ(ierr);
  ierr = DMShellSetLocalToLocalVecScatter(tdydm->dm, (&tdydm->ugdm)->Scatter_LocalCells_to_LocalCells); CHKERRQ(ierr);

  Vec global_vec, local_vec;
  ierr = TDyUGDMCreateGlobalVec(ndof, ugrid->num_cells_local, &tdydm->ugdm, &global_vec); CHKERRQ(ierr);
  ierr = TDyUGDMCreateLocalVec(ndof, ugrid->num_cells_global, &tdydm->ugdm, &local_vec); CHKERRQ(ierr);

  ierr = DMShellSetGlobalVector(tdydm->dm, global_vec); CHKERRQ(ierr);
  ierr = DMShellSetLocalVector(tdydm->dm, local_vec); CHKERRQ(ierr);

  ierr = VecDestroy(&global_vec); CHKERRQ(ierr);
  ierr = VecDestroy(&local_vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

