#include <private/tdydmimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyugridimpl.h>

/// Allocates memory and intializes a TDyDM struct
/// @param [out] tdydm A TDyDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDMCreate(TDyDM **tdydm) {

  PetscErrorCode ierr;

  ierr = TDyAlloc(sizeof(TDyDM), tdydm); CHKERRQ(ierr);

  ierr = TDyUGDMCreate(&((*tdydm)->ugdm)); CHKERRQ(ierr);
  (*tdydm)->dmtype = PLEX_TYPE;

  PetscFunctionReturn(0);
}

/// Deallocates a TDyDM struct
/// @param [inout] tdydm A TDyDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDMDestroy(TDyDM *tdydm) {

  PetscErrorCode ierr;

  ierr = DMDestroy(&tdydm->dm); CHKERRQ(ierr);
  ierr = TDyUGDMDestroy(tdydm->ugdm); CHKERRQ(ierr);

  TDyFree(tdydm);

  PetscFunctionReturn(0);
}

/// Creates a TDyDM from TDyUGrid. This is done when TDycore manages
/// its own DM via a DMShell
///
/// @param [in] ndof Number of degree of freedoms
/// @param [in] ugrid A TDyUGrid struct
/// @param [inout] tdydm A TDyDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDMCreateFromUGrid(PetscInt ndof, TDyUGrid *ugrid, TDyDM *tdydm) {

  PetscErrorCode ierr;

  tdydm->dmtype = TDYCORE_DM_TYPE;

  ierr = TDyUGDMCreateFromUGrid(ndof, ugrid, tdydm->ugdm); CHKERRQ(ierr);

  ierr = DMShellCreate(PETSC_COMM_WORLD, &tdydm->dm); CHKERRQ(ierr);
  ierr = DMShellSetGlobalToLocalVecScatter(tdydm->dm, (tdydm->ugdm)->Scatter_GlobalCells_to_LocalCells); CHKERRQ(ierr);
  ierr = DMShellSetLocalToGlobalVecScatter(tdydm->dm, (tdydm->ugdm)->Scatter_LocalCells_to_GlobalCells); CHKERRQ(ierr);
  ierr = DMShellSetLocalToLocalVecScatter(tdydm->dm, (tdydm->ugdm)->Scatter_LocalCells_to_LocalCells); CHKERRQ(ierr);

  Vec global_vec, local_vec;
  ierr = TDyUGDMCreateGlobalVec(ndof, ugrid->num_cells_local, tdydm->ugdm, &global_vec); CHKERRQ(ierr);
  ierr = TDyUGDMCreateLocalVec(ndof, ugrid->num_cells_global, tdydm->ugdm, &local_vec); CHKERRQ(ierr);

  ierr = DMShellSetGlobalVector(tdydm->dm, global_vec); CHKERRQ(ierr);
  ierr = DMShellSetLocalVector(tdydm->dm, local_vec); CHKERRQ(ierr);

  ierr = VecDestroy(&global_vec); CHKERRQ(ierr);
  ierr = VecDestroy(&local_vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

