#include <private/tdydmimpl.h>
#include <tdytimers.h>

PetscErrorCode TDyDistributeDM(DM *dm) {
  DM dmDist;
  PetscErrorCode ierr;

  /* Define 1 DOF on cell center of each cell */
  PetscInt dim;
  PetscFE fe;
  ierr = DMGetDimension(*dm, &dim); CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "p_", -1, &fe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "p");CHKERRQ(ierr);
  ierr = DMSetField(*dm, 0, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(*dm); CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  ierr = DMSetUseNatural(*dm, PETSC_TRUE); CHKERRQ(ierr);

  ierr = DMPlexDistribute(*dm, 1, NULL, &dmDist); CHKERRQ(ierr);
  if (dmDist) {
    DMDestroy(dm); CHKERRQ(ierr);
    *dm = dmDist;
  }
  ierr = DMSetFromOptions(*dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
