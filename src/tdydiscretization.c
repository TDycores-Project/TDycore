#include <private/tdycoreimpl.h>

/* -------------------------------------------------------------------------- */
/// Creates a PETSc global vector
///
/// @param [in] tdy     A TDy struct
/// @param [out] vector A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyCreateGlobalVector(TDy tdy, Vec *vector){

  PetscFunctionBegin;
  DM dm = tdy->dm;
  PetscErrorCode ierr;

  ierr = DMCreateGlobalVector(dm, vector); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a PETSc local vector
///
/// @param [in] tdy     A TDy struct
/// @param [out] vector A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyCreateLocalVector(TDy tdy, Vec *vector){

  PetscFunctionBegin;
  DM dm = tdy->dm;
  PetscErrorCode ierr;

  ierr = DMCreateLocalVector(dm, vector); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a PETSc natural vector
///
/// @param [in] tdy     A TDy struct
/// @param [out] vector A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyCreateNaturalVector(TDy tdy, Vec *vector){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = TDyCreateGlobalVector(tdy, vector); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Duplicates a PETSc vector
///
/// @param [in] tdy      A TDy struct
/// @param [in] vector1  A PETSc vector
/// @param [out] vector2 A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDuplicateVector(TDy tdy, Vec vector1, Vec *vector2){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = VecDuplicate(vector1, vector2); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a Jacobian matrix
///
/// @param [in] tdy     A TDy struct
/// @param [out] matrix A PETSc matrix
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyCreateJacobianMatrix(TDy tdy, Mat *matrix){

  PetscFunctionBegin;
  DM dm = tdy->dm;
  PetscErrorCode ierr;

  ierr = DMCreateMatrix(dm, matrix); CHKERRQ(ierr);
  ierr = MatSetOption(*matrix, MAT_KEEP_NONZERO_PATTERN, PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(*matrix, MAT_ROW_ORIENTED, PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(*matrix, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(*matrix, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Performs scatter of a global vector to a natural vector
///
/// @param [in] tdy      A TDy struct
/// @param [in] global   A PETSc vector
/// @param [out] natural A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyGlobalToNatural(TDy tdy, Vec global, Vec natural){

  PetscFunctionBegin;
  DM dm = tdy->dm;
  PetscBool useNatural;
  PetscErrorCode ierr;

  ierr = DMGetUseNatural(dm, &useNatural); CHKERRQ(ierr);
  if (!useNatural) {
    PetscPrintf(PETSC_COMM_WORLD,"TDyGlobalToNatural cannot be performed as DMGetUseNatural is false");
    exit(0);
  }

  ierr = DMPlexGlobalToNaturalBegin(dm, global, natural);CHKERRQ(ierr);
  ierr = DMPlexGlobalToNaturalEnd(dm, global, natural);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Performs scatter of a global vector to a local vector
///
/// @param [in] tdy    A TDy struct
/// @param [in] global A PETSc vector
/// @param [out] local A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyGlobalToLocal(TDy tdy, Vec global, Vec local){

  PetscFunctionBegin;
  DM dm = tdy->dm;
  PetscErrorCode ierr;

  ierr = DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/// Performs scatter of a natural vector to a global vector
///
/// @param [in] tdy    A TDy struct
/// @param [in] global A PETSc vector
/// @param [out] local A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyNaturalToGlobal(TDy tdy, Vec natural, Vec global){

  PetscFunctionBegin;
  DM dm = tdy->dm;
  PetscBool useNatural;
  PetscErrorCode ierr;

  ierr = DMGetUseNatural(dm, &useNatural); CHKERRQ(ierr);
  if (!useNatural) {
    PetscPrintf(PETSC_COMM_WORLD,"TDyNaturalToGlobal cannot be performed as DMGetUseNatural is false");
    exit(0);
  }

  ierr = DMPlexNaturalToGlobalBegin(dm, natural, global);CHKERRQ(ierr);
  ierr = DMPlexNaturalToGlobalEnd(dm, natural, global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
