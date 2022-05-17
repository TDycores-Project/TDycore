#include <private/tdycoreimpl.h>
#include <private/tdydmimpl.h>
#include <private/tdydiscretizationimpl.h>

PetscErrorCode TDyDiscretizationCreate(TDyDiscretizationType **discretization) {

  PetscErrorCode ierr;

  TDyDiscretizationType *discretization_ptr;
  discretization_ptr = (TDyDiscretizationType*) malloc(sizeof(TDyDiscretizationType));
  *discretization = discretization_ptr;

  ierr = TDyDMCreate(&((*discretization)->tdydm));

  ierr = TDyUGridCreate(&((*discretization)->ugrid));

  (*discretization)->tmp = -10;

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyDiscretizationCreateFromPFLOTRANMesh(const char *mesh_file, PetscInt ndof, TDyDiscretizationType *discretization) {

  PetscErrorCode ierr;

  ierr = TDyUGridCreateFromPFLOTRANMesh(discretization->ugrid, mesh_file); CHKERRQ(ierr);

  ierr = TDyDMCreateFromUGrid(ndof, discretization->ugrid, discretization->tdydm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode TDyDiscretizationDestroy(TDyDiscretizationType *discretization) {

  PetscErrorCode ierr;

  discretization = malloc(sizeof(TDyDiscretization));

  ierr = TDyDMDestroy (discretization->tdydm); CHKERRQ(ierr);
  free(discretization);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyDiscretizationGetTDyDM(TDyDiscretizationType *discretization, TDyDM *tdydm) {

  tdydm = (discretization->tdydm);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyDiscretizationGetDM(TDyDiscretizationType *discretization, DM *dm) {

  TDyDM *tdydm = discretization->tdydm;
  *dm = tdydm->dm;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a PETSc global vector. A  global vector is parallel, and lays out
/// data according to the global section. This section assigns dofs to the
/// points of the local Plex, but leaves out any point that is in the pointSF
/// because this means it is not owned by the process.
///
/// @param [in] tdy     A TDy struct
/// @param [out] vector A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationCreateGlobalVector(TDyDiscretizationType *discretization, Vec *vector){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;
  DM dm = tdydm->dm;
  ierr = DMCreateGlobalVector(dm, vector); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a PETSc local vector. Local: A local vector is serial, and lays out
/// data according to the local section. This section assigns dofs to the points
/// of the local Plex.
///
/// @param [in] tdy     A TDy struct
/// @param [out] vector A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationCreateLocalVector(TDyDiscretizationType *discretization, Vec *vector){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;
  DM dm = tdydm->dm;
  ierr = DMCreateLocalVector(dm, vector); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a PETSc natural vector. A natural vector is a permutation of a
/// global vector, usually to match the input ordering.
///
/// @param [in] tdy     A TDy struct
/// @param [out] vector A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationCreateNaturalVector(TDyDiscretizationType *discretization, Vec *vector){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;

  switch (tdydm->dmtype) {
    case PLEX_TYPE:
      ierr = TDyDiscretizationCreateGlobalVector(discretization, vector); CHKERRQ(ierr);
      break;
    case TDYCORE_DM_TYPE:
      //ierr = TDyUGDMCreateNaturalVec(tdydm->ugdm, vector); CHKERRQ(ierr);
      break;
    default:
      break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Creates a Jacobian matrix
///
/// @param [in] tdy     A TDy struct
/// @param [out] matrix A PETSc matrix
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationCreateJacobianMatrix(TDyDiscretizationType *discretization, Mat *matrix){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;
  DM dm = tdydm->dm;

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
/// @param [in] tdydm    A TDyDM struct
/// @param [in] global   A PETSc vector
/// @param [out] natural A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationGlobalToNatural(TDyDiscretizationType *discretization, Vec global, Vec natural){

  PetscFunctionBegin;
  PetscBool useNatural;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;
  DM dm = tdydm->dm;

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
/// @param [in] tdydm  A TDyDM struct
/// @param [in] global A PETSc vector
/// @param [out] local A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationGlobalToLocal(TDyDiscretizationType *discretization, Vec global, Vec local){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;
  DM dm = tdydm->dm;

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
PetscErrorCode TDyDiscretizationNaturalToGlobal(TDyDiscretizationType *discretization, Vec natural, Vec global){

  PetscFunctionBegin;
  PetscBool useNatural;
  PetscErrorCode ierr;

  TDyDM *tdydm = discretization->tdydm;
  DM dm = tdydm->dm;

  ierr = DMGetUseNatural(dm, &useNatural); CHKERRQ(ierr);
  if (!useNatural) {
    PetscPrintf(PETSC_COMM_WORLD,"TDyNaturalToGlobal cannot be performed as DMGetUseNatural is false");
    exit(0);
  }

  ierr = DMPlexNaturalToGlobalBegin(dm, natural, global);CHKERRQ(ierr);
  ierr = DMPlexNaturalToGlobalEnd(dm, natural, global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Performs scatter of a natural vector to a local vector
///
/// @param [in] tdy     A TDy struct
/// @param [in] natural A PETSc vector
/// @param [out] local   A PETSc vector
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyDiscretizationNaturaltoLocal(TDyDiscretizationType *discretization,Vec natural, Vec *local) {

  PetscFunctionBegin;

  PetscErrorCode ierr;
  Vec global;
  
  ierr = TDyDiscretizationCreateGlobalVector(discretization, &global);CHKERRQ(ierr);

  ierr = TDyDiscretizationNaturalToGlobal(discretization, natural, global);CHKERRQ(ierr);
  ierr = TDyDiscretizationGlobalToLocal(discretization, global, *local); CHKERRQ(ierr);

  ierr = VecDestroy(&global); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
