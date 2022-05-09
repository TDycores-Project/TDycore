#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <private/tdyugdmimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUGDMCreate(TDyUGDM *ugdm){
  PetscFunctionBegin;

  ugdm = malloc(sizeof(TDyUGDM));

  ugdm->IS_GhostedCells_in_LocalOrder = NULL;
  ugdm->IS_GhostedCells_in_PetscOrder = NULL;

  ugdm->IS_LocalCells_in_LocalOrder = NULL;
  ugdm->IS_LocalCells_in_PetscOrder = NULL;

  ugdm->IS_GhostCells_in_LocalOrder = NULL;
  ugdm->IS_GhostCells_in_PetscOrder = NULL;

  ugdm->Scatter_LocalCells_to_GlobalCells = NULL;
  ugdm->Scatter_LocalCells_to_LocalCells = NULL;
  ugdm->Scatter_LocalCells_to_LocalCells = NULL;
  ugdm->Scatter_GlobalCells_to_NaturalCells = NULL;

  ugdm->Mapping_LocalCells_to_NaturalCells = NULL;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode CreateVectors(PetscInt ngmax, PetscInt nlmax, PetscInt ndof, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  // 1. Create local vector.
  // Note: The size of the Vec is 'ngmax' (= local cells + ghost cells)
  ierr = VecCreate(PETSC_COMM_WORLD, &ugdm->LocalVec); CHKERRQ(ierr);
  ierr = VecSetSizes(ugdm->LocalVec, ngmax*ndof, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(ugdm->LocalVec, ndof); CHKERRQ(ierr);
  ierr = VecSetFromOptions(ugdm->LocalVec); CHKERRQ(ierr);

  // 2. Create global vector. Note: The size of the Vec is 'nlmax'
  ierr = VecCreate(PETSC_COMM_WORLD, &ugdm->GlobalVec); CHKERRQ(ierr);
  ierr = VecSetSizes(ugdm->GlobalVec, nlmax*ndof, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(ugdm->GlobalVec, ndof); CHKERRQ(ierr);
  ierr = VecSetFromOptions(ugdm->GlobalVec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode CreateLocalOrderIS(PetscInt ngmax, PetscInt nlmax, PetscInt nghost, PetscInt ndof, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt idxL[nlmax];
  for (PetscInt i=0; i<nlmax; i++) {
    idxL[i] = i;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nlmax, idxL, PETSC_COPY_VALUES, &ugdm->IS_LocalCells_in_LocalOrder); CHKERRQ(ierr);
  ierr = TDySavePetscISAsASCII(ugdm->IS_LocalCells_in_LocalOrder,"is_local_local_1.out");

  PetscInt idxG[ngmax];
  for (PetscInt i=0; i<ngmax; i++) {
    idxG[i] = i;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, ngmax, idxG, PETSC_COPY_VALUES, &ugdm->IS_GhostedCells_in_LocalOrder); CHKERRQ(ierr);
  ierr = TDySavePetscISAsASCII(ugdm->IS_GhostedCells_in_LocalOrder,"is_ghosted_local_1.out");

  PetscInt idxGhost[nghost];
  for (PetscInt i=0; i<nghost; i++) {
    idxGhost[i] = nlmax+i;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nghost, idxGhost, PETSC_COPY_VALUES, &ugdm->IS_GhostCells_in_LocalOrder); CHKERRQ(ierr);
  ierr = TDySavePetscISAsASCII(ugdm->IS_GhostCells_in_LocalOrder,"is_ghosts_local_1.out");

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUGDMCreateFromUGrid(PetscInt ndof, TDyUGrid *ugrid, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  // Create vectors
  ierr = CreateVectors(ugrid->num_cells_global, ugrid->num_cells_local, ndof, ugdm); CHKERRQ(ierr);

  // Create three local-ordered ISs
  ierr = CreateLocalOrderIS(ugrid->num_cells_global, ugrid->num_cells_local, ugrid->num_cells_ghost, ndof, ugdm); CHKERRQ(ierr);

  // Create three local-ordered ISs
  ierr = CreateLocalOrderIS(ugrid->num_cells_global, ugrid->num_cells_local, ugrid->num_cells_ghost, ndof, ugdm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
