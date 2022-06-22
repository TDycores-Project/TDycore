#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <private/tdyugdmimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>

/// Allocates memory and initializes a TDyUGDM struct
///
/// @param [out] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMCreate(TDyUGDM **ugdm){
  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = PetscCalloc(sizeof(TDyUGDM), ugdm); CHKERRQ(ierr);

  (*ugdm)->IS_GhostedCells_in_LocalOrder = NULL;
  (*ugdm)->IS_GhostedCells_in_GlobalOrder = NULL;

  (*ugdm)->IS_LocalCells_in_LocalOrder = NULL;
  (*ugdm)->IS_LocalCells_in_GlobalOrder = NULL;

  (*ugdm)->IS_GhostCells_in_LocalOrder = NULL;
  (*ugdm)->IS_GhostCells_in_GlobalOrder = NULL;

  (*ugdm)->IS_LocalCells_to_NaturalCells = NULL;

  (*ugdm)->Scatter_LocalCells_to_GlobalCells = NULL;
  (*ugdm)->Scatter_GlobalCells_to_LocalCells = NULL;
  (*ugdm)->Scatter_LocalCells_to_LocalCells = NULL;
  (*ugdm)->Scatter_GlobalCells_to_NaturalCells = NULL;

  (*ugdm)->Mapping_LocalCells_to_GhostedCells = NULL;

  PetscFunctionReturn(0);
}

/// Delloactes memory of a TDyUGDM struct
///
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMDestroy(TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  if (ugdm->IS_GhostedCells_in_LocalOrder != NULL) {
    ierr = ISDestroy(&ugdm->IS_GhostedCells_in_LocalOrder); CHKERRQ(ierr);
  }

  if (ugdm->IS_GhostedCells_in_GlobalOrder != NULL) {
    ierr = ISDestroy(&ugdm->IS_GhostedCells_in_GlobalOrder); CHKERRQ(ierr);
  }

  if (ugdm->IS_LocalCells_in_LocalOrder != NULL) {
    ierr = ISDestroy(&ugdm->IS_LocalCells_in_LocalOrder); CHKERRQ(ierr);
  }

  if (ugdm->IS_LocalCells_in_GlobalOrder != NULL) {
    ierr = ISDestroy(&ugdm->IS_LocalCells_in_GlobalOrder); CHKERRQ(ierr);
  }

  if (ugdm->IS_GhostCells_in_LocalOrder != NULL) {
    ierr = ISDestroy(&ugdm->IS_GhostCells_in_LocalOrder); CHKERRQ(ierr);
  }

  if (ugdm->IS_GhostCells_in_GlobalOrder != NULL) {
    ierr = ISDestroy(&ugdm->IS_GhostCells_in_GlobalOrder); CHKERRQ(ierr);
  }

  if (ugdm->IS_LocalCells_to_NaturalCells != NULL) {
    ierr = ISDestroy(&ugdm->IS_LocalCells_to_NaturalCells); CHKERRQ(ierr);
  }

  if (ugdm->Scatter_LocalCells_to_GlobalCells != NULL) {
    ierr = VecScatterDestroy(&ugdm->Scatter_LocalCells_to_GlobalCells); CHKERRQ(ierr);
  }

  if (ugdm->Scatter_GlobalCells_to_LocalCells != NULL) {
    ierr = VecScatterDestroy(&ugdm->Scatter_GlobalCells_to_LocalCells); CHKERRQ(ierr);
  }

  if (ugdm->Scatter_LocalCells_to_LocalCells != NULL) {
    ierr = VecScatterDestroy(&ugdm->Scatter_LocalCells_to_LocalCells); CHKERRQ(ierr);
  }

  if (ugdm->Scatter_GlobalCells_to_NaturalCells != NULL) {
    ierr = VecScatterDestroy(&ugdm->Scatter_GlobalCells_to_NaturalCells); CHKERRQ(ierr);
  }

  if (ugdm->Mapping_LocalCells_to_GhostedCells != NULL) {
    ISLocalToGlobalMappingDestroy(&ugdm->Mapping_LocalCells_to_GhostedCells);
  }

  free(ugdm);

  PetscFunctionReturn(0);
}

/// Creates local and global vectors of a TDyUGDM struct
///
/// @param [in] ngmax Number of global (local+ghost) cell
/// @param [in] nlmax Number of local cells
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateVectors(PetscInt ngmax, PetscInt nlmax, PetscInt ndof, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  // 1. Create local vector.
  // Note: The size of the Vec is 'ngmax' (= local cells + ghost cells)
  ierr = VecCreate(PETSC_COMM_SELF, &ugdm->LocalVec); CHKERRQ(ierr);
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

/// Creates IS for a local vector
///
/// @param [in] ngmax Number of global (local+ghost) cell
/// @param [in] nlmax Number of local cells
/// @param [in] nghost Number of ghost cells
/// @param [in] ndof Number of degrees of freedom
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateLocalOrderIS(PetscInt ngmax, PetscInt nlmax, PetscInt nghost, PetscInt ndof, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt idxL[nlmax];
  for (PetscInt i=0; i<nlmax; i++) {
    idxL[i] = i;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nlmax, idxL, PETSC_COPY_VALUES, &ugdm->IS_LocalCells_in_LocalOrder); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_LocalCells_in_LocalOrder,"is_local_local_1.out");
#endif

  PetscInt idxG[ngmax];
  for (PetscInt i=0; i<ngmax; i++) {
    idxG[i] = i;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, ngmax, idxG, PETSC_COPY_VALUES, &ugdm->IS_GhostedCells_in_LocalOrder); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_GhostedCells_in_LocalOrder,"is_ghosted_local_1.out");
#endif

  PetscInt idxGhost[nghost];
  for (PetscInt i=0; i<nghost; i++) {
    idxGhost[i] = nlmax+i;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nghost, idxGhost, PETSC_COPY_VALUES, &ugdm->IS_GhostCells_in_LocalOrder); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_GhostCells_in_LocalOrder,"is_ghosts_local_1.out");
#endif

  PetscFunctionReturn(0);
}

/// Creates IS for a Glboal-ordered vector
///
/// @param [in] ugrid A TDyUGrid struct
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode CreatePetscOrderIS(PetscInt ndof, TDyUGrid *ugrid, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt nlmax=ugrid->num_cells_local;
  PetscInt ngmax=ugrid->num_cells_global;
  PetscInt nghost=ugrid->num_cells_ghost;

  PetscInt idxL[nlmax];
  for (PetscInt i=0; i<nlmax; i++) {
    idxL[i] = i + ugrid->global_offset;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nlmax, idxL, PETSC_COPY_VALUES, &ugdm->IS_LocalCells_in_GlobalOrder); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_LocalCells_in_GlobalOrder,"is_local_petsc_1.out");
#endif

  PetscInt idxG[ngmax];
  for (PetscInt i=0; i<nlmax; i++) {
    idxG[i] = i + ugrid->global_offset;
  }
  for (PetscInt i=0; i<nghost; i++) {
    idxG[i+nlmax] = ugrid->ghost_cell_ids_petsc[i];
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, ngmax, idxG, PETSC_COPY_VALUES, &ugdm->IS_GhostedCells_in_GlobalOrder); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_GhostedCells_in_GlobalOrder,"is_ghosted_petsc_1.out");
#endif

  PetscInt idxGhost[nghost];
  for (PetscInt i=0; i<nghost; i++) {
    idxGhost[i] = ugrid->ghost_cell_ids_petsc[i];
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nghost, idxGhost, PETSC_COPY_VALUES, &ugdm->IS_GhostCells_in_GlobalOrder); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_GhostCells_in_GlobalOrder,"is_ghosts_petsc_1.out");
#endif

  PetscFunctionReturn(0);
}

/// Creates IS for a natural-ordered vector
///
/// @param [in] ugrid A TDyUGrid struct
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateNaturalOrderIS(PetscInt ndof, TDyUGrid *ugrid, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt nlmax = ugrid->num_cells_local;
  PetscInt idx[nlmax];
  for (PetscInt i=0; i<nlmax; i++) {
    idx[i] = i + ugrid->global_offset;
  }

  IS is_tmp;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, nlmax, idx, PETSC_COPY_VALUES, &is_tmp); CHKERRQ(ierr);

  ierr = AOPetscToApplicationIS(ugrid->ao_natural_to_petsc, is_tmp); CHKERRQ(ierr);

  const PetscInt *int_ptr;
  ierr = ISGetIndices(is_tmp, &int_ptr); CHKERRQ(ierr);
  for (PetscInt i=0; i<nlmax; i++) {
    idx[i] = int_ptr[i];
  }
  ierr = ISRestoreIndices(is_tmp, &int_ptr); CHKERRQ(ierr);
  ierr = ISDestroy(&is_tmp); CHKERRQ(ierr);

  ierr = ISCreateBlock(PETSC_COMM_WORLD, ndof, nlmax, idx, PETSC_COPY_VALUES, &ugdm->IS_LocalCells_to_NaturalCells); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISAsASCII(ugdm->IS_LocalCells_to_NaturalCells,"is_local_natural_1.out");
#endif

  PetscFunctionReturn(0);
}

/// Creates following vector scaters
///  - local vector to global vector
///  - global vector to local vector
///  - global vector to application/natural vector
///
/// @param [in] ndof Number of degrees of freedom
/// @param [in] nlmax Number of local grid cells
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateVecScatters(PetscInt ndof, PetscInt nlmax, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  Vec *vec_from, *vec_to;
  IS *is_from, *is_to;
  VecScatter *vec_scatter;

  vec_from    = &ugdm->LocalVec;
  vec_to      = &ugdm->GlobalVec;
  is_from     = &ugdm->IS_LocalCells_in_LocalOrder;
  is_to       = &ugdm->IS_LocalCells_in_GlobalOrder;
  vec_scatter = &ugdm->Scatter_LocalCells_to_GlobalCells;
  ierr = VecScatterCreate(*vec_from, *is_from, *vec_to, *is_to, vec_scatter); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscVecScatterAsASCII(*vec_scatter, "scatter_ltog_1.out");
#endif

  vec_from    = &ugdm->GlobalVec;
  vec_to      = &ugdm->LocalVec;
  is_from     = &ugdm->IS_GhostedCells_in_GlobalOrder;
  is_to       = &ugdm->IS_GhostedCells_in_LocalOrder;
  vec_scatter = &ugdm->Scatter_GlobalCells_to_LocalCells;
  ierr = VecScatterCreate(*vec_from, *is_from, *vec_to, *is_to, vec_scatter); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscVecScatterAsASCII(*vec_scatter, "scatter_gtol_1.out");
#endif

  ierr = VecScatterCopy(ugdm->Scatter_GlobalCells_to_LocalCells, &ugdm->Scatter_LocalCells_to_LocalCells); CHKERRQ(ierr);
  const PetscInt *int_ptr;
  ierr = ISGetIndices(ugdm->IS_LocalCells_in_LocalOrder, &int_ptr); CHKERRQ(ierr);
  PetscInt idx[nlmax];
  for (PetscInt i=0; i<nlmax; i++) {
    idx[i] = int_ptr[i];
  }
  ierr = VecScatterRemap(ugdm->Scatter_LocalCells_to_LocalCells, idx, PETSC_NULL); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ugdm->IS_LocalCells_in_LocalOrder, &int_ptr); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscVecScatterAsASCII(*vec_scatter, "scatter_ltol_1.out");
#endif

  vec_from    = &ugdm->GlobalVec;
  is_from     = &ugdm->IS_LocalCells_in_GlobalOrder;
  is_to       = &ugdm->IS_LocalCells_to_NaturalCells;
  vec_scatter = &ugdm->Scatter_GlobalCells_to_NaturalCells;

  Vec vec_tmp = NULL;
  ierr = VecCreate(PETSC_COMM_WORLD, &vec_tmp); CHKERRQ(ierr);
  ierr = VecSetSizes(vec_tmp, nlmax*ndof, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec_tmp, ndof); CHKERRQ(ierr);
  ierr = VecSetFromOptions(vec_tmp); CHKERRQ(ierr);
  ierr = VecScatterCreate(*vec_from, *is_from, vec_tmp, *is_to, vec_scatter); CHKERRQ(ierr);
  ierr = VecDestroy(&vec_tmp); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscVecScatterAsASCII(*vec_scatter, "scatter_gton_1.out");
#endif

  PetscFunctionReturn(0);
}

/// Creates TDyUGDM struct from the TDyGrid struct
///
/// @param [in] ndof Number of degrees of freedom
/// @param [in] ugrid A TDyUgrid struct
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMCreateFromUGrid(PetscInt ndof, TDyUGrid *ugrid, TDyUGDM *ugdm){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ugdm->ndof = ndof;
  // Create vectors
  ierr = CreateVectors(ugrid->num_cells_global, ugrid->num_cells_local, ndof, ugdm); CHKERRQ(ierr);

  // Create three local-ordered ISs
  ierr = CreateLocalOrderIS(ugrid->num_cells_global, ugrid->num_cells_local, ugrid->num_cells_ghost, ndof, ugdm); CHKERRQ(ierr);

  // Create three local-ordered ISs
  ierr = CreateLocalOrderIS(ugrid->num_cells_global, ugrid->num_cells_local, ugrid->num_cells_ghost, ndof, ugdm); CHKERRQ(ierr);

  // Create three Glboal-ordered ISs
  ierr = CreatePetscOrderIS(ndof, ugrid, ugdm); CHKERRQ(ierr);

  // Create natural-order IS
  ierr = CreateNaturalOrderIS(ndof, ugrid, ugdm); CHKERRQ(ierr);

  // Create four VecScatter
  ierr = CreateVecScatters(ndof, ugrid->num_cells_local, ugdm); CHKERRQ(ierr);

  // Creates the mapping
  ierr = ISLocalToGlobalMappingCreateIS(ugdm->IS_GhostedCells_in_GlobalOrder, &ugdm->Mapping_LocalCells_to_GhostedCells); CHKERRQ(ierr);
#ifdef UGRID_DEBUG
  ierr = TDySavePetscISLocalToGlobalMappingAsASCII(ugdm->Mapping_LocalCells_to_GhostedCells, "mapping_ltog_1.out"); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/// Creates a strided global vector
///
/// @param [in] ndof Number of degrees of freedom
/// @param [in] nlmax Number of local cells
/// @param [in] ugrid A TDyUgrid struct
/// @param [out] global Global vector
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMCreateGlobalVec(PetscInt ndof, PetscInt nlmax, TDyUGDM *ugdm, Vec *global) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, global); CHKERRQ(ierr);
  ierr = VecSetSizes(*global, nlmax*ndof, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*global, ugdm->Mapping_LocalCells_to_GhostedCells); CHKERRQ(ierr);
  ierr = VecSetBlockSize(*global, ndof); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a strided local vector
///
/// @param [in] ndof Number of degrees of freedom
/// @param [in] ngmax Number of global (local+ghost) cells
/// @param [in] ugrid A TDyUgrid struct
/// @param [out] local Local vector
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMCreateLocalVec(PetscInt ndof, PetscInt ngmax, TDyUGDM *ugdm, Vec *local) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, local); CHKERRQ(ierr);
  ierr = VecSetSizes(*local, ngmax*ndof, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(*local, ndof); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*local); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a strided natural vector
///
/// @param [in] ndof Number of degrees of freedom
/// @param [in] ngmax Number of global (local+ghost) cells
/// @param [in] ugrid A TDyUgrid struct
/// @param [out] natural Natural vector
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMCreateNaturalVec(PetscInt ndof, PetscInt ngmax, TDyUGDM *ugdm, Vec *natural) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, natural); CHKERRQ(ierr);
  ierr = VecSetSizes(*natural, ngmax*ndof, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(*natural, ndof); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*natural); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a strided matrix that is used for Jacobian computation
///
/// @param [in] ugrid A TDyUGrid struct
/// @param [in] nlmax Number of local cells
/// @param [in] ugrid A TDyUgrid struct
/// @param [in] ndof Number of degrees of freedom
/// @param [inout] ugdm A TDyUGDM struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyUGDMCreateMatrix(TDyUGrid *ugrid, TDyUGDM *ugdm, PetscInt ndof, Mat *matrix) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt nlmax=ugrid->num_cells_local;
  PetscInt d_nnz[nlmax], o_nnz[nlmax];

  for (PetscInt icell=0; icell<nlmax; icell++) {
    d_nnz[icell] = 1;
    o_nnz[icell] = 0;

    PetscInt nneighbors = ugrid->cell_num_neighbors_local_ghosted[icell];
    for (PetscInt ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      PetscInt id = ugrid->cell_neighbors_local_ghosted[icell][ineighbor];
      if (id >= 0) {
        d_nnz[icell]++;
      } else {
        o_nnz[icell]++;
      }
    }
  }

  PetscInt ndof_local = nlmax*ndof;

  ierr = MatCreate(PETSC_COMM_WORLD,matrix); CHKERRQ(ierr);
  ierr = MatSetType(*matrix,MATBAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(*matrix,ndof_local,ndof_local,PETSC_DETERMINE,PETSC_DETERMINE); CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(*matrix,ndof,d_nnz,o_nnz,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*matrix,ugdm->Mapping_LocalCells_to_GhostedCells,ugdm->Mapping_LocalCells_to_GhostedCells); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}