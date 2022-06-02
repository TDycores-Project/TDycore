#if !defined(TDYUGDMIMPL_H)
#define TDYUGDMIMPL_H

#include <petsc.h>
#include <private/tdyugridimpl.h>

typedef struct {
  PetscInt ndof; // number of degrees of freedom

  Vec LocalVec;  // vector size = (num_local_cells + num_ghost_cells) * ndof
  Vec GlobalVec; // vector size = num_local_cells * ndof

  IS IS_GhostedCells_in_LocalOrder; // IS of ghosted cells (local+ghost) from a local Vec
  IS IS_GhostedCells_in_PetscOrder; // IS of ghosted cells (local+ghost) from a global Vec

  IS IS_LocalCells_in_LocalOrder; // IS of local cells from a local Vec
  IS IS_LocalCells_in_PetscOrder; // IS of local cells from a global Vec

  IS IS_GhostCells_in_LocalOrder; // IS of ghost cells from a local Vec
  IS IS_GhostCells_in_PetscOrder; // IS of ghost cells from a global Vec

  IS IS_LocalCells_to_NaturalCells; // IS of local cell from a natural Vec

  VecScatter Scatter_LocalCells_to_GlobalCells;   // for scattering data from a local Vec to a global Vec
  VecScatter Scatter_GlobalCells_to_LocalCells;   // for scattering data from a global Vec to a local Vec
  VecScatter Scatter_LocalCells_to_LocalCells;    // for scattering data from a local Vec to a local Vec
  VecScatter Scatter_GlobalCells_to_NaturalCells; // for scattering data from a global Vec to a natural Vec

  ISLocalToGlobalMapping Mapping_LocalCells_to_GhostedCells;

} TDyUGDM;

PETSC_INTERN PetscErrorCode TDyUGDMCreate(TDyUGDM**);
PETSC_INTERN PetscErrorCode TDyUGDMDestroy(TDyUGDM*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateFromUGrid(PetscInt,TDyUGrid*,TDyUGDM*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateGlobalVec(PetscInt,PetscInt,TDyUGDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateLocalVec(PetscInt,PetscInt,TDyUGDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateNaturalVec(PetscInt,PetscInt,TDyUGDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateMatrix(TDyUGrid*,TDyUGDM*,PetscInt,Mat*);

#endif