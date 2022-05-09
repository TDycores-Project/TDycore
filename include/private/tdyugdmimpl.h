#if !defined(TDYUGDMIMPL_H)
#define TDYUGDMIMPL_H

#include <petsc.h>
#include <private/tdyugridimpl.h>

typedef struct {
  Vec LocalVec;
  Vec GlobalVec;

  IS IS_GhostedCells_in_LocalOrder;
  IS IS_GhostedCells_in_PetscOrder;

  IS IS_LocalCells_in_LocalOrder;
  IS IS_LocalCells_in_PetscOrder;

  IS IS_GhostCells_in_LocalOrder;
  IS IS_GhostCells_in_PetscOrder;

  IS IS_LocalCells_to_NaturalCells;

  VecScatter Scatter_LocalCells_to_GlobalCells;
  VecScatter Scatter_GlobalCells_to_LocalCells;
  VecScatter Scatter_LocalCells_to_LocalCells;
  VecScatter Scatter_GlobalCells_to_NaturalCells;

  ISLocalToGlobalMapping Mapping_LocalCells_to_GhostedCells;

} TDyUGDM;

PETSC_INTERN PetscErrorCode TDyUGDMCreate(TDyUGDM*);
PETSC_INTERN PetscErrorCode TDyUGDMDestroy(TDyUGDM*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateFromUGrid(PetscInt,TDyUGrid*,TDyUGDM*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateGlobalVec(PetscInt,PetscInt,TDyUGDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateLocalVec(PetscInt,PetscInt,TDyUGDM*,Vec*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateNaturalVec(PetscInt,PetscInt,TDyUGDM*,Vec*);

#endif