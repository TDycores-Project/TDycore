#if !defined(TDYUGDMIMPL_H)
#define TDYUGDMIMPL_H

#include <petsc.h>

typedef struct {

  PetscInt num_cells_global;
  PetscInt num_cells_local;
  PetscInt max_verts_per_cells;

  PetscInt num_verts_global;
  PetscInt num_verts_local;

  PetscInt **cell_vertices;
  PetscReal **vertices;

} TDyUGrid;

typedef struct {
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

  ISLocalToGlobalMapping Mapping_LocalCells_to_NaturalCells;

} TDyUGDM;

PETSC_INTERN PetscErrorCode TDyUGDMCreate(TDyUGDM*);
PETSC_INTERN PetscErrorCode TDyUGDMCreateFromPFLOTRANMesh(TDyUGDM*,const char*);

#endif