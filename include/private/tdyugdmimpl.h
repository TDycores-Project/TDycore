#if !defined(TDYUGDMIMPL_H)
#define TDYUGDMIMPL_H

#include <petsc.h>

typedef struct {

  PetscInt num_cells_global;
  PetscInt num_cells_local;
  PetscInt max_verts_per_cell;
  PetscInt max_ndual_per_cell;

  PetscInt num_verts_global;
  PetscInt num_verts_local;

  PetscInt global_offset;

  PetscInt **cell_vertices;
  PetscInt *cell_ids_natural;
  PetscInt *cell_ids_petsc;
  PetscReal **vertices;

  AO ao_natural_to_petsc;

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