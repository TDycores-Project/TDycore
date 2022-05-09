#if !defined(TDYUGRIDIMPL_H)
#define TDYUGRIDIMPL_H

#include <petsc.h>

typedef struct {

  PetscInt num_cells_global;
  PetscInt num_cells_local;
  PetscInt num_cells_ghost;

  PetscInt max_verts_per_cell;
  PetscInt max_ndual_per_cell;

  PetscInt num_verts_global;
  PetscInt num_verts_local;
  PetscInt num_verts_natural;

  PetscInt global_offset;

  PetscInt **cell_vertices;
  PetscInt *cell_num_vertices;
  PetscInt *cell_ids_natural;
  PetscInt *cell_ids_petsc;
  PetscInt *ghost_cell_ids_petsc;
  PetscInt **cell_neighbors_ghosted;
  PetscInt *cell_num_neighbors_ghosted;

  PetscReal **vertices;
  PetscInt *vertex_ids_natural;

  AO ao_natural_to_petsc;

} TDyUGrid;

PETSC_INTERN PetscErrorCode TDyUGridCreateFromPFLOTRANMesh(TDyUGrid*,const char*);

#endif