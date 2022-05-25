#if !defined(TDYUGRIDIMPL_H)
#define TDYUGRIDIMPL_H

#include <petsc.h>

typedef struct {

  PetscInt num_cells_global;  // Number of global cells = local/non-ghost and ghost cells
  PetscInt num_cells_local;   // Number of local/non-ghost cells
  PetscInt num_cells_ghost;   // Number of ghost cells

  PetscInt max_verts_per_cell;
  PetscInt max_ndual_per_cell;

  PetscInt num_verts_global;
  PetscInt num_verts_local;
  PetscInt num_verts_natural;

  PetscInt global_offset;

  PetscInt **cell_vertices;
  PetscInt *cell_num_vertices; // number of vertices for the cell

  PetscInt *cell_ids_natural;
  PetscInt *cell_ids_petsc;
  PetscInt *ghost_cell_ids_petsc;

  PetscInt **cell_neighbors_ghosted;
  PetscInt *cell_num_neighbors_ghosted;

  PetscReal **vertices;

  AO ao_natural_to_petsc;

  PetscInt max_cells_sharing_a_vertex;
  PetscInt max_vert_per_face;
  PetscInt max_face_per_cell;

  PetscInt num_faces;               // (1) internal faces (local+ghosted) (2) boundary faces
  PetscInt *face_num_vertices;      // number of vertices for the face
  PetscInt **face_to_vertex_natural;// size = [max_vert_per_face x max_face_per_cell*ngmax]
  PetscInt **face_to_vertex;        // size = [max_vert_per_face x num_faces]
  PetscReal **face_centroid;        // size = [num_faces x 3]
  PetscInt *connection_to_face;
  PetscReal *face_area;             // size = [num_faces]

  PetscInt **cell_to_face_ghosted;

  PetscInt *vertex_ids_natural;
  PetscInt **cell_neighbors_local_ghosted;


} TDyUGrid;

PETSC_INTERN PetscErrorCode TDyUGridCreate(TDyUGrid**);
PETSC_INTERN PetscErrorCode TDyUGridCreateFromPFLOTRANMesh(TDyUGrid*,const char*);

#endif