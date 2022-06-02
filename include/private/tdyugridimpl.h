#if !defined(TDYUGRIDIMPL_H)
#define TDYUGRIDIMPL_H

#include <petsc.h>

typedef struct {

  PetscInt num_cells_global;  // Number of global cells = local/non-ghost and ghost cells
  PetscInt num_cells_local;   // Number of local/non-ghost cells
  PetscInt num_cells_ghost;   // Number of ghost cells

  PetscInt max_verts_per_cell; // Maximum number of vertices for a cells
  PetscInt max_ndual_per_cell; // Maximum number of dual (neighors) of a cell

  PetscInt num_verts_global;  // Total number of vertices in the mesh
  PetscInt num_verts_local;   // Total number of vertices on a rank

  PetscInt global_offset;        // Offset of on each rank based on a PETSc order

  PetscInt **cell_vertices;      // Vertex ids for each cell
  PetscInt *cell_num_vertices;   // Number of vertices for the cell

  PetscInt *cell_ids_natural;    // IDs of local+ghost cells in natural order
  PetscInt *cell_ids_petsc;      // IDs of local+ghost cells in PETSc order
  PetscInt *ghost_cell_ids_petsc;// IDs of ghost cells in PETSc order

  PetscInt **cell_neighbors_local_ghosted;    // IDs of cell neighbors
  PetscInt *cell_num_neighbors_local_ghosted; // Number of neighbors

  PetscReal **vertices; // x,y,z of vertices

  AO ao_natural_to_petsc;

  PetscInt max_cells_sharing_a_vertex; // maximum number of cells that share a vertex
  PetscInt max_vert_per_face;          // maximum number of vertices of a face
  PetscInt max_face_per_cell;          // maximum number of faces of a cell

  PetscInt num_faces;               // (1) internal faces (local+ghosted) (2) boundary faces
  PetscInt *face_num_vertices;      // number of vertices for the face
  PetscInt **face_to_vertex_natural;// size = [max_vert_per_face x max_face_per_cell*ngmax]
  PetscInt **face_to_vertex;        // size = [max_vert_per_face x num_faces]
  PetscReal **face_centroid;        // size = [num_faces x 3]
  PetscInt *connection_to_face;     // face ID corresponding a given connection
  PetscReal *face_area;             // size = [num_faces]

  PetscInt **cell_to_face_ghosted;  // face ID corresponding to the i-th face of a cell

  PetscInt *vertex_ids_natural;     // ID of vertex in natural order


} TDyUGrid;

PETSC_INTERN PetscErrorCode TDyUGridCreate(TDyUGrid**);
PETSC_INTERN PetscErrorCode TDyUGridCreateFromPFLOTRANMesh(TDyUGrid*,const char*);

#endif