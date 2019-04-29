#if !defined(TDYCOREMESH_H)
#define TDYCOREMESH_H

#include <petsc.h>

typedef struct _TDy_coordinate TDy_coordinate;
typedef struct _TDy_vector     TDy_vector;
typedef struct _TDy_subcell    TDy_subcell;
typedef struct _TDy_cell       TDy_cell;
typedef struct _TDy_vertex     TDy_vertex;
typedef struct _TDy_edge       TDy_edge;
typedef struct _TDy_mesh       TDy_mesh;

typedef enum {
  SUBCELL_QUAD_TYPE=0, /* quadrilateral subcell for a 2D cell */
  SUBCELL_HEX_TYPE     /* hexahedron subcell for a 3D cell */
} TDySubcellType;

struct _TDy_coordinate {

  PetscReal X[3];

};

struct _TDy_vector {

  PetscReal V[3];

};

struct _TDy_subcell {

  PetscInt
  id;                               /* id of the subcell                                          */
  TDySubcellType
  type;                             /* triangle or tetrahedron                                    */

  PetscInt cell_id;                                /* cell id in local numbering to which the subcell belongs to */

  PetscInt num_vertices;                           /* number of vertices that form the subcell                   */
  PetscInt num_nu_vectors;                         /* number of nu vectors of the subcell                        */

  TDy_vector
  *nu_vector;                       /* nu vectors used to compute transmissibility                */

  TDy_coordinate
  *variable_continuity_coordinates; /* coordinates at which variable continuity is enforced       */
  TDy_coordinate
  *vertices_cordinates;             /* vertex coordinates that form the subcell                   */

  PetscReal volume;                                /* volume of the subcell                                      */

};

struct _TDy_cell {

  PetscInt  id;            /* id of the cell in local numbering */
  PetscInt  global_id;     /* global id of the cell in local numbering */

  PetscBool is_local;

  PetscInt  num_vertices;  /* number of vertices of the cell    */
  PetscInt  num_edges;     /* number of edges of the cell       */
  PetscInt  num_neighbors; /* number of neigbors of the cell    */
  PetscInt  num_subcells;  /* number of subcells within the cell*/

  PetscInt *vertex_ids;    /* vertice IDs that form the cell    */
  PetscInt *edge_ids;      /* edge IDs that form the cell       */
  PetscInt *neighbor_ids;  /* neighbor IDs that form the cell   */

  TDy_coordinate centroid; /* cell centroid                     */

  PetscReal volume;        /* volume of the cell                */

  TDy_subcell *subcells;   /* subcells that form the cell       */

};


struct _TDy_vertex {

  PetscInt  id;                 /* id of the vertex in local numbering                  */

  PetscInt  num_internal_cells; /* number of internal cells sharing the vertex          */
  PetscInt  num_edges;          /* number of edges sharing the vertex                   */
  PetscInt  num_boundary_cells; /* number of boundary cells sharing the vertex          */

  PetscInt *edge_ids;           /* edge IDs that share the vertex                       */
  PetscInt *internal_cell_ids;  /* internal cell IDs that share the vertex              */
  PetscInt *subcell_ids;        /* subcell IDs of internal cells that share the vertex  */

  TDy_coordinate
  coordinate;    /* (x,y,z) location of the vertex                       */
};

struct _TDy_edge {

  PetscInt  id;            /* id of the edge in local numbering         */

  PetscInt num_cells;      /* number of faces that form the edge        */
  PetscInt vertex_ids[2];  /* ids of vertices that form the edge        */
  PetscInt *cell_ids;      /* ids of cells that share the edge          */

  PetscBool is_internal;   /* false if the edge is on the mesh boundary */

  TDy_vector     normal;   /* unit normal vector                        */
  TDy_coordinate centroid; /* edge centroid                             */

  PetscReal length;        /* length of the edge                        */

};

struct _TDy_mesh {

  PetscInt   num_cells;
  PetscInt   num_faces;
  PetscInt   num_edges;
  PetscInt   num_vertices;

  TDy_cell   *cells;
  TDy_vertex *vertices;
  TDy_edge   *edges;

};


#endif
