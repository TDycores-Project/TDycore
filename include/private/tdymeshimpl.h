#if !defined(TDYCOREMESHIMPL_H)
#define TDYCOREMESHIMPL_H

#include <petsc.h>
#include "tdycore.h"

typedef struct _TDy_coordinate TDy_coordinate;
typedef struct _TDy_vector     TDy_vector;
typedef struct _TDy_subcell    TDy_subcell;
typedef struct _TDy_cell       TDy_cell;
typedef struct _TDy_vertex     TDy_vertex;
typedef struct _TDy_edge       TDy_edge;
typedef struct _TDy_face       TDy_face;
typedef struct _TDy_mesh       TDy_mesh;

typedef enum {
  CELL_QUAD_TYPE=0, /* quadrilateral cell for a 2D cell */
  CELL_HEX_TYPE     /* hexahedron cell for a 3D cell */
} TDyCellType;


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

  PetscReal T;                      /* Double product for 2D and triple product 3D subcell        */

  PetscInt num_faces;               /* number of faces */
  PetscInt *face_ids;               /* ids of faces */
  PetscReal *face_area;             /* area of faces */
  TDy_coordinate *face_centroid;    /* centroid of faces of subcell */
  PetscInt *is_face_up;             /* true if the face->cell_ids[0] is upwind of face->cell_ids[1] in cell traversal order*/
  PetscInt *face_unknown_idx;       /* index of the unknown associated with the face within the vector interface unknowns common to a vertex*/

};

struct _TDy_cell {

  PetscInt  id;            /* id of the cell in local numbering */
  PetscInt  global_id;     /* global id of the cell in local numbering */

  PetscBool is_local;

  PetscInt  num_vertices;  /* number of vertices of the cell    */
  PetscInt  num_edges;     /* number of edges of the cell       */
  PetscInt  num_faces;     /* number of faces of the cell       */
  PetscInt  num_neighbors; /* number of neigbors of the cell    */
  PetscInt  num_subcells;  /* number of subcells within the cell*/

  PetscInt *vertex_ids;    /* vertice IDs that form the cell    */
  PetscInt *edge_ids;      /* edge IDs that form the cell       */
  PetscInt *face_ids;      /* face IDs that form the cell       */
  PetscInt *neighbor_ids;  /* neighbor IDs that form the cell   */

  TDy_coordinate centroid; /* cell centroid                     */

  PetscReal volume;        /* volume of the cell                */

  TDy_subcell *subcells;   /* subcells that form the cell       */

};


struct _TDy_vertex {

  PetscInt  id;                 /* id of the vertex in local numbering                  */

  PetscBool is_local;           /* true if the vertex is shared by a local cell         */

  PetscInt  num_internal_cells; /* number of internal cells sharing the vertex          */
  PetscInt  num_edges;          /* number of edges sharing the vertex                   */
  PetscInt  num_faces;          /* number of faces sharing the vartex                   */
  PetscInt  num_boundary_cells; /* number of boundary cells sharing the vertex          */

  PetscInt *edge_ids;           /* edge IDs that share the vertex                       */
  PetscInt *face_ids;           /* face IDs that share the vertex                       */
  PetscInt *internal_cell_ids;  /* internal cell IDs that share the vertex              */
  PetscInt *subcell_ids;        /* subcell IDs of internal cells that share the vertex  */
  PetscInt *boundary_face_ids;  /* IDs of the faces that are on the boundary            */

  PetscInt *trans_row_face_ids;

  TDy_coordinate
  coordinate;    /* (x,y,z) location of the vertex                       */
};

struct _TDy_edge {

  PetscInt  id;            /* id of the edge in local numbering         */

  PetscBool is_local;      /* true if the edge : (1) */
                           /* 1. Is shared by locally owned cells, or   */
                           /* 2. Is shared by local cell and non-local  */
                           /*    cell such that global ID of local cell */
                           /*    is smaller than the global ID of       */
                           /*    non-local cell */

  PetscInt num_cells;      /* number of faces that form the edge        */
  PetscInt vertex_ids[2];  /* ids of vertices that form the edge        */
  PetscInt *cell_ids;      /* ids of cells that share the edge          */

  PetscBool is_internal;   /* false if the edge is on the mesh boundary */

  TDy_vector     normal;   /* unit normal vector                        */
  TDy_coordinate centroid; /* edge centroid                             */

  PetscReal length;        /* length of the edge                        */

};

struct _TDy_face {
  PetscInt id;             /* id of the face in local numbering */
  PetscBool is_local;      /* true if the face :  */
                           /* 1. Is shared by locally owned cells, or   */
                           /* 2. Is shared by local cell and non-local  */
                           /*    cell such that global ID of local cell */
                           /*    is smaller than the global ID of       */
                           /*    non-local cell */

  PetscBool is_internal;   /* false if the face is on the mesh boundary */

  PetscInt num_vertices;   /* number of vertices that form the face */
  PetscInt num_edges;      /* number of edges that form the face */
  PetscInt num_cells;      /* number of cells that share the face */

  PetscInt *vertex_ids;    /* id of vertices that form the face */
  PetscInt *edge_ids;      /* id of edges that form the face */
  PetscInt *cell_ids;      /* id of cells that share the face */

  TDy_coordinate centroid; /* centroid of the face */
  TDy_vector normal;       /* unit normal to the face */
  PetscReal area;          /* area of the face */
};

struct _TDy_mesh {

  PetscInt   num_cells;
  PetscInt   num_faces;
  PetscInt   num_edges;
  PetscInt   num_vertices;

  TDy_cell   *cells;
  TDy_vertex *vertices;
  TDy_edge   *edges;
  TDy_face   *faces;

};

PETSC_EXTERN PetscErrorCode OutputMesh(TDy);
PETSC_EXTERN PetscErrorCode BuildTwoDimMesh(TDy);
PETSC_EXTERN PetscErrorCode BuildMesh(TDy);
PETSC_EXTERN PetscErrorCode AllocateMemoryForMesh(DM,TDy_mesh*);
PETSC_EXTERN PetscErrorCode SubCell_GetIthNuVector(TDy_subcell*,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode SubCell_GetIthFaceCentroid(TDy_subcell*,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode Edge_GetCentroid(TDy_edge*,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode Edge_GetNormal(TDy_edge*,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode Face_GetNormal(TDy_face*,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode Face_GetCentroid(TDy_face*,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode Vertex_GetCoordinate(TDy_vertex*,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode Cell_GetCentroid(TDy_cell*,PetscInt, PetscReal*);
#endif
