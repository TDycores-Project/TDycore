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
typedef struct _TDy_region     TDy_region;
typedef struct _TDy_mesh       TDy_mesh;

typedef enum {
  CELL_QUAD_TYPE=0, /* quadrilateral cell for a 2D cell */
  CELL_WEDGE_TYPE,  /* wedge/prism cell for a 3D cell */
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

  PetscInt       *id;                              /* id of the subcell                                          */
  TDySubcellType *type;                            /* triangle or tetrahedron                                    */

  PetscInt *cell_id;                               /* cell id in local numbering to which the subcell belongs to */

  PetscInt       *num_nu_vectors;                  /* number of nu vectors of the subcell                        */
  PetscInt       *nu_vector_offset;
  TDy_vector     *nu_vector;                       /* nu vectors used to compute transmissibility                */
  TDy_vector     *nu_star_vector;                  /* nu_star vectors used to compute TPF transmissibility       */
  TDy_coordinate *variable_continuity_coordinates; /* coordinates at which variable continuity is enforced       */
  TDy_coordinate *face_centroid;                   /* centroid of faces of subcell */

  PetscInt        *num_vertices;                   /* number of vertices that form the subcell                   */
  PetscInt        *vertex_offset;
  TDy_coordinate  *vertices_coordinates;           /* vertex coordinates that form the subcell                   */

  PetscReal *T;                                    /* Double product for 2D and triple product 3D subcell        */

  PetscInt *num_faces;                             /* number of faces */
  PetscInt *face_offset;
  PetscInt *face_ids;                              /* ids of faces */
  PetscReal *face_area;                            /* area of faces */
  PetscInt *is_face_up;                            /* true if the face->cell_ids[0] is upwind of face->cell_ids[1] in cell traversal order*/
  PetscInt *face_unknown_idx;                      /* index of the unknown associated with the face within the vector interface unknowns common to a vertex*/
  PetscInt *face_flux_idx;                         /* index of the fluxes (internal, up, and down) associated with the face */

};

struct _TDy_cell {

  PetscInt  *id;            /* id of the cell in local numbering */
  PetscInt  *global_id;     /* global id of the cell in local numbering */
  PetscInt  *natural_id;    /* natural id of the cell in local numbering */

  PetscBool *is_local;

  PetscInt  *num_vertices;  /* number of vertices of the cell    */
  PetscInt  *num_edges;     /* number of edges of the cell       */
  PetscInt  *num_faces;     /* number of faces of the cell       */
  PetscInt  *num_neighbors; /* number of neigbors of the cell    */
  PetscInt  *num_subcells;  /* number of subcells within the cell*/

  PetscInt *vertex_offset;    /* vertice IDs that form the cell    */
  PetscInt *edge_offset;      /* vertice IDs that form the cell    */
  PetscInt *face_offset;      /* vertice IDs that form the cell    */
  PetscInt *neighbor_offset;  /* vertice IDs that form the cell    */

  PetscInt *vertex_ids;    /* vertice IDs that form the cell    */
  PetscInt *edge_ids;      /* edge IDs that form the cell       */
  PetscInt *face_ids;      /* face IDs that form the cell       */
  PetscInt *neighbor_ids;  /* neighbor IDs that form the cell   */

  TDy_coordinate *centroid; /* cell centroid                     */

  PetscReal *volume;        /* volume of the cell                */

};


struct _TDy_vertex {

  PetscInt  *id;                 /* id of the vertex in local numbering                  */
  PetscInt  *global_id;          /* global id of the vertex in local numbering */

  PetscBool *is_local;           /* true if the vertex is shared by a local cell         */

  PetscInt  *num_internal_cells; /* number of internal cells sharing the vertex          */
  PetscInt  *num_edges;          /* number of edges sharing the vertex                   */
  PetscInt  *num_faces;          /* number of faces sharing the vartex                   */
  PetscInt  *num_boundary_faces; /* number of boundary faces sharing the vertex          */

  PetscInt *edge_offset;           /* offset for edge IDs that share the vertex                       */
  PetscInt *face_offset;           /* offset for face IDs that share the vertex                       */
  PetscInt *internal_cell_offset;  /* offset for internal cell IDs that share the vertex              */
  PetscInt *subcell_offset;        /* offset for subcell IDs of internal cells that share the vertex  */
  PetscInt *boundary_face_offset;  /* offset for IDs of the faces that are on the boundary            */

  PetscInt *edge_ids;           /* edge IDs that share the vertex                       */
  PetscInt *face_ids;           /* face IDs that share the vertex                       */
  PetscInt *subface_ids;        /* subface IDs that share the vertex                    */
  PetscInt *internal_cell_ids;  /* internal cell IDs that share the vertex              */
  PetscInt *subcell_ids;        /* subcell IDs of internal cells that share the vertex  */
  PetscInt *boundary_face_ids;  /* IDs of the faces that are on the boundary            */

  TDy_coordinate  *coordinate;    /* (x,y,z) location of the vertex                       */
};

struct _TDy_edge {

  PetscInt  *id;            /* id of the edge in local numbering         */
  PetscInt  *global_id;     /* global id of the edge in local numbering */

  PetscBool *is_local;      /* true if the edge : (1) */
                           /* 1. Is shared by locally owned cells, or   */
                           /* 2. Is shared by local cell and non-local  */
                           /*    cell such that global ID of local cell */
                           /*    is smaller than the global ID of       */
                           /*    non-local cell */

  PetscInt *num_cells;      /* number of faces that form the edge        */
  PetscInt *vertex_ids;     /* ids of vertices that form the edge        */

  PetscInt *cell_offset; /* offset for ids of cell that share the edge */
  PetscInt *cell_ids;      /* ids of cells that share the edge          */

  PetscBool *is_internal;   /* false if the edge is on the mesh boundary */

  TDy_vector     *normal;   /* unit normal vector                        */
  TDy_coordinate *centroid; /* edge centroid                             */

  PetscReal *length;        /* length of the edge                        */

};

struct _TDy_face {
  PetscInt *id;             /* id of the face in local numbering */
  PetscInt *global_id;      /* global id of the face in local numbering */

  PetscBool *is_local;      /* true if the face :  */
                           /* 1. Is shared by locally owned cells, or   */
                           /* 2. Is shared by local cell and non-local  */
                           /*    cell such that global ID of local cell */
                           /*    is smaller than the global ID of       */
                           /*    non-local cell */

  PetscBool *is_internal;   /* false if the face is on the mesh boundary */

  PetscInt *num_vertices;   /* number of vertices that form the face */
  PetscInt *num_edges;      /* number of edges that form the face */
  PetscInt *num_cells;      /* number of cells that share the face */

  PetscInt *vertex_offset;    /* offset for id of vertices that form the face */
  PetscInt *edge_offset;      /* offset for id of edges that form the face */
  PetscInt *cell_offset;      /* offset for id of cells that share the face */

  PetscInt *vertex_ids;    /* id of vertices that form the face */
  PetscInt *edge_ids;      /* id of edges that form the face */
  PetscInt *cell_ids;      /* id of cells that share the face */

  TDy_coordinate *centroid; /* centroid of the face */
  TDy_vector *normal;       /* unit normal to the face */
  PetscReal *area;          /* area of the face */
};

struct _TDy_region {
  PetscInt num_cells;
  PetscInt *id;
};

struct _TDy_mesh {

  PetscInt   num_cells;
  PetscInt   num_faces;
  PetscInt   num_edges;
  PetscInt   num_vertices;
  PetscInt num_boundary_faces;
  PetscInt   num_subcells;

  TDy_cell    cells;
  TDy_subcell subcells;
  TDy_vertex  vertices;
  TDy_edge    edges;
  TDy_face    faces;

  TDy_region region_connected;
};

PETSC_EXTERN PetscErrorCode TDyOutputMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyBuildTwoDimMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyBuildMesh(TDy);
PETSC_EXTERN PetscErrorCode TDyAllocateMemoryForMesh(TDy);
PETSC_EXTERN PetscErrorCode TDySubCell_GetIthNuVector(TDy_subcell*,PetscInt,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDySubCell_GetIthNuStarVector(TDy_subcell*,PetscInt,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDySubCell_GetIthFaceCentroid(TDy_subcell*,PetscInt,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDy_subcell*,PetscInt,PetscInt,PetscInt *);
PETSC_EXTERN PetscErrorCode TDyEdge_GetCentroid(TDy_edge*,PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyEdge_GetNormal(TDy_edge*,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyFace_GetNormal(TDy_face*,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyFace_GetCentroid(TDy_face*,PetscInt,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyVertex_GetCoordinate(TDy_vertex*, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyCell_GetCentroid(TDy_cell*,PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDy_cell*,PetscInt,TDy_vertex*,PetscInt,TDy_subcell*,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetNumberOfLocalCells(TDy_mesh*);
PETSC_EXTERN PetscErrorCode TDyMeshGetNumberOfLocalFacess(TDy_mesh*);
PETSC_EXTERN PetscErrorCode TDyMeshGetNumberOfNonLocalFacess(TDy_mesh*);
PETSC_EXTERN PetscErrorCode TDyMeshGetNumberOfNonInternalFacess(TDy_mesh*);
#endif
