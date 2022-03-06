#if !defined(TDYCOREMESHIMPL_H)
#define TDYCOREMESHIMPL_H

#include <petsc.h>
#include <tdycore.h>
#include <private/tdyregionimpl.h>

typedef struct TDyMPFAO TDyMPFAO;

typedef struct {
  PetscReal X[3];
} TDyCoordinate;

typedef struct {
  PetscReal V[3];
} TDyVector;

typedef enum {
  CELL_QUAD_TYPE=0, /* quadrilateral cell for a 2D cell */
  CELL_WEDGE_TYPE,  /* wedge/prism cell for a 3D cell */
  CELL_HEX_TYPE     /* hexahedron cell for a 3D cell */
} TDyCellType;

typedef enum {
  SUBCELL_QUAD_TYPE=0, /* quadrilateral subcell for a 2D cell */
  SUBCELL_HEX_TYPE     /* hexahedron subcell for a 3D cell */
} TDySubcellType;

typedef struct {

  PetscInt       *id;                              /* id of the subcell                                          */
  TDySubcellType *type;                            /* triangle or tetrahedron                                    */

  PetscInt *cell_id;                               /* cell id in local numbering to which the subcell belongs to */

  PetscInt      *num_nu_vectors;                  /* number of nu vectors of the subcell                        */
  PetscInt      *nu_vector_offset;
  TDyVector     *nu_vector;                       /* nu vectors used to compute transmissibility                */
  TDyVector     *nu_star_vector;                  /* nu_star vectors used to compute TPF transmissibility       */
  TDyCoordinate *variable_continuity_coordinates; /* coordinates at which variable continuity is enforced       */
  TDyCoordinate *face_centroid;                   /* centroid of faces of subcell */

  PetscInt        *num_vertices;                   /* number of vertices that form the subcell                   */
  PetscInt        *vertex_offset;
  TDyCoordinate  *vertices_coordinates;           /* vertex coordinates that form the subcell                   */

  PetscReal *T;                                    /* Double product for 2D and triple product 3D subcell        */

  PetscInt *num_faces;                             /* number of faces */
  PetscInt *face_offset;
  PetscInt *face_ids;                              /* ids of faces */
  PetscReal *face_area;                            /* area of faces */
  PetscInt *is_face_up;                            /* true if the face->cell_ids[0] is upwind of face->cell_ids[1] in cell traversal order*/
  PetscInt *face_unknown_idx;                      /* index of the unknown associated with the face within the vector interface unknowns common to a vertex*/
  PetscInt *face_flux_idx;                         /* index of the fluxes (internal, up, and down) associated with the face */
  PetscInt *vertex_ids;                            /* index of vertix that is common to all the cell faces that are part of subcell */

} TDySubcell;

typedef struct {

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

  TDyCoordinate *centroid; /* cell centroid                     */

  PetscReal *volume;        /* volume of the cell                */

} TDyCell;


typedef struct {

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

  TDyCoordinate  *coordinate;    /* (x,y,z) location of the vertex                       */
} TDyVertex;

typedef struct {

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

  TDyVector     *normal;   /* unit normal vector                        */
  TDyCoordinate *centroid; /* edge centroid                             */

  PetscReal *length;        /* length of the edge                        */

} TDyEdge;

typedef struct {
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

  TDyCoordinate *centroid; /* centroid of the face */
  TDyVector *normal;       /* unit normal to the face */
  PetscReal *area;          /* area of the face */
} TDyFace;

typedef struct TDyMesh {

  PetscInt   num_cells;
  PetscInt   num_cells_local;
  PetscInt   num_faces;
  PetscInt   num_edges;
  PetscInt   num_vertices;
  PetscInt   num_boundary_faces;
  PetscInt   num_subcells;

  PetscInt max_vertex_cells, max_vertex_faces;

  TDyCell    cells;
  TDySubcell subcells;
  TDyVertex  vertices;
  TDyEdge    edges;
  TDyFace    faces;

  TDyRegion region_connected;

  PetscInt *closureSize, **closure, maxClosureSize;
} TDyMesh;

PETSC_INTERN PetscErrorCode TDyMeshCreate(DM,PetscReal*,PetscReal*,PetscReal*,TDyMesh**);
PETSC_INTERN PetscErrorCode TDyMeshDestroy(TDyMesh*);

// These don't work, and we'll likely get rid of them.
//PETSC_INTERN PetscErrorCode TDyMeshReadGeometry(TDyMesh*,const char*);
//PETSC_INTERN PetscErrorCode TDyMeshWriteGeometry(TDyMesh*,const char*);

PETSC_INTERN PetscErrorCode TDyMeshGetNumLocalCells(TDyMesh*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetLocalCellNaturalIDs(TDyMesh*,PetscInt*,PetscInt[]);
PETSC_INTERN PetscErrorCode TDyMeshGetMaxVertexConnectivity(TDyMesh*,PetscInt*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellEdges(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNeighbors(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellCentroid(TDyMesh*, PetscInt, TDyCoordinate*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellVolume(TDyMesh*, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumVertices(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumVertices(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumFaces(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumNeighbors(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellIsLocal(TDyMesh*, PetscInt*, PetscInt[]);

PETSC_INTERN PetscErrorCode TDyMeshGetVertexInternalCells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexSubcells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexSubfaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexBoundaryFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumInternalCells(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumSubcells(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumFaces(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumBoundaryFaces(TDyMesh*, PetscInt, PetscInt*);

PETSC_INTERN PetscErrorCode TDyMeshGetFaceCells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceCentroid(TDyMesh*, PetscInt, TDyCoordinate*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceNormal(TDyMesh*, PetscInt, TDyVector*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceArea(TDyMesh*, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceNumCells(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceNumVertices(TDyMesh*, PetscInt, PetscInt*);

PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellIsFaceUp(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceUnknownIdxs(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceFluxIdxs(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceAreas(TDyMesh*, PetscInt, PetscReal**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellNuVectors(TDyMesh*, PetscInt, TDyVector**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellNuStarVectors(TDyMesh*, PetscInt, TDyVector**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellVariableContinutiyCoordinates(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceCentroids(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellVerticesCoordinates(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellNumFaces(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellIDGivenCellIdVertexIdFaceId(TDyMesh*,PetscInt,PetscInt,PetscInt,PetscInt*);

PETSC_INTERN PetscErrorCode TDyMeshComputeGeometry(PetscReal*, PetscReal*, PetscReal*, DM);

PETSC_INTERN TDyCellType GetCellType(PetscInt);
PETSC_INTERN PetscInt GetNumVerticesForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfCellsSharingAVertexForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumCellsPerEdgeForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumCellsPerFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfCellsSharingAFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfVerticesFormingAFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfEdgesFormingAFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumEdgesForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumNeighborsForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumFacesForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumFacesSharedByVertexForCellType(TDyCellType);
PETSC_INTERN TDySubcellType GetSubcellTypeForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumSubcellsForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt GetNumOfNuVectorsForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt GetNumVerticesForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt GetNumFacesForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt TDyMeshGetNumberOfLocalFaces(TDyMesh*);
PETSC_INTERN PetscInt TDyMeshGetNumberOfNonLocalFaces(TDyMesh*);
PETSC_INTERN PetscInt TDyMeshGetNumberOfNonInternalFaces(TDyMesh*);
PETSC_INTERN PetscErrorCode TDyMeshFindFaceIDShareByTwoCells(TDyMesh*,PetscInt,PetscInt,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshPrintSubcellInfo(TDyMesh*, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDyMeshPrintFaceInfo(TDyMesh*, PetscInt);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthNuVector(TDySubcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthFaceCentroid(TDySubcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDySubcell*, PetscInt, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyEdge_GetCentroid(TDyEdge*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyFace_GetCentroid(TDyFace*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyFace_GetNormal(TDyFace*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyVertex_GetCoordinate(TDyVertex*, PetscInt, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode TDyCell_GetCentroid2(TDyCell*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode FindNeighboringVerticesOfAFace(TDyFace*, PetscInt, PetscInt, PetscInt[2]);
PETSC_INTERN PetscErrorCode SetupCell2CellConnectivity(TDyVertex*, PetscInt, TDyCell*, TDyFace*, TDySubcell*, PetscInt**);
PETSC_INTERN PetscErrorCode FindFaceIDsOfACellCommonToAVertex(PetscInt, TDyFace*, TDyVertex*, PetscInt,PetscInt[3],PetscInt*);
PETSC_INTERN PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDyCell*, PetscInt, TDyVertex*, PetscInt, TDySubcell*, PetscInt*);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthNuStarVector(TDySubcell*,PetscInt,PetscInt,PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyEdge_GetNormal(TDyEdge*,PetscInt,PetscInt, PetscReal*);
PETSC_INTERN PetscBool IsClosureWithinBounds(PetscInt,PetscInt,PetscInt);

#endif
