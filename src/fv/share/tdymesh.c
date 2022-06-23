#include <petsc.h>
#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>
#include <private/tdyregionimpl.h>
#include <private/tdydiscretizationimpl.h>

/// Allocates memory and initialize a TDyCell struct
/// @param [in] num_cells Number of cells
/// @param [in] cell_type Type of 3D cell (e.g. hexahedron, wedge, etc)
/// @param [out] cells A TDyCell struct that is allocated
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateCells(
  PetscInt    num_cells,
  TDyCellType cell_type,
  TDyCell    *cells) {

  PetscFunctionBegin;

  PetscInt num_vertices  = GetNumVerticesForCellType(cell_type);
  PetscInt num_edges     = GetNumEdgesForCellType(cell_type);
  PetscInt num_neighbors = GetNumNeighborsForCellType(cell_type);
  PetscInt num_faces     = GetNumFacesForCellType(cell_type);

  TDySubcellType subcell_type = GetSubcellTypeForCellType(cell_type);
  PetscInt num_subcells = GetNumSubcellsForSubcellType(subcell_type);
  num_subcells = num_vertices;

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_1D(&cells->id,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->global_id,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->natural_id,num_cells); CHKERRQ(ierr);

  ierr = TDyAlloc(num_cells*sizeof(PetscBool), &cells->is_local); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&cells->num_vertices,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_edges,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_faces,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_neighbors,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_subcells,num_cells); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&cells->vertex_offset  ,num_cells+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->edge_offset    ,num_cells+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->face_offset    ,num_cells+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->neighbor_offset,num_cells+1); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&cells->vertex_ids  ,num_cells*num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->edge_ids    ,num_cells*num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->neighbor_ids,num_cells*num_neighbors); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->face_ids    ,num_cells*num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_TDyCoordinate_1D(num_cells,&cells->centroid); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&cells->volume,num_cells); CHKERRQ(ierr);

  for (PetscInt icell=0; icell<num_cells; icell++) {
    cells->id[icell]            = icell;
    cells->num_vertices[icell]  = num_vertices;
    cells->num_edges[icell]     = num_edges;
    cells->num_faces[icell]     = num_faces;
    cells->num_neighbors[icell] = num_neighbors;
    cells->num_subcells[icell]  = num_subcells;
  }

  for (PetscInt icell=0; icell<=num_cells; icell++) {
    cells->vertex_offset[icell]   = icell*num_vertices;
    cells->edge_offset[icell]     = icell*num_edges;
    cells->face_offset[icell]     = icell*num_faces;
    cells->neighbor_offset[icell] = icell*num_neighbors;
  }

  PetscFunctionReturn(0);
}

/// Allocates memory and initialize a TDySubcell struct
/// @param [in] num_cells Number of cells
/// @param [in] num_subcells_per_cell Maximum number of subcells for a given cell
/// @param [in] subcell_type Type of 3D cell (e.g. hexahedron)
/// @param [out] subcells A TDySubcell struct that is allocated
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateSubcells(
  PetscInt    num_cells,
  PetscInt    num_subcells_per_cell,
  TDySubcellType subcell_type,
  TDySubcell    *subcells) {

  PetscFunctionBegin;

  PetscInt num_nu_vectors = GetNumOfNuVectorsForSubcellType(subcell_type);
  PetscInt num_vertices   = GetNumVerticesForSubcellType(subcell_type);
  PetscInt num_faces      = GetNumFacesForSubcellType(subcell_type);

  PetscErrorCode ierr;
  ierr = TDyAllocate_TDyVector_1D(    num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->nu_vector                      ); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVector_1D(    num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->nu_star_vector                      ); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->variable_continuity_coordinates); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->face_centroid                  ); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_cells*num_subcells_per_cell*num_vertices,   &subcells->vertices_coordinates           ); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&subcells->id,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->num_nu_vectors,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->num_vertices,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->num_faces,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAlloc(num_cells*num_subcells_per_cell*sizeof(TDySubcellType),
                  &subcells->type); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&subcells->nu_vector_offset,num_cells*num_subcells_per_cell+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->vertex_offset  ,num_cells*num_subcells_per_cell+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_offset     ,num_cells*num_subcells_per_cell+1); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_ids        ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->is_face_up      ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_unknown_idx,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_flux_idx   ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(   &subcells->face_area       ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->vertex_ids      ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(   &subcells->T               ,num_cells*num_subcells_per_cell           ); CHKERRQ(ierr);

  for (PetscInt isubcell=0; isubcell<num_cells*num_subcells_per_cell; isubcell++) {
    subcells->id[isubcell]             = isubcell;
    subcells->type[isubcell]           = subcell_type;

    subcells->num_nu_vectors[isubcell] = num_nu_vectors;
    subcells->num_vertices[isubcell]   = num_vertices;
    subcells->num_faces[isubcell]      = num_faces;
  }

  for (PetscInt isubcell=0; isubcell <= num_cells*num_subcells_per_cell; isubcell++) {
    subcells->nu_vector_offset[isubcell] = isubcell*num_nu_vectors;
    subcells->vertex_offset[isubcell] = isubcell*num_vertices;
    subcells->face_offset[isubcell] = isubcell*num_faces;
  }

  PetscFunctionReturn(0);
}

/// Allocates memory and initialize a TDyVertex struct.
/// @param [in] num_vertices Number of vertices
/// @param [in] ncells_per_vertex Maximum number of cells that share a vertex
/// @param [in] nfaces_per_vertex Maximum number of faces that share a vertex
/// @param [in] nedges_per_vertex Maximum number of edges that share a verteex
/// @param [in] cell_type Type of 3D cell (e.g. hexahedron, wedge, etc)
/// @param [out] vertices A TDyVertex struct that is allocated
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateVertices(
  PetscInt       num_vertices,
  PetscInt       ncells_per_vertex,
  PetscInt       nfaces_per_vertex,
  PetscInt       nedges_per_vertex,
  TDyCellType    cell_type,
  TDyVertex     *vertices) {

  PetscFunctionBegin;

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_1D(&vertices->id                ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->global_id         ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_internal_cells,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_edges         ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_faces         ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_boundary_faces,num_vertices); CHKERRQ(ierr);

  ierr = TDyAlloc(num_vertices*sizeof(PetscBool), &vertices->is_local); CHKERRQ(ierr);

  ierr = TDyAllocate_TDyCoordinate_1D(num_vertices, &vertices->coordinate); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&vertices->edge_offset         ,num_vertices+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->face_offset         ,num_vertices+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->internal_cell_offset,num_vertices+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->subcell_offset      ,num_vertices+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->boundary_face_offset,num_vertices+1); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&vertices->edge_ids         ,num_vertices*nedges_per_vertex ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->face_ids         ,num_vertices*nfaces_per_vertex ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->subface_ids      ,num_vertices*nfaces_per_vertex ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->internal_cell_ids,num_vertices*ncells_per_vertex ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->subcell_ids      ,num_vertices*nfaces_per_vertex ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->boundary_face_ids,num_vertices*nfaces_per_vertex ); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<num_vertices; ivertex++) {
    vertices->id[ivertex]                 = ivertex;
    vertices->is_local[ivertex]           = PETSC_FALSE;
    vertices->num_internal_cells[ivertex] = 0;
    vertices->num_edges[ivertex]          = 0;
    vertices->num_faces[ivertex]          = 0;
    vertices->num_boundary_faces[ivertex] = 0;
  }

  for (PetscInt ivertex=0; ivertex<=num_vertices; ivertex++) {
    vertices->edge_offset[ivertex]          = ivertex*nedges_per_vertex;
    vertices->face_offset[ivertex]          = ivertex*nfaces_per_vertex;
    vertices->internal_cell_offset[ivertex] = ivertex*ncells_per_vertex;
    vertices->subcell_offset[ivertex]       = ivertex*nfaces_per_vertex;
    vertices->boundary_face_offset[ivertex] = ivertex*nfaces_per_vertex;

  }

  TDyInitialize_IntegerArray_1D(vertices->face_ids, num_vertices*nfaces_per_vertex, 0);

  PetscFunctionReturn(0);

}

/// Allocates memory and initialize a TDyEdge struct
/// @param [in] num_edges Number of edges
/// @param [in] cell_type Type of 3D cell (e.g. hexahedron, wedge, etc)
/// @param [out] edges A TDyEdge struct that is allocated
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateEdges(
  PetscInt num_edges,
  TDyCellType cell_type,
  TDyEdge *edges) {

  PetscFunctionBegin;

  PetscInt num_cells = GetMaxNumCellsPerEdgeForCellType(cell_type);

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_1D(&edges->id,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->global_id,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->num_cells,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->vertex_ids,num_edges*2); CHKERRQ(ierr);

  ierr = TDyAlloc(num_edges*sizeof(PetscBool), &edges->is_local); CHKERRQ(ierr);
  ierr = TDyAlloc(num_edges*sizeof(PetscBool), &edges->is_internal); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&edges->cell_offset,num_edges+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->cell_ids,num_edges*num_cells); CHKERRQ(ierr);

  ierr = TDyAllocate_TDyCoordinate_1D(num_edges, &edges->centroid); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVector_1D(num_edges, &edges->normal); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&edges->length,num_edges); CHKERRQ(ierr);

  for (PetscInt iedge=0; iedge<num_edges; iedge++) {
    edges->id[iedge] = iedge;
    edges->is_local[iedge] = PETSC_FALSE;
  }
  for (PetscInt iedge=0; iedge<=num_edges; iedge++) {
    edges->cell_offset[iedge] = iedge*num_cells;
  }

  PetscFunctionReturn(0);
}

/// Allocates memory and initialize a TDyFace struct
/// @param [in] num_faces Number of face
/// @param [in] cell_type Type of 3D cell (e.g. hexahedron, wedge, etc)
/// @param [out] faces A TDyFace struct that is allocated
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateFaces(
  PetscInt num_faces,
  TDyCellType cell_type,
  TDyFace *faces) {

  PetscFunctionBegin;

  PetscInt num_cells    = GetMaxNumCellsPerFaceForCellType(cell_type);
  PetscInt num_edges    = GetMaxNumOfFaceEdgesForCellType(cell_type);
  PetscInt num_vertices = GetMaxNumOfFaceVerticesForCellType(cell_type);

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_1D(&faces->id,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_vertices,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_edges,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_cells,num_faces); CHKERRQ(ierr);

  ierr = TDyAlloc(num_faces*sizeof(PetscBool), &faces->is_local); CHKERRQ(ierr);
  ierr = TDyAlloc(num_faces*sizeof(PetscBool), &faces->is_internal); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&faces->vertex_offset,num_faces+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->cell_offset,num_faces+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->edge_offset,num_faces+1); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&faces->cell_ids,num_faces*num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->edge_ids,num_faces*num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->vertex_ids,num_faces*num_vertices); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&faces->bc_type,num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&faces->area,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&faces->projected_area,num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&faces->dist,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&faces->dist_wt_up,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&faces->dist_up_dn,num_faces,2); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&faces->unit_vec_up_dn,num_faces,2); CHKERRQ(ierr);

  ierr = TDyAllocate_TDyCoordinate_1D(num_faces, &faces->centroid); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVector_1D(num_faces, &faces->normal); CHKERRQ(ierr);

  for (PetscInt iface=0; iface<num_faces; iface++) {
    faces->id[iface] = iface;
    faces->is_local[iface]    = PETSC_FALSE;
    faces->is_internal[iface] = PETSC_FALSE;

    faces->num_edges[iface] = num_edges;

    faces->num_cells[iface] = 0;
    faces->num_vertices[iface] = 0;

    faces->bc_type[iface] = NEUMANN_BC;

    faces->cell_offset[iface] = iface*num_cells;
    faces->edge_offset[iface] = iface*num_edges;
    faces->vertex_offset[iface] = iface*num_vertices;
  }

  PetscFunctionReturn(0);
}

/// Get the cell type based on the number of vertices of a cell
/// @param [in] nverts_per_cell Number of vertices of a cell
///
/// @returns a TDyCellType 
TDyCellType GetCellType(PetscInt nverts_per_cell) {

  TDyCellType cell_type;

  PetscFunctionBegin;

  switch (nverts_per_cell) {
    case 4:
      cell_type = CELL_TET_TYPE;
      break;
    case 5:
      cell_type = CELL_PYRAMID_TYPE;
      break;
    case 6:
      cell_type = CELL_WEDGE_TYPE;
      break;
    case 8:
      cell_type = CELL_HEX_TYPE;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported nverts_per_cell for 3D mesh");
      break;
  }

  PetscFunctionReturn(cell_type);
}

/// Get the number of vertices for a given cell type
/// @param [in] cell_type The type of 3D cell
///
/// @returns an integer 
PetscInt GetNumVerticesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 4;
      break;
    case CELL_PYRAMID_TYPE:
      value = 5;
      break;
    case CELL_WEDGE_TYPE:
      value = 6;
      break;
    case CELL_HEX_TYPE:
      value = 8;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get maximum number of edges a given cell type
/// @param [in] cell_type The type of 3D cell
///
/// @returns an integer 
PetscInt GetMaxNumCellsPerEdgeForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 3;
      break;
    case CELL_PYRAMID_TYPE:
      value = 4;
      break;
    case CELL_WEDGE_TYPE:
      value = 4;
      break;
    case CELL_HEX_TYPE:
      value = 4;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get maximum number of faces a given cell type
/// @param [in] cell_type The type of 3D cell
///
/// @returns an integer 
PetscInt GetMaxNumCellsPerFaceForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 2;
      break;
    case CELL_PYRAMID_TYPE:
      value = 2;
      break;
    case CELL_WEDGE_TYPE:
      value = 2;
      break;
    case CELL_HEX_TYPE:
      value = 2;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get maximum number of face vertices of a given cell type
/// @param [in] cell_type The type of 3D cell
///
/// @returns an integer 
PetscInt GetMaxNumOfFaceVerticesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_WEDGE_TYPE:
      value = 4;
      break;
    case CELL_HEX_TYPE:
      value = 4;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get face type for the iface-th of a given cell type
/// @param [in] cell_type The type of 3D cell
/// @param [in] iface The iface-th face of the 3D cell
///
/// @returns face type
TDyFaceType TDyGetFaceTypeForCellType(TDyCellType cell_type, PetscInt iface) {
  PetscFunctionBegin;
  TDyFaceType face_type;

  switch (cell_type) {
    case CELL_TET_TYPE:
      face_type = TRI_FACE_TYPE;
      break;
    case CELL_PYRAMID_TYPE:
      if (iface > 3) {
	face_type = QUAD_FACE_TYPE;
      } else {
	face_type = TRI_FACE_TYPE;
      }
      break;
    case CELL_WEDGE_TYPE:
      if (iface > 2) {
	face_type = TRI_FACE_TYPE;
      } else {
	face_type = QUAD_FACE_TYPE;
      }
      break;
    case CELL_HEX_TYPE:
      face_type = QUAD_FACE_TYPE;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }  
  PetscFunctionReturn(face_type);
}

/// Get the number of vertices for a give face type
/// @param [in] face_type A face type
///
/// @returns num vertices
PetscInt TDyGetNumVerticesForFaceType(TDyFaceType face_type) {

  PetscInt nvertices=0;

  switch (face_type)
  {
  case TRI_FACE_TYPE:
    nvertices = 3;
    break;

  case QUAD_FACE_TYPE:
    nvertices = 4;
    break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported face_type");
    break;
  }

  PetscFunctionReturn(nvertices);
}

/// Get the number of vertices for the iface-th of a give cell type
/// @param [in] cell_type A cell type
/// @param [in] iface A iface-th of the given cell type
///
/// @returns num of vertices
PetscInt GetNumOfVerticesOfIthFacesForCellType(TDyCellType cell_type, PetscInt iface) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 3;
      break;
    case CELL_PYRAMID_TYPE:
      if (iface > 3) {
        value = 4;
      } else {
        value = 3;
      }
      break;
    case CELL_WEDGE_TYPE:
      if (iface > 2) {
        value = 3;
      } else {
        value = 4;
      }
      break;
    case CELL_HEX_TYPE:
      value = 4;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get the maximum number of face edges for a given cell type
/// @param [in] cell_type A cell type
///
/// @returns num of vertices
PetscInt GetMaxNumOfFaceEdgesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 3;
      break;
    case CELL_PYRAMID_TYPE:
      value = 4;
      break;
    case CELL_WEDGE_TYPE:
      value = 4;
      break;
    case CELL_HEX_TYPE:
      value = 4;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get the number of face for a given cell type
/// @param [in] cell_type A cell type
///
/// @returns num of faces
PetscInt GetNumEdgesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
  case CELL_TET_TYPE:
      value = 6;
      break;
    case CELL_PYRAMID_TYPE:
      value = 8;
      break;
    case CELL_WEDGE_TYPE:
      value = 9;
      break;
    case CELL_HEX_TYPE:
      value = 12;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get the number of neigbhors for a given cell type
/// @param [in] cell_type A cell type
///
/// @returns num of neighbors
PetscInt GetNumNeighborsForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 4;
      break;
    case CELL_PYRAMID_TYPE:
      value = 5;
      break;
    case CELL_WEDGE_TYPE:
      value = 5;
      break;
    case CELL_HEX_TYPE:
      value = 6;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get the number of face for a given cell type
/// @param [in] cell_type A cell type
///
/// @returns num of faces
PetscInt GetNumFacesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_TET_TYPE:
      value = 4;
      break;
    case CELL_PYRAMID_TYPE:
      value = 5;
      break;
    case CELL_WEDGE_TYPE:
      value = 5;
      break;
    case CELL_HEX_TYPE:
      value = 6;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get the type of subcells for a given cell type
/// @param [in] cell_type A cell type
///
/// @returns subcell type
TDySubcellType GetSubcellTypeForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  TDySubcellType value;
  switch (cell_type) {
    case CELL_WEDGE_TYPE:
      value = SUBCELL_HEX_TYPE;
      break;
    case CELL_HEX_TYPE:
      value = SUBCELL_HEX_TYPE;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported cell_type");
      break;
  }
  PetscFunctionReturn(value);
}

/// Get the number subcells for a given subcell type
/// @param [in] subcell_type A subcell type
///
/// @returns num of subcells
PetscInt GetNumSubcellsForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_HEX_TYPE:
    value = 8;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"1. Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/// Get the number nu-vectors for a given subcell type
/// @param [in] subcell_type A subcell type
///
/// @returns num of nu-vectors
PetscInt GetNumOfNuVectorsForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_HEX_TYPE:
    value = 3;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"2. Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/// Get the number vertices for a given subcell type
/// @param [in] subcell_type A subcell type
///
/// @returns num of vertices
PetscInt GetNumVerticesForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_HEX_TYPE:
    value = 8;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/// Get the number faces (for nu-vectors) for a given subcell type
/// @param [in] subcell_type A subcell type
///
/// @returns num of faces
PetscInt GetNumFacesForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_HEX_TYPE:
    value = GetNumOfNuVectorsForSubcellType(subcell_type);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/// Get the number of local faces
/// @param [in] mesh A TDyMesh struct
///
/// @returns num of local faces
PetscInt TDyMeshGetNumberOfLocalFaces(TDyMesh *mesh) {

  PetscInt nLocalFaces = 0;
  PetscInt iface;

  PetscFunctionBegin;

  for (iface = 0; iface<mesh->num_faces; iface++) {
    if (mesh->faces.is_local[iface]) nLocalFaces++;
  }

  PetscFunctionReturn(nLocalFaces);
}

/// Get the number of non-local faces
/// @param [in] mesh A TDyMesh struct
///
/// @returns num of non-local faces
PetscInt TDyMeshGetNumberOfNonLocalFaces(TDyMesh *mesh) {

  PetscInt nNonLocalFaces = 0;
  PetscInt iface;

  PetscFunctionBegin;

  for (iface = 0; iface<mesh->num_faces; iface++) {
    if (!mesh->faces.is_local[iface]) nNonLocalFaces++;
  }

  PetscFunctionReturn(nNonLocalFaces);
}

/// Get the number of non-internal faces (i.e. boundary faces)
/// @param [in] mesh A TDyMesh struct
///
/// @returns num of non-internal faces
PetscInt TDyMeshGetNumberOfNonInternalFaces(TDyMesh *mesh) {

  PetscInt nNonInternalFaces = 0;
  PetscInt iface;

  PetscFunctionBegin;

  for (iface = 0; iface<mesh->num_faces; iface++) {
    if (!mesh->faces.is_internal[iface]) nNonInternalFaces++;
  }

  PetscFunctionReturn(nNonInternalFaces);
}

/// Destroy a mesh, freeing any resources it uses.
/// @param [inout] mesh A mesh instance to be destroyed
PetscErrorCode TDyMeshDestroy(TDyMesh *mesh) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = TDyFree(mesh->cells.id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.global_id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.natural_id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.is_local); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.num_vertices); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.num_edges); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.num_faces); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.num_neighbors); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.num_subcells); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.vertex_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.edge_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.face_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.neighbor_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.vertex_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.edge_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.neighbor_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.face_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.centroid); CHKERRQ(ierr);
  ierr = TDyFree(mesh->cells.volume); CHKERRQ(ierr);

  ierr = TDyFree(mesh->subcells.nu_vector); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.nu_star_vector); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.variable_continuity_coordinates); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.face_centroid); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.vertices_coordinates); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.num_nu_vectors); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.num_vertices); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.num_faces); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.type); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.nu_vector_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.vertex_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.face_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.face_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.is_face_up); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.face_unknown_idx); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.face_flux_idx); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.face_area); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.vertex_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->subcells.T); CHKERRQ(ierr);

  ierr = TDyFree(mesh->vertices.id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.global_id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.num_internal_cells); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.num_edges); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.num_faces); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.num_boundary_faces); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.is_local); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.coordinate); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.edge_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.face_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.internal_cell_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.subcell_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.boundary_face_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.edge_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.face_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.subface_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.internal_cell_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.subcell_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->vertices.boundary_face_ids); CHKERRQ(ierr);

  ierr = TDyFree(mesh->edges.id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.global_id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.num_cells); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.vertex_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.is_local); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.is_internal); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.cell_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.cell_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.centroid); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.normal); CHKERRQ(ierr);
  ierr = TDyFree(mesh->edges.length); CHKERRQ(ierr);

  ierr = TDyFree(mesh->faces.id); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.num_vertices); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.num_edges); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.num_cells); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.is_local); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.is_internal); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.vertex_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.cell_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.edge_offset); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.cell_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.edge_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.vertex_ids); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.area); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.centroid); CHKERRQ(ierr);
  ierr = TDyFree(mesh->faces.normal); CHKERRQ(ierr);

  ierr = TDyFree(mesh->closureSize);
  ierr = TDyDeallocate_IntegerArray_2D(mesh->closure,
            mesh->num_cells+mesh->num_faces+mesh->num_edges+mesh->num_vertices);
  CHKERRQ(ierr);

  ierr = TDyFree(mesh); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Returns the (maximum) numbers of cells and faces per vertex in the mesh.
/// @param [in] mesh A mesh instance
/// @param [out] num_cells The maximum number of cells connected to any vertex.
/// @param [out] num_faces The maximum number of faces connected to any vertex.
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetMaxVertexConnectivity(TDyMesh *mesh,
                                               PetscInt *num_cells,
                                               PetscInt *num_faces) {
  PetscFunctionBegin;
  *num_cells = mesh->max_vertex_cells;
  *num_faces = mesh->max_vertex_faces;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, retrieve an array of cell edge indices, and
/// their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] edges Stores a pointer to an array of edges for the
///                    given cell
/// @param [out] num_edges Stores the number of edges for the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellEdges(TDyMesh *mesh,
                                   PetscInt cell,
                                   PetscInt **edges,
                                   PetscInt *num_edges) {
  PetscInt offset = mesh->cells.edge_offset[cell];
  *edges = &mesh->cells.edge_ids[offset];
  *num_edges = mesh->cells.edge_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, retrieve an array of cell vertex indices, and
/// their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] vertices Stores a pointer to an array of vertices for the
///                       given cell
/// @param [out] num_vertices Stores the number of vertices for the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellVertices(TDyMesh *mesh,
                                      PetscInt cell,
                                      PetscInt **vertices,
                                      PetscInt *num_vertices) {
  PetscInt offset = mesh->cells.vertex_offset[cell];
  *vertices = &mesh->cells.vertex_ids[offset];
  *num_vertices = mesh->cells.vertex_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, returns number of cell vertices
/// their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] num_vertices Stores the number of vertices for the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellNumVertices(TDyMesh *mesh,
                                         PetscInt cell,
                                         PetscInt *num_vertices) {
  PetscInt offset = mesh->cells.vertex_offset[cell];
  *num_vertices = mesh->cells.vertex_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, retrieve an array of indices of faces bounding the cell,
/// and their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] faces Stores a pointer to an array of faces bounding the given
///                    cell
/// @param [out] num_faces Stores the number of faces bounding the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellFaces(TDyMesh *mesh,
                                   PetscInt cell,
                                   PetscInt **faces,
                                   PetscInt *num_faces) {
  PetscInt offset = mesh->cells.face_offset[cell];
  *faces = &mesh->cells.face_ids[offset];
  *num_faces = mesh->cells.face_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, returns number of cell faces
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] num_faces Stores the number of faces bounding the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellNumFaces(TDyMesh *mesh,
                                      PetscInt cell,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->cells.face_offset[cell];
  *num_faces = mesh->cells.face_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, retrieve an array of indices of neighboring
/// cells, and their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] neighbors Stores a pointer to an array of cells neighboring the
///                        given cell
/// @param [out] num_neighbors Stores the number of cells neighboring the given
///                            cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellNeighbors(TDyMesh *mesh,
                                       PetscInt cell,
                                       PetscInt **neighbors,
                                       PetscInt *num_neighbors) {
  PetscInt offset = mesh->cells.neighbor_offset[cell];
  *neighbors = &mesh->cells.neighbor_ids[offset];
  *num_neighbors = mesh->cells.neighbor_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, return number of cell neighbors
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] num_neighbors Stores the number of cells neighboring the given
///                            cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellNumNeighbors(TDyMesh *mesh,
                                          PetscInt cell,
                                          PetscInt *num_neighbors) {
  PetscInt offset = mesh->cells.neighbor_offset[cell];
  *num_neighbors = mesh->cells.neighbor_offset[cell+1] - offset;
  PetscFunctionReturn(0);
}

/// Retrieve the centroid for the given cell in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] centroid Stores the centroid of the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellCentroid(TDyMesh *mesh,
                                      PetscInt cell,
                                      TDyCoordinate *centroid) {
  *centroid = mesh->cells.centroid[cell];
  PetscFunctionReturn(0);
}

/// Retrieve the volume for the given cell in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] centroid Stores the volume of the given cell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetCellVolume(TDyMesh *mesh,
                                    PetscInt cell,
                                    PetscReal *volume) {
  *volume = mesh->cells.volume[cell];
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, retrieve an array of associated internal
/// cell indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] int_cells Stores a pointer to an array of internal cells
///                        attached to the given vertex
/// @param [out] num_int_cells Stores the number of internal cells attached to
///                            the given vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexInternalCells(TDyMesh *mesh,
                                             PetscInt vertex,
                                             PetscInt **int_cells,
                                             PetscInt *num_int_cells) {
  PetscInt offset = mesh->vertices.internal_cell_offset[vertex];
  *int_cells = &mesh->vertices.internal_cell_ids[offset];
  *num_int_cells = mesh->vertices.internal_cell_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, return number of associated internal
/// cells
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_int_cells Stores the number of internal cells attached to
///                            the given vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexNumInternalCells(TDyMesh *mesh,
                                                PetscInt vertex,
                                                PetscInt *num_int_cells) {
  PetscInt offset = mesh->vertices.internal_cell_offset[vertex];
  *num_int_cells = mesh->vertices.internal_cell_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, return number of associated subcells
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_subcells Stores the number of subcells attached to the
///                           given vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexNumSubcells(TDyMesh *mesh,
                                           PetscInt vertex,
                                           PetscInt *num_subcells) {
  PetscInt offset = mesh->vertices.subcell_offset[vertex];
  *num_subcells = mesh->vertices.subcell_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, retrieve an array of associated face
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] faces Stores a pointer to an array of faces attached to
///                    the given vertex
/// @param [out] num_faces Stores the number of faces attached to the given
///                        vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexFaces(TDyMesh *mesh,
                                     PetscInt vertex,
                                     PetscInt **faces,
                                     PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.face_offset[vertex];
  *faces = &mesh->vertices.face_ids[offset];
  *num_faces = mesh->vertices.face_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, return number of faces
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_faces Stores the number of faces attached to the given
///                        vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexNumFaces(TDyMesh *mesh,
                                        PetscInt vertex,
                                        PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.face_offset[vertex];
  *num_faces = mesh->vertices.face_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, retrieve an array of associated subface
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] subfaces Stores a pointer to an array of faces attached to
///                    the given vertex
/// @param [out] num_subfaces Stores the number of faces attached to the given
///                        vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexSubfaces(TDyMesh *mesh,
                                     PetscInt vertex,
                                     PetscInt **subfaces,
                                     PetscInt *num_subfaces) {
  PetscInt offset = mesh->vertices.face_offset[vertex];
  *subfaces = &mesh->vertices.subface_ids[offset];
  *num_subfaces = mesh->vertices.face_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, retrieve an array of associated boundary
////face indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] faces Stores a pointer to an array of boundary faces attached
///                    to the given vertex
/// @param [out] num_faces Stores the number of boundary faces attached to the
///                        given vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexBoundaryFaces(TDyMesh *mesh,
                                             PetscInt vertex,
                                             PetscInt **faces,
                                             PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.boundary_face_offset[vertex];
  *faces = &mesh->vertices.boundary_face_ids[offset];
  *num_faces = mesh->vertices.boundary_face_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a vertex index, return number of associated boundary
////faces
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_faces Stores the number of boundary faces attached to the
///                        given vertex
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetVertexNumBoundaryFaces(TDyMesh *mesh,
                                                PetscInt vertex,
                                                PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.boundary_face_offset[vertex];
  *num_faces = mesh->vertices.boundary_face_offset[vertex+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a face index, retrieve an array of associated cell indices
/// and their number.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] cells Stores a pointer to an array of cells attached to the
///                    given face
/// @param [out] num_cells Stores the number of cells attached to the given
///                        face
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceCells(TDyMesh *mesh,
                                   PetscInt face,
                                   PetscInt **cells,
                                   PetscInt *num_cells) {
  PetscInt offset = mesh->faces.cell_offset[face];
  *cells = &mesh->faces.cell_ids[offset];
  *num_cells = mesh->faces.cell_offset[face+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a face index, return number of associate cells
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] num_cells Stores the number of cells attached to the given
///                        face
///
/// @returns 0 on success, or a non-zero error code on failure
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceNumCells(TDyMesh *mesh,
                                      PetscInt face,
                                      PetscInt *num_cells) {
  PetscInt offset = mesh->faces.cell_offset[face];
  *num_cells = mesh->faces.cell_offset[face+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a face index, retrieve an array of associated vertex
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] vertices Stores a pointer to an array of vertices attached to
///                       the given face
/// @param [out] num_vertices Stores the number of subcells attached to the given
///                           face
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceVertices(TDyMesh *mesh,
                                      PetscInt face,
                                      PetscInt **vertices,
                                      PetscInt *num_vertices) {
  PetscInt offset = mesh->faces.vertex_offset[face];
  *vertices = &mesh->faces.vertex_ids[offset];
  *num_vertices = mesh->faces.vertex_offset[face+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a face index, return number of associated vertex
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] num_vertices Stores the number of subcells attached to the given
///                           face
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceNumVertices(TDyMesh *mesh,
                                         PetscInt face,
                                         PetscInt *num_vertices) {
  PetscInt offset = mesh->faces.vertex_offset[face];
  *num_vertices = mesh->faces.vertex_offset[face+1] - offset;
  PetscFunctionReturn(0);
}

/// Retrieve the centroid for the given face in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a face within the mesh
/// @param [out] centroid Stores the centroid of the given face
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceCentroid(TDyMesh *mesh,
                                      PetscInt face,
                                      TDyCoordinate *centroid) {
  *centroid = mesh->faces.centroid[face];
  PetscFunctionReturn(0);
}

/// Retrieve the normal vector for the given face in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a face within the mesh
/// @param [out] normal Stores the normal vector for the given face
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceNormal(TDyMesh *mesh,
                                    PetscInt face,
                                    TDyVector *normal) {
  *normal = mesh->faces.normal[face];
  PetscFunctionReturn(0);
}

/// Retrieve the area of the given face in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a face within the mesh
/// @param [out] area Stores the area of the given face
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetFaceArea(TDyMesh *mesh,
                                  PetscInt face,
                                  PetscReal *area) {
  *area = mesh->faces.area[face];
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated face
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] faces Stores a pointer to an array of faces attached to
///                    the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellFaces(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **faces,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *faces = &mesh->subcells.face_ids[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, return number of associated faces
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellNumFaces(TDyMesh *mesh,
                                         PetscInt subcell,
                                         PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated is_face_up
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of is_face_up array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellIsFaceUp(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **is_face_up,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *is_face_up = &mesh->subcells.is_face_up[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated face_unknown_idx
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face_unkown_idx array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellFaceUnknownIdxs(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **face_unknown_idx,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_unknown_idx = &mesh->subcells.face_unknown_idx[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated face_flux_idx
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face_flux_idx array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellFaceFluxIdxs(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **face_flux_idx,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_flux_idx = &mesh->subcells.face_flux_idx[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated face areas
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face areas array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellFaceAreas(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscReal **face_area,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_area = &mesh->subcells.face_area[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated vertices
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of vertices array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellVertices(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **vertices,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *vertices = &mesh->subcells.vertex_ids[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated nu_vectors
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of nu_vectors array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellNuVectors(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyVector **nu_vectors,
                                      PetscInt *num_nu_vectors) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *nu_vectors = &mesh->subcells.nu_vector[offset];
  *num_nu_vectors = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated nu_star_vectors
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of nu_star_vectors array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellNuStarVectors(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyVector **nu_star_vectors,
                                      PetscInt *num_nu_star_vectors) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *nu_star_vectors = &mesh->subcells.nu_star_vector[offset];
  *num_nu_star_vectors = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated variable_continuity_coordinates
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of variable_continuity_coordinates array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellVariableContinutiyCoordinates(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyCoordinate **variable_continuity_coordinates,
                                      PetscInt *num_variable_continuity_coordinates) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *variable_continuity_coordinates = &mesh->subcells.variable_continuity_coordinates[offset];
  *num_variable_continuity_coordinates = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated face_centroid
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face_centroid array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellFaceCentroids(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyCoordinate **face_centroids,
                                      PetscInt *num_face_centroids) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_centroids = &mesh->subcells.variable_continuity_coordinates[offset];
  *num_face_centroids = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

/// Given a mesh and a subcell index, retrieve an array of associated vertices_coordinates
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of vertices_coordinates array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyMeshGetSubcellVerticesCoordinates(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyCoordinate **vertices_coordinates,
                                      PetscInt *num_vertices_coordinates) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *vertices_coordinates = &mesh->subcells.vertices_coordinates[offset];
  *num_vertices_coordinates = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  PetscFunctionReturn(0);
}

