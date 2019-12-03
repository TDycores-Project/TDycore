#include <petsc.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>

/* ---------------------------------------------------------------- */

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start,
                                PetscInt end) {
  return (closure >= start) && (closure < end);
}

/* ---------------------------------------------------------------- */
TDyCellType GetCellType(PetscInt dim, PetscInt nverts_per_cell) {

  TDyCellType cell_type;
  
  PetscFunctionBegin;

  switch (dim) {
    case 2:
      switch (nverts_per_cell) {
        case 4:
          cell_type = CELL_QUAD_TYPE;
          break;
        default:
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported nverts_per_cell for 2D mesh");
          break;
      }
      break;
      
    case 3:
      switch (nverts_per_cell) {
        case 8:
          cell_type = CELL_HEX_TYPE;
          break;
        default:
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported nverts_per_cell for 3D mesh");
          break;
      }
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim");
      break;
  }
  
  PetscFunctionReturn(cell_type);
}

/* ---------------------------------------------------------------- */
PetscInt GetNumVerticesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 4;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumOfCellsSharingAVertexForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 4;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumCellsPerEdgeForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 2;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumCellsPerFaceForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 0;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumOfCellsSharingAFaceForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 0;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumOfVerticesFormingAFaceForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 0;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumOfEdgesFormingAFaceForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 0;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumEdgesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 4;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumNeighborsForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 4;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumFacesForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 0;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumFacesSharedByVertexForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = 0;
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
/* ---------------------------------------------------------------- */
TDySubcellType GetSubcellTypeForCellType(TDyCellType cell_type) {
  PetscFunctionBegin;
  TDySubcellType value;
  switch (cell_type) {
    case CELL_QUAD_TYPE:
      value = SUBCELL_QUAD_TYPE;
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

/* ---------------------------------------------------------------- */
PetscInt GetNumSubcellsForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_QUAD_TYPE:
    value = 4;
    break;
  case SUBCELL_HEX_TYPE:
    value = 8;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"1. Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/* ---------------------------------------------------------------- */
PetscInt GetNumOfNuVectorsForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_QUAD_TYPE:
    value = 2;
    break;
  case SUBCELL_HEX_TYPE:
    value = 3;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"2. Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/* ---------------------------------------------------------------- */
PetscInt GetNumVerticesForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_QUAD_TYPE:
    value = 4;
    break;
  case SUBCELL_HEX_TYPE:
    value = 8;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/* ---------------------------------------------------------------- */
PetscInt GetNumFacesForSubcellType(TDySubcellType subcell_type) {
  PetscFunctionBegin;
  PetscInt value;
  switch (subcell_type) {
  case SUBCELL_QUAD_TYPE:
    value = 0;
    break;
  case SUBCELL_HEX_TYPE:
    value = GetNumOfNuVectorsForSubcellType(subcell_type);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported subcell type");
    break;
  }
  PetscFunctionReturn(value);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForCells(
  PetscInt    num_cells,
  TDyCellType cell_type,
  TDy_cell    *cells) {

  PetscFunctionBegin;

  PetscInt       icell;
  PetscInt       num_vertices;
  PetscInt       num_edges;
  PetscInt       num_neighbors;
  PetscInt       num_faces;
  PetscInt       num_subcells;
  TDySubcellType subcell_type;
  PetscErrorCode ierr;

  num_vertices  = GetNumVerticesForCellType(cell_type);
  num_edges     = GetNumEdgesForCellType(cell_type);
  num_neighbors = GetNumNeighborsForCellType(cell_type);
  num_faces     = GetNumFacesForCellType(cell_type);

  subcell_type  = GetSubcellTypeForCellType(cell_type);
  num_subcells  = GetNumSubcellsForSubcellType(subcell_type);

  ierr = TDyAllocate_IntegerArray_1D(&cells->id,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->global_id,num_cells); CHKERRQ(ierr);

   cells->is_local = (PetscBool *)malloc(num_cells*sizeof(PetscBool));

  ierr = TDyAllocate_IntegerArray_1D(&cells->num_vertices,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_edges,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_faces,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_neighbors,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->num_subcells,num_cells); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&cells->vertex_offset  ,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->edge_offset    ,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->face_offset    ,num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->neighbor_offset,num_cells); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&cells->vertex_ids  ,num_cells*num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->edge_ids    ,num_cells*num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->neighbor_ids,num_cells*num_neighbors); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cells->face_ids    ,num_cells*num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_TDyCoordinate_1D(num_cells,&cells->centroid); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&cells->volume,num_cells); CHKERRQ(ierr);

  for (icell=0; icell<num_cells; icell++) {
    cells->id[icell]            = icell;
    cells->num_vertices[icell]  = num_vertices;
    cells->num_edges[icell]     = num_edges;
    cells->num_faces[icell]     = num_faces;
    cells->num_neighbors[icell] = num_neighbors;
    cells->num_subcells[icell]  = num_subcells;

    cells->vertex_offset[icell]   = icell*num_vertices;
    cells->edge_offset[icell]     = icell*num_edges;
    cells->face_offset[icell]     = icell*num_faces;
    cells->neighbor_offset[icell] = icell*num_neighbors;

  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForSubcells(
  PetscInt    num_cells,
  PetscInt    num_subcells_per_cell,
  TDySubcellType subcell_type,
  TDy_subcell    *subcells) {

  PetscFunctionBegin;

  PetscInt isubcell;
  PetscErrorCode ierr;

  PetscInt num_nu_vectors = GetNumOfNuVectorsForSubcellType(subcell_type);
  PetscInt num_vertices   = GetNumVerticesForSubcellType(subcell_type);
  PetscInt num_faces      = GetNumFacesForSubcellType(subcell_type);

  ierr = TDyAllocate_TDyVector_1D(    num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->nu_vector                      ); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->variable_continuity_coordinates); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_cells*num_subcells_per_cell*num_nu_vectors, &subcells->face_centroid                  ); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_cells*num_subcells_per_cell*num_vertices,   &subcells->vertices_coordinates           ); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&subcells->id,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->num_nu_vectors,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->num_vertices,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->num_faces,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  subcells->type = (TDySubcellType *)malloc(num_cells*num_subcells_per_cell*sizeof(TDySubcellType));

  ierr = TDyAllocate_IntegerArray_1D(&subcells->nu_vector_offset,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->vertex_offset  ,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_offset     ,num_cells*num_subcells_per_cell          ); CHKERRQ(ierr);
  
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_ids        ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->is_face_up      ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_unknown_idx,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcells->face_flux_idx   ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(   &subcells->face_area       ,num_cells*num_subcells_per_cell*num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(   &subcells->T               ,num_cells*num_subcells_per_cell           ); CHKERRQ(ierr);

  for (isubcell=0; isubcell<num_cells*num_subcells_per_cell; isubcell++) {
    subcells->id[isubcell]             = isubcell;
    subcells->type[isubcell]           = subcell_type;

    subcells->num_nu_vectors[isubcell] = num_nu_vectors;
    subcells->num_vertices[isubcell]   = num_vertices;
    subcells->num_faces[isubcell]      = num_faces;

    subcells->nu_vector_offset[isubcell] = isubcell*num_nu_vectors;
    subcells->vertex_offset[isubcell]   = isubcell*num_vertices;
    subcells->face_offset[isubcell]      = isubcell*num_faces;

  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForVertices(
  PetscInt       num_vertices,
  TDyCellType    cell_type,
  TDy_vertex     *vertices) {

  PetscFunctionBegin;

  PetscInt ivertex;
  PetscErrorCode ierr;

  PetscInt num_internal_cells = GetNumOfCellsSharingAVertexForCellType(cell_type);
  PetscInt num_edges          = GetNumEdgesForCellType(cell_type);
  PetscInt num_faces          = GetNumFacesSharedByVertexForCellType(cell_type);

  ierr = TDyAllocate_IntegerArray_1D(&vertices->id                ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->global_id         ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_internal_cells,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_edges         ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_faces         ,num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->num_boundary_cells,num_vertices); CHKERRQ(ierr);

  vertices->is_local = (PetscBool *)malloc(num_vertices*sizeof(PetscBool));

  ierr = TDyAllocate_TDyCoordinate_1D(num_vertices, &vertices->coordinate); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&vertices->edge_offset         ,num_vertices ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->face_offset         ,num_vertices ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->internal_cell_offset,num_vertices ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->subcell_offset      ,num_vertices ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->boundary_face_offset,num_vertices ); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&vertices->edge_ids         ,num_vertices*num_edges ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->face_ids         ,num_vertices*num_faces ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->internal_cell_ids,num_vertices*num_internal_cells ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->subcell_ids      ,num_vertices*num_internal_cells ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertices->boundary_face_ids,num_vertices*num_internal_cells ); CHKERRQ(ierr);

  for (ivertex=0; ivertex<num_vertices; ivertex++) {
    vertices->id[ivertex]                 = ivertex;
    vertices->is_local[ivertex]           = PETSC_FALSE;
    vertices->num_internal_cells[ivertex] = 0;
    vertices->num_edges[ivertex]          = num_edges;
    vertices->num_faces[ivertex]          = 0;
    vertices->num_boundary_cells[ivertex] = 0;

    vertices->edge_offset[ivertex]          = ivertex*num_edges;
    vertices->face_offset[ivertex]          = ivertex*num_faces;
    vertices->internal_cell_offset[ivertex] = ivertex*num_internal_cells;
    vertices->subcell_offset[ivertex]       = ivertex*num_internal_cells;
    vertices->boundary_face_offset[ivertex] = ivertex*num_internal_cells;

  }

  TDyInitialize_IntegerArray_1D(vertices->face_ids, num_vertices*num_faces, 0);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForEdges(
  PetscInt num_edges,
  TDyCellType cell_type,
  TDy_edge *edges) {

  PetscFunctionBegin;

  PetscInt iedge;
  PetscErrorCode ierr;
  
  PetscInt num_cells = GetNumCellsPerEdgeForCellType(cell_type);

  ierr = TDyAllocate_IntegerArray_1D(&edges->id,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->global_id,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->num_cells,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->vertex_ids,num_edges*2); CHKERRQ(ierr);

  edges->is_local = (PetscBool *)malloc(num_edges*sizeof(PetscBool));
  edges->is_internal = (PetscBool *)malloc(num_edges*sizeof(PetscBool));

  ierr = TDyAllocate_IntegerArray_1D(&edges->cell_offset,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->cell_ids,num_edges*num_cells); CHKERRQ(ierr);

  ierr = TDyAllocate_TDyCoordinate_1D(num_edges, &edges->centroid); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVector_1D(num_edges, &edges->normal); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&edges->length,num_edges); CHKERRQ(ierr);

  for (iedge=0; iedge<num_edges; iedge++) {
    edges->id[iedge] = iedge;
    edges->is_local[iedge] = PETSC_FALSE;
    edges->cell_offset[iedge] = iedge*num_cells;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForFaces(
  PetscInt num_faces,
  TDyCellType cell_type,
  TDy_face *faces) {

  PetscFunctionBegin;

  PetscInt iface;
  PetscErrorCode ierr;

  PetscInt num_cells    = GetNumCellsPerFaceForCellType(cell_type);
  PetscInt num_edges    = GetNumOfEdgesFormingAFaceForCellType(cell_type);
  PetscInt num_vertices = GetNumOfVerticesFormingAFaceForCellType(cell_type);

  ierr = TDyAllocate_IntegerArray_1D(&faces->id,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_vertices,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_edges,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_cells,num_faces); CHKERRQ(ierr);

  faces->is_local = (PetscBool *)malloc(num_faces*sizeof(PetscBool));
  faces->is_internal = (PetscBool *)malloc(num_faces*sizeof(PetscBool));

  ierr = TDyAllocate_IntegerArray_1D(&faces->vertex_offset,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->cell_offset,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->edge_offset,num_faces); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&faces->cell_ids,num_faces*num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->edge_ids,num_faces*num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->vertex_ids,num_faces*num_vertices); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&faces->area,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_faces, &faces->centroid); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVector_1D(num_faces, &faces->normal); CHKERRQ(ierr);

  for (iface=0; iface<num_faces; iface++) {
    faces->id[iface] = iface;
    faces->is_local[iface]    = PETSC_FALSE;
    faces->is_internal[iface] = PETSC_FALSE;

    faces->num_edges[iface] = num_edges;

    faces->num_cells[iface] = 0;
    faces->num_vertices[iface] = 0;

    faces->cell_offset[iface] = iface*num_cells;
    faces->edge_offset[iface] = iface*num_edges;
    faces->vertex_offset[iface] = iface*num_vertices;
  }

  PetscFunctionReturn(0);

}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyAllocateMemoryForMesh(TDy tdy) {

  PetscFunctionBegin;

  DM dm;
  TDy_mesh *mesh;
  PetscInt nverts_per_cell;
  PetscInt cStart, cEnd, cNum;
  PetscInt vStart, vEnd, vNum;
  PetscInt eStart, eEnd, eNum;
  PetscInt fNum, num_subcells;
  PetscInt dim;
  TDyCellType cell_type;
  TDySubcellType subcell_type;

  PetscErrorCode ierr;

  dm = tdy->dm;
  mesh = tdy->mesh;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (dim!= 2 && dim!=3 ) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only 2D and 3D grids are supported");
  }

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  cNum = cEnd - cStart;
  eNum = eEnd - eStart;
  vNum = vEnd - vStart;
  
  if (dim == 3) {
    PetscInt fStart, fEnd;
    ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);
    fNum = fEnd - fStart;
  } else {
    fNum = 0;
  }

  mesh->num_cells    = cNum;
  mesh->num_faces    = fNum;
  mesh->num_edges    = eNum;
  mesh->num_vertices = vNum;

  tdy->maxClosureSize = 27;
  ierr = TDyAllocate_IntegerArray_1D(&tdy->closureSize, cNum+fNum+eNum+fNum); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&tdy->closure, cNum+fNum+eNum+fNum, 2*tdy->maxClosureSize); CHKERRQ(ierr);
  ierr = TDySaveClosures(dm, tdy->closureSize, tdy->closure, tdy->maxClosureSize); CHKERRQ(ierr);

  /* compute number of vertices per grid cell */
  nverts_per_cell = TDyGetNumberOfCellVerticesWithClosures(dm, tdy->closureSize, tdy->closure);
  tdy->ncv = nverts_per_cell;
  cell_type = GetCellType(dim, nverts_per_cell);

  ierr = AllocateMemoryForCells(cNum, cell_type, &mesh->cells); CHKERRQ(ierr);
  ierr = AllocateMemoryForVertices(vNum, cell_type, &mesh->vertices); CHKERRQ(ierr);
  ierr = AllocateMemoryForEdges(eNum, cell_type, &mesh->edges); CHKERRQ(ierr);
  ierr = AllocateMemoryForFaces(fNum, cell_type, &mesh->faces); CHKERRQ(ierr);

  subcell_type  = GetSubcellTypeForCellType(cell_type);
  num_subcells  = GetNumSubcellsForSubcellType(subcell_type);
  ierr = AllocateMemoryForSubcells(cNum, num_subcells, subcell_type, &mesh->subcells); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AreFacesNeighbors(TDy_face *faces, PetscInt face_id_1, PetscInt face_id_2) {

  PetscBool are_neighbors;
  PetscInt iedge_1, iedge_2;

  are_neighbors = PETSC_FALSE;

  PetscInt fOffetEdge_1 = faces->edge_offset[face_id_1];
  PetscInt fOffetEdge_2 = faces->edge_offset[face_id_2];

  for (iedge_1=0; iedge_1<faces->num_edges[face_id_1]; iedge_1++) {
    for (iedge_2=0; iedge_2<faces->num_edges[face_id_2]; iedge_2++) {
      if (faces->edge_ids[fOffetEdge_1 + iedge_1] == faces->edge_ids[fOffetEdge_2 + iedge_2]) {
        are_neighbors = PETSC_TRUE;
        break;
      }
    }
  }

  PetscFunctionReturn(are_neighbors);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SaveMeshGeometricAttributes(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  TDy_face       *faces;
  PetscInt       dim;
  PetscInt       cStart, cEnd;
  PetscInt       vStart, vEnd;
  PetscInt       eStart, eEnd;
  PetscInt       fStart, fEnd;
  PetscInt       pStart, pEnd;
  PetscInt       icell, iedge, ivertex, ielement, iface, d;
  PetscErrorCode ierr;

  dm       = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetChart(        dm, &pStart, &pEnd); CHKERRQ(ierr);

  if (dim == 3) {
    ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);
  } else {
    fStart = 0; fEnd = 0;
  }

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  edges    = &mesh->edges;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;

  for (ielement=pStart; ielement<pEnd; ielement++) {

    if (IsClosureWithinBounds(ielement, vStart, vEnd)) { // is the element a vertex?
      ivertex = ielement - vStart;
      for (d=0; d<dim; d++) {
        vertices->coordinate[ivertex].X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, eStart,
                                     eEnd)) { // is the element an edge?
      iedge = ielement - eStart;
      for (d=0; d<dim; d++) {
        edges->centroid[iedge].X[d] = tdy->X[ielement*dim + d];
        edges->normal[iedge].V[d]   = tdy->N[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, cStart,
                                     cEnd)) { // is the elment a cell?
      icell = ielement - cStart;
      for (d=0; d<dim; d++) {
        cells->centroid[icell].X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, fStart,
                                     fEnd)) { // is the elment a face?
      iface = ielement - fStart;
      for (d=0; d<dim; d++) {
        faces->centroid[iface].X[d] = tdy->X[ielement*dim + d];
      }
      faces->area[iface] = tdy->V[ielement];
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode SaveMeshConnectivityInfo(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  TDy_face       *faces;
  PetscInt       dim;
  PetscInt       cStart, cEnd;
  PetscInt       vStart, vEnd;
  PetscInt       eStart, eEnd;
  PetscInt       fStart, fEnd;
  PetscInt       pStart, pEnd;
  PetscInt       icell;
  PetscInt       supportSize, coneSize;
  const PetscInt *support, *cone;
  PetscInt       c2vCount, c2eCount, c2fCount;
  PetscInt       nverts_per_cell;
  PetscInt       i,j,e,v,s;
  PetscReal      v_1[3], v_2[3];
  PetscInt       d;
  PetscErrorCode ierr;

  dm = tdy->dm;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  nverts_per_cell = tdy->ncv;

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetChart(        dm, &pStart, &pEnd); CHKERRQ(ierr);

  if (dim == 3) {
    ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);
  } else {
    fStart = 0; fEnd = 0;
  }

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  edges    = &mesh->edges;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;

  // cell--to--vertex
  // edge--to--cell
  // cell--to--edge
  // edge--to--cell
  for (icell=cStart; icell<cEnd; icell++) {

    c2vCount = 0;
    c2eCount = 0;
    c2fCount = 0;

    for (i=0; i<tdy->closureSize[icell]*2; i+=2)  {

      if (IsClosureWithinBounds(tdy->closure[icell][i], vStart,
                                vEnd)) { /* Is the closure a vertex? */
        PetscInt ivertex = tdy->closure[icell][i] - vStart;
        PetscInt vOffset = cells->vertex_offset[icell];
        cells->vertex_ids[vOffset + c2vCount] = ivertex ;

        PetscInt offsetCell = vertices->internal_cell_offset[ivertex];
        PetscInt offsetSubcell = vertices->subcell_offset[ivertex];

        for (j=0; j<nverts_per_cell; j++) {
          if (vertices->internal_cell_ids[offsetCell + j] == -1) {
            vertices->num_internal_cells[ivertex]++;
            vertices->internal_cell_ids[offsetCell + j] = icell;
            vertices->subcell_ids[offsetSubcell + j]    = c2vCount;
            break;
          }
        }
        c2vCount++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], eStart,
                                       eEnd)) { /* Is the closure an edge? */
        PetscInt iedge = tdy->closure[icell][i] - eStart;
        PetscInt eOffset = cells->edge_offset[icell];
        cells->edge_ids[eOffset + c2eCount] = iedge;
        PetscInt eOffsetCell = edges->cell_offset[iedge];
        for (j=0; j<2; j++) {
          if (edges->cell_ids[eOffsetCell + j] == -1) {
            edges->cell_ids[eOffsetCell + j] = icell;
            break;
          }
        }
        
        c2eCount++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], fStart,
                                       fEnd)) { /* Is the closure a face? */
        PetscInt iface = tdy->closure[icell][i] - fStart;
        PetscInt fOffset = cells->face_offset[icell];
        PetscInt fOffsetCell = faces->cell_offset[iface];
        cells->face_ids[fOffset + c2fCount] = iface;
        for (j=0; j<2; j++) {
          if (faces->cell_ids[fOffsetCell + j] < 0) {
            faces->cell_ids[fOffsetCell + j] = icell;
            faces->num_cells[iface]++;
            break;
          }
        }
        c2fCount++;
      }
    }

  }


  // edge--to--vertex
  for (e=eStart; e<eEnd; e++) {
    ierr = DMPlexGetConeSize(dm, e, &coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, e, &cone); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, e, &supportSize); CHKERRQ(ierr);
    PetscInt iedge = e-eStart;

    if (supportSize == 1) edges->is_internal[iedge] = PETSC_FALSE;
    else                  edges->is_internal[iedge] = PETSC_TRUE;

    edges->vertex_ids[iedge*2 + 0] = cone[0]-vStart;
    edges->vertex_ids[iedge*2 + 1] = cone[1]-vStart;

    ierr = TDyVertex_GetCoordinate(vertices, edges->vertex_ids[iedge*2 + 0], dim, &v_1[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, edges->vertex_ids[iedge*2 + 1], dim, &v_2[0]); CHKERRQ(ierr);

    for (d=0; d<dim; d++) {
      edges->centroid[iedge].X[d] = (v_1[d] + v_2[d])/2.0;
    }

    ierr = TDyComputeLength(v_1, v_2, dim, &(edges->length[iedge])); CHKERRQ(ierr);
  }

  // vertex--to--edge
  for (v=vStart; v<vEnd; v++) {
    ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, v, &supportSize); CHKERRQ(ierr);
    PetscInt ivertex = v - vStart;
    vertices->num_edges[ivertex] = supportSize;
    for (s=0; s<supportSize; s++) {
      PetscInt iedge = support[s] - eStart;
      PetscInt offsetEdge = vertices->edge_offset[ivertex];
      vertices->edge_ids[offsetEdge + s] = iedge;
      if (!edges->is_internal[iedge]) vertices->num_boundary_cells[ivertex]++;
    }
  }

  PetscInt f;
  for (f=fStart; f<fEnd; f++){
    PetscInt iface = f-fStart;
    PetscInt fOffsetEdge = faces->edge_offset[iface];
    PetscInt fOffsetVertex = faces->vertex_offset[iface];
    //face = &faces[iface];

    // face--to--edge
    ierr = DMPlexGetConeSize(dm, f, &coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, f, &cone); CHKERRQ(ierr);

    PetscInt c;
    for (c=0;c<coneSize;c++) {
      faces->edge_ids[fOffsetEdge + c] = cone[c]-eStart;
    }

    // face--to-vertex
    PetscInt i;
    for (i=0; i<tdy->closureSize[f]*2; i+=2)  {
      if (IsClosureWithinBounds(tdy->closure[f][i],vStart,vEnd)) {
        faces->vertex_ids[fOffsetVertex + faces->num_vertices[iface]] = tdy->closure[f][i]-vStart;
        faces->num_vertices[iface]++;

        PetscBool found = PETSC_FALSE;
        PetscInt ivertex = tdy->closure[f][i]-vStart;
        PetscInt offsetFace = vertices->face_offset[ivertex];
        //vertex = &vertices[ivertex];
        PetscInt ii;
        for (ii=0; ii<vertices->num_faces[ivertex]; ii++) {
          if (vertices->face_ids[offsetFace+ii] == iface) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          vertices->face_ids[offsetFace+vertices->num_faces[ivertex]] = iface;
          vertices->num_faces[ivertex]++;
        }
      }
    }

    // face--to--cell
    ierr = DMPlexGetSupportSize(dm, f, &supportSize); CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, f, &support); CHKERRQ(ierr);

    if (supportSize == 2) faces->is_internal[iface] = PETSC_TRUE;
    else                  faces->is_internal[iface] = PETSC_FALSE;

    for (s=0; s<supportSize; s++) {
      icell = support[s] - cStart;
      //cell = &cells[icell];
        PetscBool found = PETSC_FALSE;
        PetscInt ii;
        PetscInt faceStart = cells->face_offset[icell];
        for (ii=0; ii<cells->num_faces[icell]; ii++) {
          if (cells->face_ids[faceStart+ii] == f-fStart) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          cells->face_ids[faceStart + cells->num_faces[icell]] = f-fStart;
          cells->num_faces[icell]++;
          found = PETSC_TRUE;
        }
    }

    // If it is a boundary face, increment the number of boundary
    // cells by 1 for all vertices that form the face
    if (!faces->is_internal[iface]) {
      for (v=0; v<faces->num_vertices[iface]; v++) {
        vertices->num_boundary_cells[faces->vertex_ids[fOffsetVertex + v]]++;
      }
    }

  }

  // allocate memory to save ids of faces on the boundary
  /*
  if (dim == 3) {
  for (v=vStart; v<vEnd; v++) {
    vertex = &vertices[v-vStart];
    ierr = TDyAllocate_IntegerArray_1D(&vertex->boundary_face_ids,vertex->num_boundary_cells); CHKERRQ(ierr);
  }
  }
  */

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode ComputeTheta(PetscReal x, PetscReal y, PetscReal *theta) {

  PetscFunctionBegin;

  if (x>0.0) {
    if (y>= 0.0) *theta = atan(y/x);
    else         *theta = atan(y/x) + 2.0*PETSC_PI;
  } else if (x==0.0) {
    if      (y>  0.0) *theta = 0.5*PETSC_PI;
    else if (y ==0.0) *theta = 0.;
    else              *theta = 1.5*PETSC_PI;
  } else {
    *theta = atan(y/x) + PETSC_PI;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAVertex(TDy tdy, PetscInt ivertex) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  PetscInt       icell, iedge;
  PetscInt       ncells, nedges;
  PetscReal      x,y;
  PetscReal      theta[8];
  PetscInt       idx[8], ids[8], subcell_ids[8];
  PetscInt       tmp_subcell_ids[4];
  PetscInt       tmp_cell_ids[4], tmp_edge_ids[4], tmp_ncells, tmp_nedges;
  PetscInt       count;
  PetscInt       start_idx;
  PetscBool      is_cell[8];
  PetscBool      is_internal_edge[8];
  PetscBool      boundary_edge_present;
  PetscInt       i;
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  edges    = &mesh->edges;
  vertices = &mesh->vertices;

  ncells = vertices->num_internal_cells[ivertex];
  nedges = vertices->num_edges[ivertex];
  count  = 0;

  PetscInt offsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt offsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt offsetEdge    = vertices->edge_offset[ivertex];

  // compute angle to all cell centroids w.r.t. the shared vertix
  for (i=0; i<ncells; i++) {
    icell = vertices->internal_cell_ids[offsetCell + i];

    x = cells->centroid[icell].X[0] - vertices->coordinate[ivertex].X[0];
    y = cells->centroid[icell].X[1] - vertices->coordinate[ivertex].X[1];

    ids[count]              = icell;
    idx[count]              = count;
    subcell_ids[count]      = vertices->subcell_ids[offsetSubcell + i];
    is_cell[count]          = PETSC_TRUE;
    is_internal_edge[count] = PETSC_FALSE;

    ierr = ComputeTheta(x, y, &theta[count]);

    count++;
  }

  // compute angle to face centroids w.r.t. the shared vertix
  boundary_edge_present = PETSC_FALSE;

  for (i=0; i<nedges; i++) {
    iedge = vertices->edge_ids[offsetEdge + i];
    x = edges->centroid[iedge].X[0] - vertices->coordinate[ivertex].X[0];
    y = edges->centroid[iedge].X[1] - vertices->coordinate[ivertex].X[1];

    ids[count]              = iedge;
    idx[count]              = count;
    subcell_ids[count]      = -1;
    is_cell[count]          = PETSC_FALSE;
    is_internal_edge[count] = edges->is_internal[iedge];

    if (!edges->is_internal[iedge]) boundary_edge_present = PETSC_TRUE;

    ierr = ComputeTheta(x, y, &theta[count]); CHKERRQ(ierr);
    count++;

  }

  // sort the thetas in anti-clockwise direction
  PetscSortRealWithPermutation(count, theta, idx);

  // determine the starting sorted index
  start_idx = -1;
  if (boundary_edge_present) {
    // for a boundary vertex, find the last boundary edge in the
    // anitclockwise direction around the vertex
    for (i=0; i<count; i++) {
      if (!is_cell[idx[i]]) { // is this an edge?
        if (!is_internal_edge[idx[i]]) { // is this a boundary edge?
          start_idx = i;
        }
      }
    }
    // if the starting index is the last index AND the first index
    // is an edge index, reset the starting index to be the first index
    if (start_idx == count-1 && !is_cell[idx[0]]) start_idx = 0;

  } else {
    // For an internal vertex, the starting index should be a
    // face centroid
    if ( is_cell[idx[0]] ) start_idx = count-1;
    else                   start_idx = 0; // assuming that is_cell[idx[1]] = TRUE
  }

  tmp_ncells = 0;
  tmp_nedges = 0;

  // save information about cell/vetex from [start_idx+1:count] in temporary arrays
  for (i=start_idx+1; i<count; i++) {

    if (is_cell[idx[i]]) {
      tmp_cell_ids   [tmp_ncells] = ids[idx[i]];
      tmp_subcell_ids[tmp_ncells] = subcell_ids[idx[i]];;
      tmp_ncells++;
    } else {
      tmp_edge_ids   [tmp_nedges] = ids[idx[i]];
      tmp_nedges++;
    }
  }

  // save information about cell/vetex from [0:start_idx] in temporary arrays
  for (i=0; i<=start_idx; i++) {
    if (is_cell[idx[i]]) {
      tmp_cell_ids   [tmp_ncells] = ids[idx[i]];
      tmp_subcell_ids[tmp_ncells] = subcell_ids[idx[i]];;
      tmp_ncells++;
    } else {
      tmp_edge_ids   [tmp_nedges] = ids[idx[i]];
      tmp_nedges++;
    }
  }

  // save information about sorted cell ids
  for (i=0; i<tmp_ncells; i++) {
    vertices->internal_cell_ids[offsetCell + i] = tmp_cell_ids[i];
    vertices->subcell_ids[offsetSubcell + i]       = tmp_subcell_ids[i];
  }

  // save information about sorted edge ids
  for (i=0; i<tmp_nedges; i++) {
    vertices->edge_ids[offsetEdge + i] = tmp_edge_ids[i];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAVertex2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscInt       vStart, vEnd;
  TDy_vertex     *vertices;
  PetscInt       ivertex;
  PetscInt       edge_id_1, edge_id_2;
  TDy_edge       *edges;
  PetscReal      x,y, theta_1, theta_2;
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  edges    = &mesh->edges;
  vertices = &mesh->vertices;


  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<vEnd-vStart; ivertex++) {

    PetscInt offsetEdge = vertices->edge_offset[ivertex];

    if (vertices->num_internal_cells[ivertex] > 1) {
      ierr = UpdateCellOrientationAroundAVertex(tdy, ivertex); CHKERRQ(ierr);
    } else {

      edge_id_1 = vertices->edge_ids[offsetEdge + 0];
      edge_id_2 = vertices->edge_ids[offsetEdge + 1];

      x = edges->centroid[edge_id_1].X[0] - vertices->coordinate[ivertex].X[0];
      y = edges->centroid[edge_id_1].X[1] - vertices->coordinate[ivertex].X[1];
      ierr = ComputeTheta(x, y, &theta_1);

      x = edges->centroid[edge_id_2].X[0] - vertices->coordinate[ivertex].X[0];
      y = edges->centroid[edge_id_2].X[1] - vertices->coordinate[ivertex].X[1];
      ierr = ComputeTheta(x, y, &theta_2);

      if (theta_1 < theta_2) {
        if (theta_2 - theta_1 <= PETSC_PI) {
          vertices->edge_ids[offsetEdge + 0] = edge_id_2;
          vertices->edge_ids[offsetEdge + 1] = edge_id_1;
        } else {
          vertices->edge_ids[offsetEdge + 0] = edge_id_1;
          vertices->edge_ids[offsetEdge + 1] = edge_id_2;
        }
      } else {
        if (theta_1 - theta_2 <= PETSC_PI) {
          vertices->edge_ids[offsetEdge + 0] = edge_id_1;
          vertices->edge_ids[offsetEdge + 1] = edge_id_2;
        } else {
          vertices->edge_ids[offsetEdge + 0] = edge_id_2;
          vertices->edge_ids[offsetEdge + 1] = edge_id_1;
        }
      }
    }

  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateFaceOrderAroundAVertex3DMesh(TDy tdy) {

  /*
    For a vertex, save face ids such that all internal faces
    are listed first followed by boundary faces.
  */

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscInt       vStart, vEnd;
  TDy_vertex     *vertices;
  PetscInt       ivertex;
  TDy_face       *faces;
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;

  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<vEnd-vStart; ivertex++) {
    
    PetscInt face_ids_sorted[vertices->num_faces[ivertex]];
    PetscInt offsetFace = vertices->face_offset[ivertex];
    PetscInt count=0, iface;

    // First find all internal faces (i.e. face shared by two cells)
    for (iface=0;iface<vertices->num_faces[ivertex];iface++) {
      PetscInt face_id = vertices->face_ids[offsetFace + iface];
      if (faces->num_cells[face_id]==2) {
        face_ids_sorted[count] = face_id;
        count++;
      }
    }
    
    // Now find all boundary faces (i.e. face shared by a single cell)
    for (iface=0;iface<vertices->num_faces[ivertex];iface++) {
      PetscInt face_id = vertices->face_ids[offsetFace+iface];
      if (faces->num_cells[face_id]==1) {
        face_ids_sorted[count] = face_id;
        count++;
      }
    }

    // Save the sorted faces
    for (iface=0;iface<vertices->num_faces[ivertex];iface++) {
      vertices->face_ids[offsetFace+iface] = face_ids_sorted[iface];
    }
    
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAEdge2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscReal      dot_product;
  PetscInt       eStart, eEnd;
  PetscInt       iedge;
  TDy_cell       *cells;
  TDy_edge       *edges;
  PetscErrorCode ierr;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = &mesh->cells;
  edges = &mesh->edges;

  ierr = DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd); CHKERRQ(ierr);

  for (iedge=0; iedge<eEnd-eStart; iedge++) {
    PetscInt offsetCell = edges->cell_offset[iedge];

    if (edges->is_internal[iedge]) {
      //cell_from = &cells[edge->cell_ids[0]];
      //cell_to   = &cells[edge->cell_ids[1]];
      TDy_coordinate *cell_from_centroid = &cells->centroid[edges->cell_ids[offsetCell + 0]];
      TDy_coordinate *cell_to_centroid   = &cells->centroid[edges->cell_ids[offsetCell + 1]];

      dot_product = (cell_to_centroid->X[0] - cell_from_centroid->X[0]) * edges->normal[iedge].V[0] +
                    (cell_to_centroid->X[1] - cell_from_centroid->X[1]) * edges->normal[iedge].V[1];
      if (dot_product < 0.0) {
        PetscInt tmp = edges->cell_ids[offsetCell + 0];
        edges->cell_ids[offsetCell + 0] = edges->cell_ids[offsetCell + 1];
        edges->cell_ids[offsetCell + 1] = tmp;
      }
    } else {
      //cell_from = &cells[edge->cell_ids[0]];
      TDy_coordinate *cell_from_centroid = &cells->centroid[edges->cell_ids[offsetCell + 0]];

      dot_product = (edges->centroid[iedge].X[0] - cell_from_centroid->X[0]) *
                    edges->normal[iedge].V[0] +
                    (edges->centroid[iedge].X[1] - cell_from_centroid->X[1]) * edges->normal[iedge].V[1];
      if (dot_product < 0.0) {
        PetscInt tmp = edges->cell_ids[offsetCell + 0];
        edges->cell_ids[offsetCell + 0] = -1;
        edges->cell_ids[offsetCell + 1] = tmp;
      }

    }
  }

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeVariableContinuityPoint(PetscReal vertex[3],
    PetscReal edge[3], PetscReal alpha, PetscInt dim, PetscReal *point) {

  PetscInt d;
  PetscFunctionBegin;

  for (d=0; d<dim; d++) point[d] = (1.0 - alpha)*vertex[d] + edge[d]*alpha;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeRightNormalVector(PetscReal v1[3], PetscReal v2[3],
                                        PetscInt dim, PetscReal *normal) {

  PetscInt d;
  PetscReal vec_from_1_to_2[3];
  PetscReal norm;

  PetscFunctionBegin;

  if (dim != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,
                          "ComputeRightNormalVector only support 2D grids");

  norm = 0.0;

  for (d=0; d<dim; d++) {
    vec_from_1_to_2[d] = v2[d] - v1[d];
    norm += pow(vec_from_1_to_2[d], 2.0);
  }
  norm = pow(norm, 0.5);

  normal[0] =  vec_from_1_to_2[1]/norm;
  normal[1] = -vec_from_1_to_2[0]/norm;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeAreaOf2DTriangle(PetscReal v1[3], PetscReal v2[3],
                                       PetscReal v3[3], PetscReal *area) {

  PetscFunctionBegin;

  /*
   *
   *  v1[0] v1[1] 1.0
   *  v2[0] v2[1] 1.0
   *  v3[0] v3[1] 1.0
   *
  */

  *area = fabs(v1[0]*(v2[1] - v3[1]) - v1[1]*(v2[0] - v3[0]) + 1.0*
               (v2[0]*v3[1] - v2[1]*v3[0]))/2.0;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetIthNuVector(TDy_subcell *subcells, PetscInt isubcell, PetscInt i, PetscInt dim, PetscReal *nu_vec) {
  PetscFunctionBegin;
  PetscInt d;
  PetscInt sOffsetNu = subcells->nu_vector_offset[isubcell];

  if (i>=subcells->num_nu_vectors[isubcell]) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th nu_vec exceeds max nu_vecs");
  }
  
  for (d=0; d<dim; d++) nu_vec[d] = subcells->nu_vector[sOffsetNu + i].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetIthFaceCentroid(TDy_subcell *subcells, PetscInt isubcell, PetscInt i, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  PetscInt sOffsetNu = subcells->nu_vector_offset[isubcell];

  if (i>=subcells->num_faces[isubcell]) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th face centroid exceeds max num_faces");
  }

  for (d=0; d<dim; d++) centroid[d] = subcells->face_centroid[sOffsetNu + i].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDy_subcell* subcells, PetscInt isubcell, PetscInt face_id, PetscInt *face_idx) {

  PetscFunctionBegin;
  PetscInt i;
  *face_idx = -1;

  PetscInt sOffsetFace = subcells->face_offset[isubcell];
  for (i=0; i<subcells->num_faces[isubcell];i++) {
    if (subcells->face_ids[sOffsetFace+i] == face_id) {
      *face_idx = i;
      break;
    }
  }
  if (*face_idx == -1) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"For the given subcell, did not find any face that matched face_id");
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyEdge_GetCentroid(TDy_edge *edges, PetscInt iedge, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = edges->centroid[iedge].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyEdge_GetNormal(TDy_edge *edges, PetscInt iedge, PetscInt dim, PetscReal *normal) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) normal[d] = edges->normal[iedge].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFace_GetCentroid(TDy_face *faces, PetscInt iface, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = faces->centroid[iface].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFace_GetNormal(TDy_face *faces, PetscInt iface, PetscInt dim, PetscReal *normal) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) normal[d] = faces->normal[iface].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyVertex_GetCoordinate(TDy_vertex *vertices, PetscInt ivertex, PetscInt dim, PetscReal *coor) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) coor[d] = vertices->coordinate[ivertex].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
PetscErrorCode TDyCell_GetCentroid(TDy_cell *cell, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = cell->centroid.X[d];
  PetscFunctionReturn(0);
}
*/

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyCell_GetCentroid2(TDy_cell *cells, PetscInt icell, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = cells->centroid[icell].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode SetupSubcellsFor2DMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_subcell    *subcells;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  PetscInt       cStart, cEnd, num_subcells;
  PetscInt       icell, isubcell;
  PetscInt       dim, d;
  PetscInt       e_idx_up, e_idx_dn;
  PetscReal      cell_cen[3], e_cen_up[3], e_cen_dn[3], v_c[3];
  PetscReal      cp_up[3], cp_dn[3], nu_vec_up[3], nu_vec_dn[3];
  PetscReal      len_up, len_dn;
  PetscReal      alpha;
  PetscReal      normal[2], centroid;
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  edges    = &mesh->edges;
  vertices = &mesh->vertices;
  subcells = &mesh->subcells;

  alpha = 1.0;

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<cEnd-cStart; icell++) {

    // set pointer to cell
    //cell = &cells[icell];

    // save cell centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cells->num_subcells[icell];

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      PetscInt vertexStart = cells->vertex_offset[icell];

      //vertex  = &vertices[cells->vertex_ids[vertexStart+isubcell]];
      //subcell = &subcells[icell*num_subcells+isubcell];
      PetscInt subcell_id = icell*num_subcells+isubcell;
      PetscInt sOffsetNuVectors = subcells->nu_vector_offset[subcell_id];

      // save coorindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertices, cells->vertex_ids[vertexStart+isubcell], dim, &v_c[0]); CHKERRQ(ierr);

      // determine ids of up & down edges
      PetscInt edgeStart = cells->edge_offset[icell];
      e_idx_up = cells->edge_ids[edgeStart + isubcell];

      if (isubcell == 0) e_idx_dn = cells->edge_ids[edgeStart + num_subcells-1];
      else               e_idx_dn = cells->edge_ids[edgeStart + isubcell    -1];

      // set points to up/down edges
      //edge_up = &edges[e_idx_up];
      //edge_dn = &edges[e_idx_dn];

      // save centroids of up/down edges
      ierr = TDyEdge_GetCentroid(edges, e_idx_up, dim, &e_cen_up[0]); CHKERRQ(ierr);
      ierr = TDyEdge_GetCentroid(edges, e_idx_dn, dim, &e_cen_dn[0]); CHKERRQ(ierr);

      // compute continuity point
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_up, alpha, dim, cp_up);
      CHKERRQ(ierr);
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_dn, alpha, dim, cp_dn);
      CHKERRQ(ierr);

      // save continuity point
      for (d=0; d<dim; d++) {
        subcells->variable_continuity_coordinates[sOffsetNuVectors + 0].X[d] = cp_up[d];
        subcells->variable_continuity_coordinates[sOffsetNuVectors + 1].X[d] = cp_dn[d];
      }

      // compute the 'direction' of nu-vector
      ierr = ComputeRightNormalVector(cp_up, cell_cen, dim, nu_vec_dn); CHKERRQ(ierr);
      ierr = ComputeRightNormalVector(cell_cen, cp_dn, dim, nu_vec_up); CHKERRQ(ierr);

      // compute length of nu-vectors
      ierr = TDyComputeLength(cp_up, cell_cen, dim, &len_dn); CHKERRQ(ierr);
      ierr = TDyComputeLength(cp_dn, cell_cen, dim, &len_up); CHKERRQ(ierr);

      // save nu-vectors
      // note: length of nu-vectors is equal to length of edge diagonally
      //       opposite to the vector
      for (d=0; d<dim; d++) {
        subcells->nu_vector[sOffsetNuVectors + 0].V[d] = nu_vec_up[d]*len_up;
        subcells->nu_vector[sOffsetNuVectors + 1].V[d] = nu_vec_dn[d]*len_dn;
      }

      PetscReal area;
      ierr = ComputeAreaOf2DTriangle(cp_up, cell_cen, cp_dn, &area);
      subcells->T[subcell_id] = 2.0*area;

    }
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cells->volume[icell]), &centroid,
                                        &normal[0]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode FindNeighboringVerticesOfAFace(TDy_face *faces, PetscInt iface, PetscInt vertex_id,
                                              PetscInt neighboring_vertex_ids[2]) {

  PetscFunctionBegin;

  PetscInt vertex_ids[faces->num_vertices[iface]];
  PetscInt count = 0, ivertex, start = -1;
  PetscInt fOffsetVertex = faces->vertex_offset[iface];

  // Find the first occurance of "vertex_id" within list of vertices forming the face
  for (ivertex=0; ivertex<faces->num_vertices[iface]; ivertex++) {
    if (faces->vertex_ids[fOffsetVertex + ivertex] == vertex_id) {
      start = ivertex;
      break;
    }
  }

  if (start == -1) {
    printf("Did not find vertex id = %d within face id = %d\n",vertex_id,iface);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Stopping in FindNeighboringVerticesOfAFace");
  }

  // Save vertex indices starting from the first occurance of "vertex_id" to the last vertex
  // from the list
  for (ivertex=start; ivertex<faces->num_vertices[iface]; ivertex++){
    vertex_ids[count] = faces->vertex_ids[fOffsetVertex + ivertex];
    count++;
  }

  // Now, save vertex indices starting from 0 to the first occurance of "vertex_id"
  // from the list
  for (ivertex=0; ivertex<start; ivertex++){
    vertex_ids[count] = faces->vertex_ids[fOffsetVertex + ivertex];
    count++;
  }

  // The vertex_ids[1] and vertex_ids[end] are neighbors of "vertex_id"
  neighboring_vertex_ids[0] = vertex_ids[1];
  neighboring_vertex_ids[1] = vertex_ids[faces->num_vertices[iface]-1];

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode FindFaceIDsOfACellCommonToAVertex(PetscInt cell_id, TDy_face *faces,
                                                 TDy_vertex *vertices, PetscInt ivertex,
                                                 PetscInt f_idx[3],
                                                 PetscInt *num_shared_faces) {
  
  PetscFunctionBegin;
  
  PetscInt iface;
  //PetscInt cell_id = cell->id;
  
  *num_shared_faces = 0;
  
  // Find the faces of "cell_id" that are shared by the vertex
  for (iface=0; iface<vertices->num_faces[ivertex]; iface++){
    
    PetscInt offsetFace = vertices->face_offset[ivertex];
    PetscInt face_id = vertices->face_ids[offsetFace + iface];
    //TDy_face *face = &faces[face_id];
    PetscInt fOffsetCell = faces->cell_offset[face_id];
    
    if (faces->cell_ids[fOffsetCell + 0] == cell_id || faces->cell_ids[fOffsetCell + 1] == cell_id){
      
      // Check if the number of shared faces doesn't exceed max faces
      if ((*num_shared_faces) == 3) {
        (*num_shared_faces)++;
        break;
      }
      f_idx[*num_shared_faces] = face_id;
      (*num_shared_faces)++;
    }
  }
  
  if (*num_shared_faces != 3) {
    printf("Was expecting to find 3 faces of the cell to be shared by the vertex, but instead found %d common faces",*num_shared_faces);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported vertex type for 3D mesh");
  }
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateFaceOrientationAroundAVertex(TDy_coordinate *cell_centroid, TDy_face *faces,
                                                  TDy_vertex *vertices, PetscInt ivertex, PetscInt dim,
                                                  PetscInt f_idx[3]) {
  
  PetscFunctionBegin;
  
  PetscInt d;
  PetscErrorCode ierr;

  PetscReal a[3],b[3],c[3],axb[3],dot_prod;
  
  for (d=0; d<dim; d++) {
    a[d] = faces->centroid[f_idx[1]].X[d] - faces->centroid[f_idx[0]].X[d];
    b[d] = faces->centroid[f_idx[2]].X[d] - faces->centroid[f_idx[0]].X[d];
    c[d] = cell_centroid->X[d] - vertices->coordinate[ivertex].X[d];
  }
  
  ierr = TDyCrossProduct(a,b,axb); CHKERRQ(ierr);
  ierr = TDyDotProduct(axb,c,&dot_prod);

  // If the dot prod is positive (f_idx[0], f_idx[1], f_idx[2]) form a plane such
  // that the normal to the plane is pointing towards the centroid of the control
  // volume. If the dot product is negative, need to swap the order of face idx.
  if (dot_prod<0) {
    PetscInt tmpId = f_idx[1];
    f_idx[1] = f_idx[2];
    f_idx[2] = tmpId;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetupCell2CellConnectivity(TDy_vertex *vertices, PetscInt ivertex, TDy_cell *cells, TDy_face *faces, TDy_subcell *subcells, PetscInt **cell2cell_conn) {

  /*
   
    cell2cell_conn[i][j]:  0 if i-th and j-th cell are unconnected
                        :  1 if i-th and j-th cell are connected
                        : -1 if i-th and j-th are connected but j-th is a face (note: i-th could itself be a face)

    dimensions: (0:ncells-1)               corresponds to cells, while
                (ncells:ncells+ncells_bnd) corresponds to faces.
   
*/

  PetscFunctionBegin;

  PetscInt icell, isubcell, iface, ncells, cell_id, ncells_bnd, bnd_count;

  ncells = vertices->num_internal_cells[ivertex];
  ncells_bnd = vertices->num_boundary_cells[ivertex];
  bnd_count = 0;

  PetscInt offsetCell = vertices->internal_cell_offset[ivertex];
  PetscInt offsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt offsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  for (icell=0; icell<ncells; icell++) {

    //TDy_cell    *cell;
    //TDy_subcell *subcell;

    // Determine the cell and subcell id
    cell_id  = vertices->internal_cell_ids[offsetCell + icell];
    isubcell = vertices->subcell_ids[offsetSubcell + icell];

    // Get access to the cell and subcell
    //cell    = &cells[cell_id];
    //subcell = &subcells[cell_id*cells->num_subcells[icell]+isubcell];
    PetscInt subcell_id = cell_id*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    // Loop over all faces of the subcell
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

      //TDy_face *face = &faces[subcell->face_ids[iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Skip boundary face
      if (faces->cell_ids[fOffsetCell + 0] < 0 || faces->cell_ids[fOffsetCell + 1] < 0) {
        PetscInt cell_1, cell_2;
        if (faces->cell_ids[fOffsetCell + 0] < 0) cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 1]);
        else                       cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 0]);
      
        cell_2 = ncells + bnd_count;
        vertices->boundary_face_ids[offsetBoundaryFace + bnd_count] = face_id;
        bnd_count++;

        // Add 1 to indicate cell_1 and cell_2 are connected
        cell2cell_conn[cell_1][cell_2] = 1;
        cell2cell_conn[cell_2][cell_1] = 1;

      } else {
      // Find the index of cells given by faces->cell_ids[fOffsetCell + 0:1] within the cell id list given
      // vertex->internal_cell_ids
      PetscInt cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 0]);
      PetscInt cell_2 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 1]);

      // Add 1 to indicate cell_1 and cell_2 are connected
      cell2cell_conn[cell_1][cell_2] = 1;
      cell2cell_conn[cell_2][cell_1] = 1;
      }
    }
  }

  PetscInt ii,jj;
  for (ii=0;ii<ncells_bnd-1;ii++) {
    for (jj=ii+1;jj<ncells_bnd;jj++) {
      PetscInt face_id_1 = vertices->boundary_face_ids[offsetBoundaryFace + ii];
      PetscInt face_id_2 = vertices->boundary_face_ids[offsetBoundaryFace + jj];
      if (AreFacesNeighbors(faces,face_id_1,face_id_2)) {
        cell2cell_conn[ncells+ii][ncells+jj] = -1;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ExtractCentroidForIthJthTraversalOrder(TDy_vertex *vertices, PetscInt ivertex, TDy_face *faces, TDy_cell *cells, PetscInt cell_traversal_ij, PetscReal cen[3]) {

  PetscFunctionBegin;
  PetscInt dim = 3;
  PetscErrorCode ierr;

  if (cell_traversal_ij>=0) { // Is a cell?
    PetscInt cell_id;
    PetscInt offsetCell = vertices->internal_cell_offset[ivertex];

    cell_id = vertices->internal_cell_ids[offsetCell + cell_traversal_ij];
    ierr = TDyCell_GetCentroid2(cells, cell_id, dim, &cen[0]); CHKERRQ(ierr);

  } else { // is a face
  
    PetscInt face_id, idx;
    PetscInt offsetBoundaryFace = vertices->boundary_face_offset[ivertex];

    idx = -cell_traversal_ij - vertices->num_internal_cells[ivertex];
    face_id = vertices->boundary_face_ids[offsetBoundaryFace + idx];

    ierr = TDyFace_GetCentroid(faces, face_id, dim, &cen[0]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateIthTraversalOrder(TDy_vertex *vertices, PetscInt ivertex, TDy_face *faces, TDy_cell *cells, PetscInt i, PetscBool flip_if_dprod_is_negative, PetscInt **cell_traversal) {

  PetscFunctionBegin;

  PetscInt dim, d, j;
  PetscReal v1[3],v2[3],v3[3],v4[3],c2v_vec[3],normal[3];
  PetscReal dot_product;
  PetscErrorCode ierr;

  dim = 3;

  // Get pointers to the four cells
  j = 0; ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][j], v1);
  j = 1; ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][j], v2);
  j = 2; ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][j], v3);
  j = 3; ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][j], v4);

  // Save (x,y,z) of the four cells, and
  // a vector joining centroid of four cells and the vertex (c2v_vec)
  for (d=0; d<dim; d++) {
    c2v_vec[d] = vertices->coordinate[ivertex].X[d] - (v1[d] + v2[d] + v3[d] + v4[d])/4.0;
  }

  // Compute the normal to the plane formed by four cells
  ierr = TDyNormalToQuadrilateral(v1, v2, v3, v4, normal); CHKERRQ(ierr);

  // Determine the dot product between normal and c2v_vec
  dot_product = 0.0;
  for (d=0; d<dim; d++) dot_product += normal[d]*c2v_vec[d];

  if (flip_if_dprod_is_negative) {
    if (dot_product<0.0) { // Cell order was "c0 --> c4 --> c5 --> c2", so flip it.
      PetscInt tmp;
        tmp                  = cell_traversal[i][1] ;
        cell_traversal[i][1] = cell_traversal[i][3];
        cell_traversal[i][3] = tmp;
    }
  } else {
    if (dot_product>0.0) { // Cell order was "c3 <-- c1 <-- c6 <-- c7", so flip it.
      PetscInt tmp;
        tmp                  = cell_traversal[i][1] ;
        cell_traversal[i][1] = cell_traversal[i][3];
        cell_traversal[i][3] = tmp;
    }
  }

  PetscInt num_faces = 0;
  for (j=0;j<4;j++) if (cell_traversal[i][j]<0) num_faces++;
  if (num_faces>0) {
    if (num_faces != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"UpdateIthTraversalOrder: Unsupported num_faces");
    
    if (cell_traversal[i][0]>=0 && cell_traversal[i][1]>=0) {
      // do nothing
    } else if (cell_traversal[i][0]>=0 && cell_traversal[i][3]>=0) {
      PetscInt tmp[4];
      tmp[0] = cell_traversal[i][3];
      tmp[1] = cell_traversal[i][0];
      tmp[2] = cell_traversal[i][1];
      tmp[3] = cell_traversal[i][2];
      for (j=0;j<4;j++) cell_traversal[i][j] = tmp[j];
    } else if (cell_traversal[i][1]>=0 && cell_traversal[i][2]>=0) {
      PetscInt tmp[4];
      tmp[0] = cell_traversal[i][1];
      tmp[1] = cell_traversal[i][2];
      tmp[2] = cell_traversal[i][3];
      tmp[3] = cell_traversal[i][0];
      for (j=0;j<4;j++) cell_traversal[i][j] = tmp[j];
    } else {
      PetscInt tmp[4];
      tmp[0] = cell_traversal[i][2];
      tmp[1] = cell_traversal[i][3];
      tmp[2] = cell_traversal[i][0];
      tmp[3] = cell_traversal[i][1];
      for (j=0;j<4;j++) cell_traversal[i][j] = tmp[j];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeFirstTraversalOrder(TDy_vertex *vertices, PetscInt ivertex, TDy_face *faces, TDy_cell *cells, PetscInt **cell2cell_conn, PetscInt **cell_traversal)
{
  PetscFunctionBegin;

  PetscInt i, j, m, ncells, ncells_bnd;
  PetscErrorCode ierr;

  // Objective to find: c0 --> c2 --> c5 --> c4

  ncells    = vertices->num_internal_cells[ivertex];
  ncells_bnd= vertices->num_boundary_cells[ivertex];

  i = 0;
  m = 0;
  cell_traversal[i][0] = m; // c0

  // Find the first two cells that are connected to c0 (i.e. c2 and c4)
  PetscInt count = 0;
  for (j=0; j<ncells+ncells_bnd; j++){
    if (cell2cell_conn[i][j] == 1) {
      count++;
      cell_traversal[i][count] = j;
      if (count == 2) break;
    }
  }

  // Find the common connecting cell for the previously found two cells
  // that is not the c0 (i.e. c5)
  PetscInt cell_1 = cell_traversal[i][1];
  PetscInt cell_2 = cell_traversal[i][2];
  PetscBool found;
  found = PETSC_FALSE;
  for (j=0;j<ncells+ncells_bnd;j++){
    if (cell2cell_conn[cell_1][j] == 1 && cell2cell_conn[cell_2][j] == 1 && j != cell_traversal[i][0] ) {
      cell_traversal[i][3] = cell_traversal[i][2];
      cell_traversal[i][2] = j;
      found = PETSC_TRUE;
      break;
    }
  }

  // If no connecting cell was found, it implies that there must a be connecting boundary face
  if (!found) {
    for (j=0;j<ncells+ncells_bnd;j++){
      if (abs(cell2cell_conn[cell_1][j]) == 1 && abs(cell2cell_conn[cell_2][j]) == 1 && j != cell_traversal[i][0] ) {
        cell_traversal[i][3] = cell_traversal[i][2];
        cell_traversal[i][2] = j;
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"ComputeFirstTraversalOrder: Did not find a common cell or a boundary face");
  }

  for (j=0;j<4;j++){
    if (cell_traversal[i][j]>ncells-1) cell_traversal[i][j] = -cell_traversal[i][j];
  }

  // If the traversal order is "c0 --> c4 --> c5 --> c2" (i.e. the dot product of
  // normal to the plane formed by cells points away from the vertex), update the
  // traversal order to be "c0 --> c2 --> c5 --> c4"
  PetscBool flip_if_dprod_is_negative;
  flip_if_dprod_is_negative = PETSC_TRUE;
  ierr = UpdateIthTraversalOrder(vertices, ivertex, faces, cells, i, flip_if_dprod_is_negative, cell_traversal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeSecondTraversalOrder(TDy_vertex *vertices, PetscInt ivertex, TDy_face *faces, TDy_cell *cells, PetscInt **cell2cell_conn, PetscInt **cell_traversal) {

  PetscFunctionBegin;

  PetscInt i, j, k, m, ncells, count, cell_1, cell_2;
  PetscBool found;
  PetscErrorCode ierr;

  // Objective to find: c3 <-- c7 <-- c6 <-- c1

  i = 1;
  ncells = vertices->num_internal_cells[ivertex];

  // Find a cell that is not part of the cells in cell_traversal[0][:] (i.e. c3)
  for (j=0;j<ncells;j++){
    found = PETSC_FALSE;
    for (k=0;k<4;k++) {
      if (cell_traversal[i-1][k] == j) {
        found = PETSC_TRUE;
        break;
      }
    }
    if (found == PETSC_FALSE) {
      cell_traversal[i][0] = j;
      break;
    }
  }

  // Find the first two cells that are connected to c3 and
  // are not part of the cells in cell_traversal[0][:] (i.e. c7 and c1)
  m = cell_traversal[i][0];
  count = 0;
  for (j=0;j<ncells;j++) {
    if (cell2cell_conn[m][j] == 1) { // Is m-th cell connected to j-th cell?
      found = PETSC_FALSE;
      for (k=0;k<4;k++) {
        if (cell_traversal[i-1][k] == j) { // Is the j-th cell part of cell_traversal[i-1][:]
          found = PETSC_TRUE;
          break;
        }
      }
      if (found == PETSC_FALSE) {
        count++;
        cell_traversal[i][count] = j;
        if (count==2) break;
      }
    }
  }

  // Find the common connecting cell for the previously found two cells
  // that is not c3 (i.e. c6)
  cell_1 = cell_traversal[i][1];
  cell_2 = cell_traversal[i][2];
  for (j=0;j<ncells;j++){
    if (cell2cell_conn[cell_1][j] == 1 && cell2cell_conn[cell_2][j] == 1 && j != cell_traversal[i][0] ) {
      cell_traversal[i][3] = cell_traversal[i][2];
      cell_traversal[i][2] = j;
    }
  }

  // If the traversal order is "c3 <-- c1 <-- c6 <-- c7" (i.e. the dot product of
  // normal to the plane formed by cells points towards from the vertex), udpate the
  // traversal order to be "c3 <-- c7 <-- c6 <-- c1"
  PetscBool flip_if_dprod_is_negative;
  flip_if_dprod_is_negative = PETSC_FALSE;
  ierr = UpdateIthTraversalOrder(vertices, ivertex, faces, cells, i, flip_if_dprod_is_negative, cell_traversal); CHKERRQ(ierr);

  // Rearrange cell_traversal[1][:] such that
  // cell_traversal[0][0] and cell_traversal[1][0] are connected
  cell_1 = cell_traversal[0][0];
  PetscInt idx_beg=0;
  for (j=0;j<4;j++) {
    cell_2 = cell_traversal[1][j];
    if ( cell2cell_conn[cell_1][cell_2] == 1) {
      idx_beg = j;
      break;
    }
  }
  if (idx_beg>0) {
    PetscInt tmp[4];
    count=0;
    for (j=idx_beg;j<4;j++) {
      tmp[count] = cell_traversal[1][j];
      count++;
    }
    for (j=0;j<idx_beg;j++) {
      tmp[count] = cell_traversal[1][j];
      count++;
    }
    for (j=0;j<4;j++) cell_traversal[1][j] = tmp[j];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetupUpwindFacesForSubcell(TDy_vertex *vertices, PetscInt ivertex, TDy_cell *cells, TDy_face *faces, TDy_subcell *subcells, PetscInt **cell_up2dw) {

  PetscFunctionBegin;

  PetscInt icell, isubcell, iface, ncells, cell_id, nflux_in;
  PetscInt ii, boundary_cell_count;

  ncells = vertices->num_internal_cells[ivertex];
  boundary_cell_count = 0;

  switch (ncells) {
  case 1:
    nflux_in = 0;
    break;
  case 2:
    nflux_in = 1;
    break;
  case 4:
    nflux_in = 4;
    break;
  case 8:
    nflux_in = 12;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"SetupUpwindFacesForSubcell: Unsupported vertex->num_internal_cells");
    break;
  }

  PetscInt offsetCell = vertices->internal_cell_offset[ivertex];
  PetscInt offsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt offsetFace = vertices->face_offset[ivertex];

  for (icell=0; icell<ncells; icell++) {

    //TDy_subcell *subcell;

    // Determine the cell and subcell id
    cell_id  = vertices->internal_cell_ids[offsetCell + icell];
    isubcell = vertices->subcell_ids[offsetSubcell + icell];

    // Get access to the cell and subcell
    //cell    = &cells[cell_id];
    //subcell = &subcells[cell_id*cells->num_subcells[cell_id]+isubcell];
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    // Loop over all faces of the subcell
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

      //TDy_face *face = &faces[subcell->face_ids[iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Skip boundary face
      if (faces->cell_ids[fOffsetCell + 0] < 0 || faces->cell_ids[fOffsetCell + 1] < 0) {
        subcells->face_unknown_idx[sOffsetFace + iface] = boundary_cell_count+nflux_in;
        for (ii=0; ii<12; ii++) {
          if (cell_up2dw[ii][0] == subcells->face_unknown_idx[sOffsetFace + iface]){ subcells->is_face_up[sOffsetFace + iface] = PETSC_FALSE; break;}
          if (cell_up2dw[ii][1] == subcells->face_unknown_idx[sOffsetFace + iface]){ subcells->is_face_up[sOffsetFace + iface] = PETSC_TRUE ; break;}
        }
        boundary_cell_count++;
        continue;
      }

      // Find the index of cells given by faces->cell_ids[fOffsetCell + 0:1] within the cell id list given
      // vertex->internal_cell_ids
      PetscInt cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 0]);
      PetscInt cell_2 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 1]);

      for (ii=0; ii<12; ii++) {
        if (cell_up2dw[ii][0] == cell_1 && cell_up2dw[ii][1] == cell_2) {

          subcells->face_unknown_idx[sOffsetFace + iface] = ii;
          if (cells->id[cell_id] == faces->cell_ids[fOffsetCell + 0])  subcells->is_face_up[sOffsetFace + iface] = PETSC_TRUE;
          else                                                         subcells->is_face_up[sOffsetFace + iface] = PETSC_FALSE;
          break;
        } else if (cell_up2dw[ii][0] == cell_2 && cell_up2dw[ii][1] == cell_1) {

          subcells->face_unknown_idx[sOffsetFace + iface] = ii;
          if (cells->id[cell_id] == faces->cell_ids[fOffsetCell + 1]) subcells->is_face_up[sOffsetFace + iface] = PETSC_TRUE;
          else                                                        subcells->is_face_up[sOffsetFace + iface] = PETSC_FALSE;
          break;
        }
      }
    }
  }

  PetscInt nup_bnd_flux=0, ndn_bnd_flux=0;
  PetscInt nflux_bc = 0;

  nflux_bc = vertices->num_boundary_cells[ivertex]/2;

  // Save the face index that corresponds to the flux in transmissibility matrix
  for (icell=0; icell<ncells; icell++) {

    //TDy_subcell *subcell;

    // Determine the cell and subcell id
    cell_id  = vertices->internal_cell_ids[offsetCell+icell];
    isubcell = vertices->subcell_ids[offsetSubcell+icell];

    // Get access to the cell and subcell
    //cell    = &cells[cell_id];
    //subcell = &subcells[cell_id*cells->num_subcells[cell_id]+isubcell];
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    // Loop over all faces of the subcell and
    // - Updates face_ids for a vertex such that first all
    //   internal faces are listed, followed by upwind boundary
    //   faces, and the downward boundary faces are last
    // - Similary, the index of the flux through the faces of a
    //   subcell are identifed. The internal fluxes are first,
    //   followed by upwind boundary and downwind boundary faces.
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {
      //TDy_face *face = &faces[subcell->face_ids[iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];

      PetscInt idx_flux = subcells->face_unknown_idx[sOffsetFace + iface];
      if (faces->is_internal[face_id]) {
        vertices->face_ids[offsetFace+idx_flux] = face_id;
        subcells->face_flux_idx[sOffsetFace + iface] = idx_flux;
      } else {
        if (subcells->is_face_up[sOffsetFace + iface]) {
          vertices->face_ids[offsetFace+nflux_in+nup_bnd_flux] = face_id;
          subcells->face_flux_idx[sOffsetFace + iface] = nflux_in+nup_bnd_flux;
          nup_bnd_flux++;
        } else {
          vertices->face_ids[offsetFace+nflux_in+nflux_bc+ndn_bnd_flux] = face_id;
          subcells->face_flux_idx[sOffsetFace + iface] = nflux_in+ndn_bnd_flux;
          ndn_bnd_flux++;
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode DetermineUpwindFacesForSubcell(TDy tdy, PetscInt ivertex) {

  PetscFunctionBegin;
  /*
  
    The "vertex" (not shown here) is shared by

            8 cells
        (internal vertex)
         c1 ------------- c6           c1 ------------- c6           c1 ------------- c6
        /|               /|           /|               /|           /|               /|
       / |              / |          / |              / |          / |              / |
      /  |             /  |         /  |             /  |         /  |             /  |
     /   |            /   |        /   |            /   |        /   |            /   |
    c3 --|---------- c7   |       f3   |           f7   |       c3 --|---------- c7   |
    |    |           |    |            |                |       |    |           |    |
    |    c4 ---------|--- c5           f4               f5      |    f4          |    f5
    |   /            |   /                                      |                |
    |  /             |  /                                       |                |
    | /              | /                                        |                |
    |/               |/                                         |                |
    c0 ------------- c2                                        f0               f2

    Traversal for above cells is given as:
    c0 --> c2 --> c5 --> c4       c1 --> c6 --> f7 --> f3       c3 --> c1 --> c6 --> c7
     |      ^      |      ^        |      ^                      |      ^      |      ^
     |      |      |      |        |      |                      |      |      |      |
     |      |      |      |        |      |                      |      |      |      |
     v      |      v      |        v      |                      v      |      v      |
    c3 <-- c7 <-- c6 <-- c1       f4 <-- f7                     f0 <-- f4 <-- f5 <-- f2

  */

  TDy_face *faces;
  TDy_cell *cells;
  TDy_subcell *subcells;
  TDy_vertex *vertices;

  PetscInt ncells,ncells_bnd;
  PetscInt i, j, k;
  PetscInt **cell_traversal;
  PetscInt max_faces = 2;
  PetscInt max_cells_per_face = 4;
  PetscInt **cell2cell_conn;
  PetscInt count;
  PetscInt **cell_up2dw;
  PetscErrorCode ierr;

  cells = &tdy->mesh->cells;
  faces = &tdy->mesh->faces;
  subcells = &tdy->mesh->subcells;
  vertices = &tdy->mesh->vertices;

  ncells    = vertices->num_internal_cells[ivertex];
  ncells_bnd= vertices->num_boundary_cells[ivertex];

  switch (ncells) {
  case 2:
    break;
  case 4:
    break;
  case 8:
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"DetermineUpwindFacesForSubcell: Unsupported vertex->num_internal_cells ");
    break;
  }

  ierr = TDyAllocate_IntegerArray_2D(&cell2cell_conn, ncells+ncells_bnd, ncells+ncells_bnd); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_2D(cell2cell_conn, ncells+ncells_bnd, ncells+ncells_bnd, 0); CHKERRQ(ierr);

  // For all cells that are common to the give vertex, create a matrix (cell2cell_conn)
  // that stores information about which cell is connected to which other cell
  ierr = SetupCell2CellConnectivity(vertices, ivertex, cells, faces, subcells, cell2cell_conn); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_2D(&cell_traversal, max_faces, max_cells_per_face); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_2D(cell_traversal, max_faces, max_cells_per_face, -(ncells+ncells_bnd)); CHKERRQ(ierr);

  // First traversal:
  //   c0 --> c2 --> c5 --> c4, or
  //   c1 --> c6 --> f7 --> f3, or
  //   c3 --> c1 --> c6 --> c7
  ierr = ComputeFirstTraversalOrder(vertices, ivertex, faces,cells,cell2cell_conn,cell_traversal); CHKERRQ(ierr);

  // Second traversal:
  if (ncells == 2) {
    // f4 <--- f7
    i = cell_traversal[0][0];
    for (j=ncells;j<ncells+ncells_bnd;j++) {
      if (abs(cell2cell_conn[i][j]) == 1){
        PetscInt found = PETSC_FALSE;
        for (k=0;k<max_cells_per_face;k++) {
          if ( abs(cell_traversal[0][k]) == j) found = PETSC_TRUE;
        }
        if (!found) cell_traversal[1][0] = j;
      }
    }
    
    i = cell_traversal[0][1];
    for (j=ncells;j<ncells+ncells_bnd;j++) {
      if (abs(cell2cell_conn[i][j]) == 1){
        PetscInt found = PETSC_FALSE;
        for (k=0;k<max_cells_per_face;k++) {
          if ( abs(cell_traversal[0][k]) == j) found = PETSC_TRUE;
        }
        if (!found) cell_traversal[1][1] = j;
      }
    }
    
    cell_traversal[0][2] = -cell_traversal[0][2];
    cell_traversal[0][3] = -cell_traversal[0][3];
  }

  if (ncells == 4) {
    // Find c3 <-- c7 <-- f5 <-- f2
    for (j=0;j<4;j++) cell_traversal[1][j] = cell_traversal[0][j]+4;
  }

  if (ncells == 8) {
    // Find c3 <-- c7 <-- c6 <-- c1
    ierr = ComputeSecondTraversalOrder(vertices,ivertex,faces,cells,cell2cell_conn,cell_traversal); CHKERRQ(ierr);
  }

  ierr = TDyAllocate_IntegerArray_2D(&cell_up2dw, 12, 2); CHKERRQ(ierr);

  if (ncells == 2) {
    // c1 --> c6 --> f7 --> f3
    // [0]    [1]    [2]    [3]
    //  |      ^
    //  |      |
    //  v      |
    // f4 <-- f7
    // [4]    [5]
    //
    // Note: numbers in [*] are local indices of cells and faces

    count = 0;

    // c1 --> c6
    cell_up2dw[count][0] = cell_traversal[0][0]; cell_up2dw[count][1] = cell_traversal[0][1]; count++;

    // c6 --> f7
    cell_up2dw[count][0] = cell_traversal[0][1]; cell_up2dw[count][1] = cell_traversal[0][2]; count++;

    // f3 --> c1
    cell_up2dw[count][0] = cell_traversal[0][3]; cell_up2dw[count][1] = cell_traversal[0][0]; count++;

    // c1 --> f4
    cell_up2dw[count][0] = cell_traversal[0][0]; cell_up2dw[count][1] = cell_traversal[1][0]; count++;

    // f7 --> c6
    cell_up2dw[count][0] = cell_traversal[1][1]; cell_up2dw[count][1] = cell_traversal[0][1]; count++;
  }

  // c3 --> c1 --> c6 --> c7
  //  |      ^      |      ^
  //  |      |      |      |
  //  v      |      v      |
  // f0 <-- f4 <-- f5 <-- f2
  if (ncells >= 4) {

    // c3 --> c1 --> c6 --> c7
    i=0;
    count=0;
    for (j=0;j<4;j++){
      cell_up2dw[count][0] = cell_traversal[i][j];
      if (j<3) cell_up2dw[count][1] = cell_traversal[i][j+1];
      else     cell_up2dw[count][1] = cell_traversal[i][0];
      count++;
    }
    
    // c3 --> f0
    cell_up2dw[count][0] = cell_traversal[0][0]; cell_up2dw[count][1] = cell_traversal[1][0]; count++;

    // f4 --> c1
    cell_up2dw[count][0] = cell_traversal[1][1]; cell_up2dw[count][1] = cell_traversal[0][1]; count++;

    // c6 --> f5
    cell_up2dw[count][0] = cell_traversal[0][2]; cell_up2dw[count][1] = cell_traversal[1][2]; count++;

    // f2 --> c7
    cell_up2dw[count][0] = cell_traversal[1][3]; cell_up2dw[count][1] = cell_traversal[0][3]; count++;
  }

  if (ncells == 8) {
    count=0;

    // c0 --> c2 --> c5 --> c4
    i=0;
    for (j=0;j<4;j++){
      cell_up2dw[count][0] = cell_traversal[i][j];
      if (j<3) cell_up2dw[count][1] = cell_traversal[i][j+1];
      else     cell_up2dw[count][1] = cell_traversal[i][0];
      count++;
    }

    // c3 <-- c7 <-- c6 <-- c1
    i=1;
    for (j=0;j<4;j++){
      cell_up2dw[count][1] = cell_traversal[i][j];
      if (j<3) cell_up2dw[count][0] = cell_traversal[i][j+1];
      else     cell_up2dw[count][0] = cell_traversal[i][0];
      count++;
    }

    // c5 --> c6
    cell_up2dw[count][0] = cell_traversal[0][2]; cell_up2dw[count][1] = cell_traversal[1][2]; count++;

    // c7 --> c2
    cell_up2dw[count][0] = cell_traversal[1][1]; cell_up2dw[count][1] = cell_traversal[0][1]; count++;

    // c0 --> c3
    cell_up2dw[count][0] = cell_traversal[0][0]; cell_up2dw[count][1] = cell_traversal[1][0]; count++;

    // c1 --> c4
    cell_up2dw[count][0] = cell_traversal[1][3]; cell_up2dw[count][1] = cell_traversal[0][3]; count++;

  }
  
  ierr = SetupUpwindFacesForSubcell(vertices,ivertex,cells,faces,subcells,cell_up2dw); CHKERRQ(ierr);

  ierr = TDyDeallocate_IntegerArray_2D(cell2cell_conn, ncells); CHKERRQ(ierr);
  ierr = TDyDeallocate_IntegerArray_2D(cell_traversal, max_faces); CHKERRQ(ierr);
  ierr = TDyDeallocate_IntegerArray_2D(cell_up2dw, 12); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode SetupSubcellsFor3DMesh(TDy tdy) {

  /*
    For each subcell:
      - Determine face IDs in such a order so the normal to plane formed by
        centroid of face IDs points toward the vertex of cell shared by subcell
      - Compute area of faces
      - Compute nu_vector
      - Compute volume of subcell (and subsequently compute the volume of cell)
  */

  PetscFunctionBegin;

  DM dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_subcell    *subcells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  PetscInt       cStart, cEnd, num_subcells;
  PetscInt       icell, isubcell, ivertex;
  PetscInt       dim, d;
  PetscReal      cell_cen[3], v_c[3];
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  subcells = &mesh->subcells;

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<cEnd-cStart; icell++) {

    // set pointer to cell
    //cell = &cells[icell];

    // save cell centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cells->num_subcells[icell];

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      PetscInt vStart = cells->vertex_offset[icell];
      //vertex  = &vertices[cells->vertex_ids[vStart+isubcell]];
      ivertex = cells->vertex_ids[vStart+isubcell];
      //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->face_offset[subcell_id];
      PetscInt sOffsetNuVec = subcells->nu_vector_offset[subcell_id];

      // save coorindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertices, ivertex, dim, &v_c[0]); CHKERRQ(ierr);

      PetscInt num_shared_faces;

      // For a given cell, find all face ids that are share a vertex
      PetscInt f_idx[3];
      for (d=0;d<3;d++) f_idx[d] = subcells->face_ids[sOffsetFace + d];

      ierr = FindFaceIDsOfACellCommonToAVertex(cells->id[icell], faces, vertices, ivertex, f_idx, &num_shared_faces);

      // Update order of faces in f_idx so (face_ids[0], face_ids[1], face_ids[2])
      // form a plane such that normal to plane points toward the cell centroid
      ierr = UpdateFaceOrientationAroundAVertex(&cells->centroid[icell], faces, vertices, ivertex, dim, f_idx);
      for (d=0;d<3;d++) subcells->face_ids[sOffsetFace + d] = f_idx[d];

      PetscReal face_cen[3][3];

      PetscInt iface;
      for (iface=0; iface<num_shared_faces; iface++){

        /*      n1 ------------------ x
                /                    /
               /                    /
              /                    /
             e1 ------- fc[iface] /
            /          /         /
           /          /         /
          vc ------- e0 -------n0
        */
  
        PetscReal f_normal[3];

        //face = &faces[subcell->face_ids[iface]];
        //PetscInt face_id = subcell->face_ids[iface];
        PetscInt face_id = subcells->face_ids[sOffsetFace + iface];

        ierr = TDyFace_GetCentroid(faces, face_id, dim, &face_cen[iface][0]); CHKERRQ(ierr);

        PetscInt neighboring_vertex_ids[2];

        // Find 'n0' and 'n1'
        ierr = FindNeighboringVerticesOfAFace(faces,face_id,ivertex,neighboring_vertex_ids);
        
        PetscReal edge0_cen[3], edge1_cen[3];

        for (d=0; d<dim; d++) {
          edge0_cen[d] = (v_c[d] + vertices->coordinate[neighboring_vertex_ids[0]].X[d])/2.0;
          edge1_cen[d] = (v_c[d] + vertices->coordinate[neighboring_vertex_ids[1]].X[d])/2.0;
          subcells->face_centroid[sOffsetFace + iface].X[d] = (v_c[d] + edge0_cen[d] + face_cen[iface][d] + edge1_cen[d])/4.0;
        }

        // area of face
        ierr = TDyQuadrilateralArea(v_c, edge0_cen, face_cen[iface], edge1_cen, &subcells->face_area[sOffsetFace + iface]);

        // nu_vec on the "iface"-th is given as:
        //  = (x_{iface+1} - x_{cell_centroid}) x (x_{iface+2} - x_{cell_centroid})
        //  = (x_{f1_idx } - x_{cell_centroid}) x (x_{f2_idx } - x_{cell_centroid})
        PetscInt f1_idx, f2_idx;

        // determin the f1_idx and f2_idx
        PetscReal f1[3], f2[3];
        switch (iface) {
          case 0:
          f1_idx = 1;
          f2_idx = 2;
          break;
          
          case 1:
          f1_idx = 2;
          f2_idx = 0;
          break;

          case 2:
          f1_idx = 0;
          f2_idx = 1;
          break;

          default:
          break;
        }
        
        // Save x_{f1_idx } and x_{f2_idx }
        //TDy_face *face1, *face2;
        //face1 = &faces[subcell->face_ids[f1_idx]];
        //face2 = &faces[subcell->face_ids[f2_idx]];
        PetscInt face_id_1 = subcells->face_ids[sOffsetFace + f1_idx];
        PetscInt face_id_2 = subcells->face_ids[sOffsetFace + f2_idx];

        ierr = TDyFace_GetCentroid(faces, face_id_1, dim, &f1[0]); CHKERRQ(ierr);
        ierr = TDyFace_GetCentroid(faces, face_id_2, dim, &f2[0]); CHKERRQ(ierr);

        // Compute (x_{f1_idx } - x_{cell_centroid}) x (x_{f2_idx } - x_{cell_centroid})
        ierr = TDyNormalToTriangle(cell_cen, f1, f2, f_normal);

        // Save the data
        for (d=0; d<dim; d++) subcells->nu_vector[sOffsetNuVec + iface].V[d] = f_normal[d];

      }

      PetscReal volume;
      ierr = TDyComputeVolumeOfTetrahedron(cell_cen, face_cen[0], face_cen[1], face_cen[2], &volume); CHKERRQ(ierr);
      subcells->T[subcell_id] = volume*6.0;
    }
    PetscReal normal[3], centroid;
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cells->volume[icell]), &centroid,
                                        &normal[0]); CHKERRQ(ierr);

  }

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    if (vertices->num_internal_cells[ivertex] > 1 && vertices->is_local[ivertex]) {
      ierr = DetermineUpwindFacesForSubcell(tdy, ivertex );
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateCellOrientationAroundAFace3DMesh(TDy tdy) {

  /*
  
  Ensure the order of cell_ids for a given face is such that:
    Vector from faces->cell_ids[fOffsetCell + 0] to faces->cell_ids[fOffsetCell + 1] points in the direction
    of the normal vector to the face.
  
  */

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscInt       iface, dim;
  TDy_vertex     *vertices;
  TDy_cell       *cells;
  TDy_face       *faces;
  PetscErrorCode ierr;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = &mesh->cells;
  faces = &mesh->faces;
  vertices = &mesh->vertices;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {
    //face = &(faces[iface]);

    PetscReal v1[3], v2[3], v3[3], v4[3], normal[3];
    PetscReal f_cen[3], c_cen[3], f2c[3], dot_prod;
    PetscInt d;

    /*
    vertex = &vertices[face->vertex_ids[0]];
    vertex = &vertices[face->vertex_ids[1]];
    vertex = &vertices[face->vertex_ids[2]];
    vertex = &vertices[face->vertex_ids[3]];
    */
    PetscInt fOffsetVertex = faces->vertex_offset[iface];
    PetscInt fOffsetCell = faces->cell_offset[iface];

    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 0], dim, &v1[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 1], dim, &v2[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 2], dim, &v3[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 3], dim, &v4[0]); CHKERRQ(ierr);

    ierr = TDyFace_GetCentroid(faces, iface, dim, &f_cen[0]); CHKERRQ(ierr);

    ierr = TDyCell_GetCentroid2(cells, faces->cell_ids[fOffsetCell+0], dim, &c_cen[0]); CHKERRQ(ierr);

    ierr = TDyNormalToQuadrilateral(v1, v2, v3, v4, normal); CHKERRQ(ierr);
    for (d=0; d<dim; d++){ faces->normal[iface].V[d] = normal[d];}

    ierr = TDyCreateVecJoiningTwoVertices(f_cen, c_cen, f2c); CHKERRQ(ierr);
    ierr = TDyDotProduct(normal,f2c,&dot_prod); CHKERRQ(ierr);
    if ( dot_prod > 0.0 ) {
      PetscInt tmp;
      tmp = faces->cell_ids[fOffsetCell + 0];
      faces->cell_ids[fOffsetCell + 0] = faces->cell_ids[fOffsetCell + 1];
      faces->cell_ids[fOffsetCell + 1] = tmp;
    }
  }
  
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputCells2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_subcell    *subcells;
  PetscInt       dim;
  PetscInt       icell, d, k;
  PetscErrorCode ierr;

  dm = tdy->dm;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  subcells = &mesh->subcells;

  Vec cell_cen, cell_vol;
  Vec cell_neigh_ids, cell_vertex_ids, cell_edge_ids;
  Vec scell_nu, scell_cp, scell_vol, scell_gmatrix;

  PetscScalar *cell_cen_v, *cell_vol_v;
  PetscScalar *neigh_id_v, *vertex_id_v, *edge_id_v;
  PetscScalar *scell_nu_v, *scell_cp_v, *scell_vol_v, *scell_gmatrix_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*dim, &cell_cen);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells, &cell_vol);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4, &cell_neigh_ids);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4, &cell_vertex_ids);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4, &cell_edge_ids);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4*dim*2, &scell_nu);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4*dim*2, &scell_cp);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4, &scell_vol);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4*4, &scell_gmatrix);
  CHKERRQ(ierr);

  ierr = VecGetArray(cell_cen, &cell_cen_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_vol, &cell_vol_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_neigh_ids, &neigh_id_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_vertex_ids, &vertex_id_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_edge_ids, &edge_id_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_nu, &scell_nu_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_cp, &scell_cp_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_vol, &scell_vol_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_gmatrix, &scell_gmatrix_v); CHKERRQ(ierr);

  PetscInt count = 0;
  for (icell = 0; icell < mesh->num_cells; icell++) {

    // save centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen_v[icell*dim]); CHKERRQ(ierr);

    // save volume
    cell_vol_v[icell] = cells->volume[icell];

    PetscInt nStart = cells->neighbor_offset[icell];
    PetscInt vStart = cells->vertex_offset[icell];
    PetscInt eStart = cells->edge_offset[icell];

    for (k=0; k<4; k++) {
      neigh_id_v [icell*4 + k] = cells->neighbor_ids[nStart+k];
      vertex_id_v[icell*4 + k] = cells->vertex_ids[vStart+k];
      edge_id_v  [icell*4 + k] = cells->edge_ids[eStart+k];

      //subcell = &subcells[icell*cells->num_subcells[icell]+k];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+k;

      scell_vol_v[icell*4 + k] = subcells->T[subcell_id];

      scell_gmatrix_v[icell*4*4 + k*4 + 0] = tdy->subc_Gmatrix[icell][k][0][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 1] = tdy->subc_Gmatrix[icell][k][0][1];
      scell_gmatrix_v[icell*4*4 + k*4 + 2] = tdy->subc_Gmatrix[icell][k][1][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 3] = tdy->subc_Gmatrix[icell][k][1][1];

      ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, 0, dim, &scell_nu_v[count]); CHKERRQ(ierr);
      PetscInt sOffetNuVectors = subcells->nu_vector_offset[subcell_id];
      for (d=0; d<dim; d++) {
        scell_cp_v[count] = subcells->variable_continuity_coordinates[sOffetNuVectors + 0].X[d];
        count++;
      }

      ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, 1, dim, &scell_nu_v[count]); CHKERRQ(ierr);
      for (d=0; d<dim; d++) {
        scell_cp_v[count] = subcells->variable_continuity_coordinates[sOffetNuVectors + 1].X[d];
        count++;
      }
    }

  }

  ierr = VecRestoreArray(cell_cen, &cell_cen_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_vol, &cell_vol_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_neigh_ids, &neigh_id_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_vertex_ids, &vertex_id_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_edge_ids, &edge_id_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_nu, &scell_nu_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_cp, &scell_cp_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_vol, &scell_vol_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_gmatrix, &scell_gmatrix_v); CHKERRQ(ierr);

  ierr = TDySavePetscVecAsBinary(cell_cen, "cell_cen.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(cell_vol, "cell_vol.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(cell_neigh_ids, "cell_neigh_ids.bin");
  CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(cell_vertex_ids, "cell_vertex_ids.bin");
  CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(cell_edge_ids, "cell_edge_ids.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(scell_nu, "subcell_nu.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(scell_cp, "subcell_cp.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(scell_vol, "subcell_vol.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(scell_gmatrix, "subcell_gmatrix.bin");
  CHKERRQ(ierr);

  ierr = VecDestroy(&cell_cen); CHKERRQ(ierr);
  ierr = VecDestroy(&cell_vol); CHKERRQ(ierr);
  ierr = VecDestroy(&cell_neigh_ids); CHKERRQ(ierr);
  ierr = VecDestroy(&cell_vertex_ids); CHKERRQ(ierr);
  ierr = VecDestroy(&cell_edge_ids); CHKERRQ(ierr);
  ierr = VecDestroy(&scell_nu); CHKERRQ(ierr);
  ierr = VecDestroy(&scell_cp); CHKERRQ(ierr);
  ierr = VecDestroy(&scell_vol); CHKERRQ(ierr);
  ierr = VecDestroy(&scell_gmatrix); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputEdges2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_edge       *edges;
  PetscInt       dim;
  PetscInt       iedge;
  PetscErrorCode ierr;

  dm = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh  = tdy->mesh;
  edges = &mesh->edges;

  Vec edge_cen, edge_nor;
  PetscScalar *edge_cen_v, *edge_nor_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_cen);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_nor);
  CHKERRQ(ierr);

  ierr = VecGetArray(edge_cen, &edge_cen_v); CHKERRQ(ierr);
  ierr = VecGetArray(edge_nor, &edge_nor_v); CHKERRQ(ierr);

  for (iedge=0; iedge<mesh->num_edges; iedge++) {
    ierr = TDyEdge_GetCentroid(edges, iedge, dim, &edge_cen_v[iedge*dim]); CHKERRQ(ierr);
    ierr = TDyEdge_GetNormal(  edges, iedge, dim, &edge_nor_v[iedge*dim]); CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(edge_cen, &edge_cen_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(edge_nor, &edge_nor_v); CHKERRQ(ierr);

  ierr = TDySavePetscVecAsBinary(edge_cen, "edge_cen.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(edge_nor, "edge_nor.bin"); CHKERRQ(ierr);

  ierr = VecDestroy(&edge_cen); CHKERRQ(ierr);
  ierr = VecDestroy(&edge_nor); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputVertices2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_vertex     *vertices;
  PetscInt       dim;
  PetscInt       ivertex, i;
  PetscErrorCode ierr;

  dm = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  vertices = &mesh->vertices;

  Vec vert_coord, vert_icell_ids, vert_edge_ids, vert_subcell_ids;
  PetscScalar *vert_coord_v, *vert_icell_ids_v, *vert_edge_ids_v,
              *vert_subcell_ids_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*dim, &vert_coord);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*4, &vert_icell_ids);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*4, &vert_edge_ids);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*4, &vert_subcell_ids);
  CHKERRQ(ierr);

  ierr = VecGetArray(vert_coord, &vert_coord_v); CHKERRQ(ierr);
  ierr = VecGetArray(vert_icell_ids, &vert_icell_ids_v); CHKERRQ(ierr);
  ierr = VecGetArray(vert_edge_ids, &vert_edge_ids_v); CHKERRQ(ierr);
  ierr = VecGetArray(vert_subcell_ids, &vert_subcell_ids_v); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    ierr = TDyVertex_GetCoordinate(vertices, ivertex, dim, &vert_coord_v[ivertex*dim]); CHKERRQ(ierr);
    PetscInt offsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt offsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt offsetEdge    = vertices->edge_offset[ivertex];
    for (i=0; i<4; i++) {
      vert_icell_ids_v[ivertex*4 + i]   = vertices->internal_cell_ids[offsetCell+i];
      vert_edge_ids_v[ivertex*4 + i]    = vertices->edge_ids[offsetSubcell+i];
      vert_subcell_ids_v[ivertex*4 + i] = vertices->subcell_ids[offsetEdge+i];
    }
  }

  ierr = VecRestoreArray(vert_coord, &vert_coord_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(vert_icell_ids, &vert_icell_ids_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(vert_edge_ids, &vert_edge_ids_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(vert_subcell_ids, &vert_subcell_ids_v); CHKERRQ(ierr);

  ierr = TDySavePetscVecAsBinary(vert_coord, "vert_coord.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(vert_icell_ids, "vert_icell_ids.bin");
  CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(vert_edge_ids, "vert_edge_ids.bin"); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsBinary(vert_subcell_ids, "vert_subcell_ids.bin");
  CHKERRQ(ierr);

  ierr = VecDestroy(&vert_coord); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputTransmissibilityMatrix2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscInt       dim;
  PetscInt       ivertex, i, j;
  PetscErrorCode ierr;

  dm = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;

  Vec tmat;
  PetscScalar *tmat_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*5*5, &tmat);
  CHKERRQ(ierr);

  ierr = VecGetArray(tmat, &tmat_v); CHKERRQ(ierr);

  PetscInt count;

  count = 0;
  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    for (i=0; i<5; i++) {
      for (j=0; j<5; j++) {
        tmat_v[count] = tdy->Trans[ivertex][i][j];
        count++;
      }
    }
  }

  ierr = VecRestoreArray(tmat, &tmat_v); CHKERRQ(ierr);

  ierr = TDySavePetscVecAsBinary(tmat, "trans_matrix.bin"); CHKERRQ(ierr);

  ierr = VecDestroy(&tmat); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode Output2DMesh(TDy tdy) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = OutputCells2DMesh(tdy); CHKERRQ(ierr);
  ierr = OutputVertices2DMesh(tdy); CHKERRQ(ierr);
  ierr = OutputEdges2DMesh(tdy); CHKERRQ(ierr);
  ierr = OutputTransmissibilityMatrix2DMesh(tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyOutputMesh(TDy tdy) {

  PetscErrorCode ierr;
  PetscInt dim;

  PetscFunctionBegin;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);
  switch(dim) {
    case 2:
      ierr = Output2DMesh(tdy); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Output of mesh only supported for 2D meshes");
      break;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalCells(TDy tdy) {

  PetscErrorCode ierr;
  DM             dm;
  Vec            junkVec;
  PetscInt       junkInt;
  PetscInt       gref;
  PetscInt       cStart, cEnd, c;
  TDy_cell       *cells;

  PetscFunctionBegin;

  dm = tdy->dm;
  cells = &tdy->mesh->cells;

  // Once needs to atleast haved called a DMCreateXYZ() before using DMPlexGetPointGlobal()
  ierr = DMCreateGlobalVector(dm, &junkVec); CHKERRQ(ierr);
  ierr = VecDestroy(&junkVec); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      cells->is_local[c] = PETSC_TRUE;
      cells->global_id[c] = gref;
    } else {
      cells->is_local[c] = PETSC_FALSE;
      cells->global_id[c] = -gref-1;
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalVertices(TDy tdy) {

  PetscInt       ivertex, icell, c;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  PetscInt       vStart, vEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    for (c=0; c<vertices->num_internal_cells[ivertex]; c++) {
      PetscInt offsetCell = vertices->internal_cell_offset[ivertex];
      icell = vertices->internal_cell_ids[offsetCell + c];
      if (cells->is_local[icell]) vertices->is_local[ivertex] = PETSC_TRUE;
    }

  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalEdges(TDy tdy) {

  PetscInt iedge, icell_1, icell_2;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_edge *edges;
  PetscInt       eStart, eEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = &mesh->cells;
  edges = &mesh->edges;

  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd); CHKERRQ(ierr);

  for (iedge=0; iedge<mesh->num_edges; iedge++) {

    PetscInt offsetCell = edges->cell_offset[iedge];

    if (!edges->is_internal[iedge]) { // Is it a boundary edge?

      // Determine the cell ID for the boundary edge
      if (edges->cell_ids[offsetCell + 0] != -1) icell_1 = edges->cell_ids[offsetCell + 0];
      else                                icell_1 = edges->cell_ids[offsetCell + 1];

      // Is the cell locally owned?
      if (cells->is_local[icell_1]) edges->is_local[iedge] = PETSC_TRUE;

    } else { // An internal edge

      // Save the two cell ID
      icell_1 = edges->cell_ids[offsetCell + 0];
      icell_2 = edges->cell_ids[offsetCell + 1];

      if (cells->is_local[icell_1] && cells->is_local[icell_2]) { // Are both cells locally owned?

        edges->is_local[iedge] = PETSC_TRUE;

      } else if (cells->is_local[icell_1] && !cells->is_local[icell_2]) { // Is icell_1 locally owned?

        // Is the global ID of icell_1 lower than global ID of icell_2?
        if (cells->global_id[icell_1] < cells->global_id[icell_2]) edges->is_local[iedge] = PETSC_TRUE;

      } else if (!cells->is_local[icell_1] && cells->is_local[icell_2]) { // Is icell_2 locally owned

        // Is the global ID of icell_2 lower than global ID of icell_1?
        if (cells->global_id[icell_2] < cells->global_id[icell_1]) edges->is_local[iedge] = PETSC_TRUE;

      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalFaces(TDy tdy) {

  PetscInt iface, icell_1, icell_2;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  PetscInt       fStart, fEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = &mesh->cells;
  faces = &mesh->faces;

  mesh->num_boundary_faces = 0;

  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    PetscInt fOffsetCell = faces->cell_offset[iface];

    if (!faces->is_internal[iface]) { // Is it a boundary face?

      mesh->num_boundary_faces++;

      // Determine the cell ID for the boundary edge
      if (faces->cell_ids[fOffsetCell + 0] >= 0) {
        icell_1 = faces->cell_ids[fOffsetCell + 0];
        faces->cell_ids[fOffsetCell + 1] = -mesh->num_boundary_faces;
      } else {
        icell_1 = faces->cell_ids[fOffsetCell + 1];
        faces->cell_ids[0] = -mesh->num_boundary_faces;
      }

      // Is the cell locally owned?
      if (cells->is_local[icell_1]) faces->is_local[iface] = PETSC_TRUE;

    } else { // An internal face

      // Save the two cell ID
      icell_1 = faces->cell_ids[fOffsetCell + 0];
      icell_2 = faces->cell_ids[fOffsetCell + 1];

      if (cells->is_local[icell_1] && cells->is_local[icell_2]) { // Are both cells locally owned?

        faces->is_local[iface] = PETSC_TRUE;

      } else if (cells->is_local[icell_1] && !cells->is_local[icell_2]) { // Is icell_1 locally owned?

        // Is the global ID of icell_1 lower than global ID of icell_2?
        if (cells->global_id[icell_1] < cells->global_id[icell_2]) faces->is_local[iface] = PETSC_TRUE;

      } else if (!cells->is_local[icell_1] && cells->is_local[icell_2]) { // Is icell_2 locally owned

        // Is the global ID of icell_2 lower than global ID of icell_1?
        if (cells->global_id[icell_2] < cells->global_id[icell_1]) faces->is_local[iface] = PETSC_TRUE;

      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyBuildMesh(TDy tdy) {

  PetscInt dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;


  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = SaveMeshGeometricAttributes(tdy); CHKERRQ(ierr);
    ierr = SaveMeshConnectivityInfo(   tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAVertex2DMesh(tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor2DMesh     (  tdy->dm, tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAEdge2DMesh(  tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalCells(tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalVertices(tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalEdges(tdy); CHKERRQ(ierr);
    break;

  case 3:
    ierr = SaveMeshGeometricAttributes(tdy); CHKERRQ(ierr);
    ierr = SaveMeshConnectivityInfo(   tdy); CHKERRQ(ierr);
    ierr = UpdateFaceOrderAroundAVertex3DMesh(tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAFace3DMesh(  tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalCells(tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalVertices(tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalEdges(tdy); CHKERRQ(ierr);
    ierr = IdentifyLocalFaces(tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor3DMesh     (tdy); CHKERRQ(ierr);
    break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in TDyBuildMesh");
    break;
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDy_cell *cells, PetscInt cell_id, TDy_vertex *vertices, PetscInt ivertex, TDy_subcell *subcells, PetscInt *subcell_id) {

  PetscFunctionBegin;

  PetscInt i, isubcell = -1;

  for (i=0; i<vertices->num_internal_cells[ivertex];i++){
    PetscInt offsetCell = vertices->internal_cell_offset[ivertex];
    PetscInt offsetSubcell = vertices->subcell_offset[ivertex];
    if (vertices->internal_cell_ids[offsetCell+i] == cells->id[cell_id]) {
      isubcell = vertices->subcell_ids[offsetSubcell+i];
      break;
    }
  }

  if (isubcell == -1) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a subcell of a given cell that includes the given vertex");
  }
  
  //*subcell = &subcells[cells->id[cell_id]*cells->num_subcells[cell_id]+isubcell];
  *subcell_id = cells->id[cell_id]*cells->num_subcells[cell_id]+isubcell;
  

  PetscFunctionReturn(0);
}
