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
PetscErrorCode AllocateMemoryForASubcell(
  TDy_subcell    *subcell,
  TDySubcellType subcell_type) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscInt num_nu_vectors = GetNumOfNuVectorsForSubcellType(subcell_type);
  PetscInt num_vertices   = GetNumVerticesForSubcellType(subcell_type);
  PetscInt num_faces      = GetNumFacesForSubcellType(subcell_type);

  subcell->type           = subcell_type;
  subcell->num_nu_vectors = num_nu_vectors;
  subcell->num_vertices   = num_vertices;
  subcell->num_faces      = num_faces;

  ierr = TDyAllocate_TDyVector_1D(num_nu_vectors, &subcell->nu_vector); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_nu_vectors, &subcell->variable_continuity_coordinates); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_vertices, &subcell->vertices_cordinates); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_nu_vectors, &subcell->face_centroid); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&subcell->face_ids,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&subcell->face_area,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcell->is_face_up,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&subcell->face_unknown_idx,num_faces); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForACell(
  TDy_cell       *cell,
  TDyCellType cell_type
) {

  PetscFunctionBegin;

  PetscErrorCode ierr;
  PetscInt       isubcell;
  PetscInt       num_subcells;
  TDy_subcell    *subcells;
  PetscInt       num_vertices;
  PetscInt       num_edges;
  PetscInt       num_neighbors;
  PetscInt       num_faces;
  TDySubcellType subcell_type;

  num_vertices  = GetNumVerticesForCellType(cell_type);
  num_edges     = GetNumEdgesForCellType(cell_type);
  num_neighbors = GetNumNeighborsForCellType(cell_type);
  num_faces     = GetNumFacesForCellType(cell_type);
  subcell_type  = GetSubcellTypeForCellType(cell_type);
  num_subcells  = GetNumSubcellsForSubcellType(subcell_type);

  cell->is_local      = PETSC_FALSE;
  cell->num_vertices  = num_vertices;
  cell->num_edges     = num_edges;
  cell->num_neighbors = num_neighbors;
  cell->num_faces     = 0;

  ierr = TDyAllocate_IntegerArray_1D(&cell->vertex_ids  ,num_vertices ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cell->edge_ids    ,num_edges    ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cell->neighbor_ids,num_neighbors); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&cell->face_ids    ,num_faces    ); CHKERRQ(ierr);

  TDyInitialize_IntegerArray_1D(cell->face_ids, num_faces, 0);

  cell->num_subcells = num_subcells;

  ierr = TDyAllocate_TDySubcell_1D(num_subcells, &cell->subcells); CHKERRQ(ierr);

  subcells = cell->subcells;
  for (isubcell=0; isubcell<num_subcells; isubcell++) {

    subcells[isubcell].id      = isubcell;
    subcells[isubcell].cell_id = cell->id;

    ierr = AllocateMemoryForASubcell(&subcells[isubcell], subcell_type);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForCells(
  PetscInt    num_cells,
  TDyCellType cell_type,
  TDy_cell    *cells) {

  PetscFunctionBegin;

  PetscInt icell;
  PetscErrorCode ierr;

  /* allocate memory for cells within the mesh*/
  for (icell=0; icell<num_cells; icell++) {
    cells[icell].id = icell;
    ierr = AllocateMemoryForACell(&cells[icell], cell_type); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForAVertex(
  TDy_vertex     *vertex,
  TDyCellType    cell_type) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscInt num_internal_cells = GetNumOfCellsSharingAVertexForCellType(cell_type);
  PetscInt num_edges          = GetNumEdgesForCellType(cell_type);
  PetscInt num_faces          = GetNumFacesSharedByVertexForCellType(cell_type);

  vertex->is_local           = PETSC_FALSE;
  vertex->num_internal_cells = 0;
  vertex->num_edges          = num_edges;
  vertex->num_faces          = 0;
  vertex->num_boundary_cells = 0;

  ierr = TDyAllocate_IntegerArray_1D(&vertex->edge_ids         ,num_edges ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertex->face_ids         ,num_faces ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertex->internal_cell_ids,num_internal_cells ); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&vertex->subcell_ids      ,num_internal_cells ); CHKERRQ(ierr);

  TDyInitialize_IntegerArray_1D(vertex->face_ids, num_faces, 0);

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

  /* allocate memory for vertices within the mesh*/
  for  (ivertex=0; ivertex<num_vertices; ivertex++) {
    vertices[ivertex].id = ivertex;
    ierr = AllocateMemoryForAVertex(&vertices[ivertex], cell_type); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForAEdge(
  TDy_edge *edge,
  TDyCellType cell_type) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt num_cells = GetNumCellsPerEdgeForCellType(cell_type);

  edge->num_cells = num_cells;
  edge->is_local = PETSC_FALSE;

  ierr = TDyAllocate_IntegerArray_1D(&edge->cell_ids,num_cells); CHKERRQ(ierr);

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

  /* allocate memory for edges within the mesh*/
  for (iedge=0; iedge<num_edges; iedge++) {
    edges[iedge].id = iedge;
    ierr = AllocateMemoryForAEdge(&edges[iedge], cell_type); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForAFace(
  TDy_face *face,
  TDyCellType cell_type) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  face->num_cells    = GetNumCellsPerFaceForCellType(cell_type);
  face->num_edges    = GetNumOfEdgesFormingAFaceForCellType(cell_type);
  face->num_vertices = GetNumOfVerticesFormingAFaceForCellType(cell_type);

  face->is_local    = PETSC_FALSE;
  face->is_internal = PETSC_FALSE;

  ierr = TDyAllocate_IntegerArray_1D(&face->cell_ids,face->num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&face->edge_ids,face->num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&face->vertex_ids,face->num_vertices); CHKERRQ(ierr);

  face->num_cells = 0;
  face->num_vertices = 0;

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

  /* allocate memory for edges within the mesh*/
  for (iface=0; iface<num_faces; iface++) {
    faces[iface].id = iface;
    ierr = AllocateMemoryForAFace(&faces[iface], cell_type); CHKERRQ(ierr);
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
  PetscInt fNum;
  PetscInt dim;
  TDyCellType cell_type;

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

  ierr = TDyAllocate_TDyCell_1D(cNum, &mesh->cells); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyFace_1D(fNum, &mesh->faces); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyEdge_1D(eNum, &mesh->edges); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVertex_1D(vNum, &mesh->vertices); CHKERRQ(ierr);

  ierr = AllocateMemoryForCells(cNum, cell_type, mesh->cells); CHKERRQ(ierr);
  ierr = AllocateMemoryForVertices(vNum, cell_type, mesh->vertices); CHKERRQ(ierr);

  ierr = AllocateMemoryForEdges(eNum, cell_type, mesh->edges); CHKERRQ(ierr);

  ierr = AllocateMemoryForFaces(fNum, cell_type, mesh->faces); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AreFacesNeighbors(TDy_face *face_1, TDy_face *face_2) {

  PetscBool are_neighbors;
  PetscInt iedge_1, iedge_2;

  are_neighbors = PETSC_FALSE;

  for (iedge_1=0; iedge_1<face_1->num_edges; iedge_1++) {
    for (iedge_2=0; iedge_2<face_2->num_edges; iedge_2++) {
      if (face_1->edge_ids[iedge_1] == face_2->edge_ids[iedge_2]) {
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
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;
  faces    = mesh->faces;

  for (ielement=pStart; ielement<pEnd; ielement++) {

    if (IsClosureWithinBounds(ielement, vStart, vEnd)) { // is the element a vertex?
      ivertex = ielement - vStart;
      for (d=0; d<dim; d++) {
        vertices[ivertex].coordinate.X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, eStart,
                                     eEnd)) { // is the element an edge?
      iedge = ielement - eStart;
      for (d=0; d<dim; d++) {
        edges[iedge].centroid.X[d] = tdy->X[ielement*dim + d];
        edges[iedge].normal.V[d]   = tdy->N[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, cStart,
                                     cEnd)) { // is the elment a cell?
      icell = ielement - cStart;
      for (d=0; d<dim; d++) {
        cells[icell].centroid.X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, fStart,
                                     fEnd)) { // is the elment a face?
      iface = ielement - fStart;
      for (d=0; d<dim; d++) {
        faces[iface].centroid.X[d] = tdy->X[ielement*dim + d];
      }
      faces[iface].area = tdy->V[ielement];
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode SaveMeshConnectivityInfo(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  TDy_face       *faces, *face;
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
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;
  faces    = mesh->faces;

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
        cells[icell].vertex_ids[c2vCount] = ivertex ;
        for (j=0; j<nverts_per_cell; j++) {
          if (vertices[ivertex].internal_cell_ids[j] == -1) {
            vertices[ivertex].num_internal_cells++;
            vertices[ivertex].internal_cell_ids[j] = icell;
            vertices[ivertex].subcell_ids[j]       = c2vCount;
            break;
          }
        }
        c2vCount++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], eStart,
                                       eEnd)) { /* Is the closure an edge? */
        PetscInt iedge = tdy->closure[icell][i] - eStart;
        cells[icell].edge_ids[c2eCount] = iedge;
        for (j=0; j<2; j++) {
          if (edges[iedge].cell_ids[j] == -1) {
            edges[iedge].cell_ids[j] = icell;
            break;
          }
        }
        c2eCount++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], fStart,
                                       fEnd)) { /* Is the closure a face? */
        PetscInt iface = tdy->closure[icell][i] - fStart;
        cells[icell].face_ids[c2fCount] = iface;
        for (j=0; j<2; j++) {
          if (faces[iface].cell_ids[j] == -1) {
            faces[iface].cell_ids[j] = icell;
            faces[iface].num_cells++;
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

    if (supportSize == 1) edges[iedge].is_internal = PETSC_FALSE;
    else                  edges[iedge].is_internal = PETSC_TRUE;

    edges[iedge].vertex_ids[0] = cone[0]-vStart;
    edges[iedge].vertex_ids[1] = cone[1]-vStart;

    ierr = TDyVertex_GetCoordinate(&vertices[edges[iedge].vertex_ids[0]], dim, &v_1[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(&vertices[edges[iedge].vertex_ids[1]], dim, &v_2[0]); CHKERRQ(ierr);

    for (d=0; d<dim; d++) {
      edges[iedge].centroid.X[d] = (v_1[d] + v_2[d])/2.0;
    }

    ierr = TDyComputeLength(v_1, v_2, dim, &(edges[iedge].length)); CHKERRQ(ierr);
  }

  // vertex--to--edge
  for (v=vStart; v<vEnd; v++) {
    ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, v, &supportSize); CHKERRQ(ierr);
    PetscInt ivertex = v - vStart;
    vertices[ivertex].num_edges = supportSize;
    for (s=0; s<supportSize; s++) {
      PetscInt iedge = support[s] - eStart;
      vertices[ivertex].edge_ids[s] = iedge;
      if (!edges[iedge].is_internal) vertices[ivertex].num_boundary_cells++;
    }
  }

  PetscInt f;
  TDy_vertex *vertex;
  for (f=fStart; f<fEnd; f++){
    PetscInt iface = f-fStart;
    face = &faces[iface];

    // face--to--edge
    ierr = DMPlexGetConeSize(dm, f, &coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, f, &cone); CHKERRQ(ierr);

    PetscInt c;
    for (c=0;c<coneSize;c++) {
      faces[iface].edge_ids[c] = cone[c]-eStart;
    }

    // face--to-vertex
    PetscInt i;
    for (i=0; i<tdy->closureSize[f]*2; i+=2)  {
      if (IsClosureWithinBounds(tdy->closure[f][i],vStart,vEnd)) {
        face->vertex_ids[face->num_vertices] = tdy->closure[f][i]-vStart;
        face->num_vertices++;

        PetscBool found = PETSC_FALSE;
        PetscInt ivertex = tdy->closure[f][i]-vStart;
        vertex = &vertices[ivertex];
        PetscInt ii;
        for (ii=0; ii<vertex->num_faces; ii++) {
          if (vertex->face_ids[ii] == iface) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          vertex->face_ids[vertex->num_faces] = iface;
          vertex->num_faces++;
        }
      }
    }

    // face--to--cell
    ierr = DMPlexGetSupportSize(dm, f, &supportSize); CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, f, &support); CHKERRQ(ierr);

    if (supportSize == 2) faces[iface].is_internal = PETSC_TRUE;
    else                  faces[iface].is_internal = PETSC_FALSE;

    for (s=0; s<supportSize; s++) {
      icell = support[s] - cStart;
      cell = &cells[icell];
        PetscBool found = PETSC_FALSE;
        PetscInt ii;
        for (ii=0; ii<cell->num_faces; ii++) {
          if (cell->face_ids[ii] == f-fStart) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          cell->face_ids[cell->num_faces] = f-fStart;
          cell->num_faces++;
          found = PETSC_TRUE;
        }
    }

    // If it is a boundary face, increment the number of boundary
    // cells by 1 for all vertices that form the face
    if (!faces[iface].is_internal) {
      for (v=0; v<face->num_vertices; v++) {
        vertex = &vertices[face->vertex_ids[v]];
        vertex->num_boundary_cells++;
      }
    }

  }

  // allocate memory to save ids of faces on the boundary
  if (dim == 3) {
  for (v=vStart; v<vEnd; v++) {
    vertex = &vertices[v-vStart];

    PetscInt nflux_in=0;
    switch (vertex->num_internal_cells) {
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
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Unsupported number of internal cells.");
        break;
    }
    ierr = TDyAllocate_IntegerArray_1D(&vertex->boundary_face_ids,vertex->num_boundary_cells); CHKERRQ(ierr);
    ierr = TDyAllocate_IntegerArray_1D(&vertex->trans_row_face_ids,nflux_in+vertex->num_boundary_cells); CHKERRQ(ierr); ;
  }
  }

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
  TDy_vertex     *vertex;
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
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertex   = &(mesh->vertices[ivertex]);

  ncells = vertex->num_internal_cells;
  nedges = vertex->num_edges;
  count  = 0;

  // compute angle to all cell centroids w.r.t. the shared vertix
  for (i=0; i<ncells; i++) {
    icell = vertex->internal_cell_ids[i];

    x = cells[icell].centroid.X[0] - vertex->coordinate.X[0];
    y = cells[icell].centroid.X[1] - vertex->coordinate.X[1];

    ids[count]              = icell;
    idx[count]              = count;
    subcell_ids[count]      = vertex->subcell_ids[i];
    is_cell[count]          = PETSC_TRUE;
    is_internal_edge[count] = PETSC_FALSE;

    ierr = ComputeTheta(x, y, &theta[count]);

    count++;
  }

  // compute angle to face centroids w.r.t. the shared vertix
  boundary_edge_present = PETSC_FALSE;

  for (i=0; i<nedges; i++) {
    iedge = vertex->edge_ids[i];
    x = edges[iedge].centroid.X[0] - vertex->coordinate.X[0];
    y = edges[iedge].centroid.X[1] - vertex->coordinate.X[1];

    ids[count]              = iedge;
    idx[count]              = count;
    subcell_ids[count]      = -1;
    is_cell[count]          = PETSC_FALSE;
    is_internal_edge[count] = edges[iedge].is_internal;

    if (!edges[iedge].is_internal) boundary_edge_present = PETSC_TRUE;

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
    vertex->internal_cell_ids[i] = tmp_cell_ids[i];
    vertex->subcell_ids[i]       = tmp_subcell_ids[i];
  }

  // save information about sorted edge ids
  for (i=0; i<tmp_nedges; i++) {
    vertex->edge_ids[i] = tmp_edge_ids[i];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAVertex2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscInt       vStart, vEnd;
  TDy_vertex     *vertices, *vertex;
  PetscInt       ivertex;
  PetscInt       edge_id_1, edge_id_2;
  TDy_edge       *edges;
  PetscReal      x,y, theta_1, theta_2;
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<vEnd-vStart; ivertex++) {

    if (vertices[ivertex].num_internal_cells > 1) {
      ierr = UpdateCellOrientationAroundAVertex(tdy, ivertex); CHKERRQ(ierr);
    } else {

      vertex = &vertices[ivertex];
      edge_id_1 = vertex->edge_ids[0];
      edge_id_2 = vertex->edge_ids[1];

      x = edges[edge_id_1].centroid.X[0] - vertex->coordinate.X[0];
      y = edges[edge_id_1].centroid.X[1] - vertex->coordinate.X[1];
      ierr = ComputeTheta(x, y, &theta_1);

      x = edges[edge_id_2].centroid.X[0] - vertex->coordinate.X[0];
      y = edges[edge_id_2].centroid.X[1] - vertex->coordinate.X[1];
      ierr = ComputeTheta(x, y, &theta_2);

      if (theta_1 < theta_2) {
        if (theta_2 - theta_1 <= PETSC_PI) {
          vertex->edge_ids[0] = edge_id_2;
          vertex->edge_ids[1] = edge_id_1;
        } else {
          vertex->edge_ids[0] = edge_id_1;
          vertex->edge_ids[1] = edge_id_2;
        }
      } else {
        if (theta_1 - theta_2 <= PETSC_PI) {
          vertex->edge_ids[0] = edge_id_1;
          vertex->edge_ids[1] = edge_id_2;
        } else {
          vertex->edge_ids[0] = edge_id_2;
          vertex->edge_ids[1] = edge_id_1;
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
  TDy_vertex     *vertices, *vertex;
  PetscInt       ivertex;
  TDy_face       *faces;
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  faces    = mesh->faces;
  vertices = mesh->vertices;

  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<vEnd-vStart; ivertex++) {
    
    vertex = &vertices[ivertex];

    PetscInt face_ids_sorted[vertex->num_faces];
    PetscInt count=0, iface;

    // First find all internal faces (i.e. face shared by two cells)
    for (iface=0;iface<vertex->num_faces;iface++) {
      PetscInt face_id = vertex->face_ids[iface];
      if (faces[face_id].num_cells==2) {
        face_ids_sorted[count] = face_id;
        count++;
      }
    }
    
    // Now find all boundary faces (i.e. face shared by a single cell)
    for (iface=0;iface<vertex->num_faces;iface++) {
      PetscInt face_id = vertex->face_ids[iface];
      if (faces[face_id].num_cells==1) {
        face_ids_sorted[count] = face_id;
        count++;
      }
    }

    // Save the sorted faces
    for (iface=0;iface<vertex->num_faces;iface++) {
      vertex->face_ids[iface] = face_ids_sorted[iface];
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
  TDy_cell       *cells, *cell_from, *cell_to;
  TDy_edge       *edges, *edge;
  PetscErrorCode ierr;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = mesh->cells;
  edges = mesh->edges;

  ierr = DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd); CHKERRQ(ierr);

  for (iedge=0; iedge<eEnd-eStart; iedge++) {
    edge = &(edges[iedge]);
    if (edge->is_internal) {
      cell_from = &cells[edge->cell_ids[0]];
      cell_to   = &cells[edge->cell_ids[1]];

      dot_product = (cell_to->centroid.X[0] - cell_from->centroid.X[0]) *
                    edge->normal.V[0] +
                    (cell_to->centroid.X[1] - cell_from->centroid.X[1]) * edge->normal.V[1];
      if (dot_product < 0.0) {
        edge->cell_ids[0] = cell_to->id;
        edge->cell_ids[1] = cell_from->id;
      }
    } else {
      cell_from = &cells[edge->cell_ids[0]];

      dot_product = (edge->centroid.X[0] - cell_from->centroid.X[0]) *
                    edge->normal.V[0] +
                    (edge->centroid.X[1] - cell_from->centroid.X[1]) * edge->normal.V[1];
      if (dot_product < 0.0) {
        edge->cell_ids[0] = -1;
        edge->cell_ids[1] = cell_from->id;
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
PetscErrorCode TDySubCell_GetIthNuVector(TDy_subcell *subcell, PetscInt i, PetscInt dim, PetscReal *nu_vec) {
  PetscFunctionBegin;
  PetscInt d;

  if (i>=subcell->num_nu_vectors) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th nu_vec exceeds max nu_vecs");
  }
  
  for (d=0; d<dim; d++) nu_vec[d] = subcell->nu_vector[i].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetIthFaceCentroid(TDy_subcell *subcell, PetscInt i, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;

  if (i>=subcell->num_faces) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th face centroid exceeds max num_faces");
  }

  for (d=0; d<dim; d++) centroid[d] = subcell->face_centroid[i].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDy_subcell* subcell, PetscInt face_id, PetscInt *face_idx) {

  PetscFunctionBegin;
  PetscInt i;
  *face_idx = -1;

  for (i=0; i<subcell->num_faces;i++) {
    if (subcell->face_ids[i] == face_id) {
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
PetscErrorCode TDyEdge_GetCentroid(TDy_edge *edge, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = edge->centroid.X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyEdge_GetNormal(TDy_edge *edge, PetscInt dim, PetscReal *normal) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) normal[d] = edge->normal.V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFace_GetCentroid(TDy_face *face, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = face->centroid.X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFace_GetNormal(TDy_face *face, PetscInt dim, PetscReal *normal) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) normal[d] = face->normal.V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyVertex_GetCoordinate(TDy_vertex *vertex, PetscInt dim, PetscReal *coor) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) coor[d] = vertex->coordinate.X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyCell_GetCentroid(TDy_cell *cell, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = cell->centroid.X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode SetupSubcellsFor2DMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges, *edge_up, *edge_dn;
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
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  alpha = 1.0;

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<cEnd-cStart; icell++) {

    // set pointer to cell
    cell = &cells[icell];

    // save cell centroid
    ierr = TDyCell_GetCentroid(cell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cell->num_subcells;

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      vertex  = &vertices[cell->vertex_ids[isubcell]];
      subcell = &cell->subcells[isubcell];

      // save coorindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertex, dim, &v_c[0]); CHKERRQ(ierr);

      // determine ids of up & down edges
      e_idx_up = cells[icell].edge_ids[isubcell];
      if (isubcell == 0) e_idx_dn = cells[icell].edge_ids[num_subcells-1];
      else               e_idx_dn = cells[icell].edge_ids[isubcell    -1];

      // set points to up/down edges
      edge_up = &edges[e_idx_up];
      edge_dn = &edges[e_idx_dn];

      // save centroids of up/down edges
      ierr = TDyEdge_GetCentroid(edge_up, dim, &e_cen_up[0]); CHKERRQ(ierr);
      ierr = TDyEdge_GetCentroid(edge_dn, dim, &e_cen_dn[0]); CHKERRQ(ierr);

      // compute continuity point
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_up, alpha, dim, cp_up);
      CHKERRQ(ierr);
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_dn, alpha, dim, cp_dn);
      CHKERRQ(ierr);

      // save continuity point
      for (d=0; d<dim; d++) {
        subcell->variable_continuity_coordinates[0].X[d] = cp_up[d];
        subcell->variable_continuity_coordinates[1].X[d] = cp_dn[d];
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
        subcell->nu_vector[0].V[d] = nu_vec_up[d]*len_up;
        subcell->nu_vector[1].V[d] = nu_vec_dn[d]*len_dn;
      }

      PetscReal area;
      ierr = ComputeAreaOf2DTriangle(cp_up, cell_cen, cp_dn, &area);
      subcell->T = 2.0*area;

    }
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cell->volume), &centroid,
                                        &normal[0]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode FindNeighboringVerticesOfAFace(TDy_face *face, PetscInt vertex_id,
                                              PetscInt neighboring_vertex_ids[2]) {

  PetscFunctionBegin;

  PetscInt vertex_ids[face->num_vertices];
  PetscInt count = 0, ivertex, start = -1;

  // Find the first occurance of "vertex_id" within list of vertices forming the face
  for (ivertex=0; ivertex<face->num_vertices; ivertex++) {
    if (face->vertex_ids[ivertex] == vertex_id) {
      start = ivertex;
      break;
    }
  }

  if (start == -1) {
    printf("Did not find vertex id = %d within face id = %d\n",vertex_id,face->id);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Stopping in FindNeighboringVerticesOfAFace");
  }

  // Save vertex indices starting from the first occurance of "vertex_id" to the last vertex
  // from the list
  for (ivertex=start; ivertex<face->num_vertices; ivertex++){
    vertex_ids[count] = face->vertex_ids[ivertex];
    count++;
  }

  // Now, save vertex indices starting from 0 to the first occurance of "vertex_id"
  // from the list
  for (ivertex=0; ivertex<start; ivertex++){
    vertex_ids[count] = face->vertex_ids[ivertex];
    count++;
  }

  // The vertex_ids[1] and vertex_ids[end] are neighbors of "vertex_id"
  neighboring_vertex_ids[0] = vertex_ids[1];
  neighboring_vertex_ids[1] = vertex_ids[face->num_vertices-1];

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode FindFaceIDsOfACellCommonToAVertex(TDy_cell *cell, TDy_face *faces,
                                                 TDy_vertex *vertex, PetscInt f_idx[3],
                                                 PetscInt *num_shared_faces) {
  
  PetscFunctionBegin;
  
  PetscInt iface;
  PetscInt cell_id = cell->id;
  
  *num_shared_faces = 0;
  
  // Find the faces of "cell_id" that are shared by the vertex
  for (iface=0; iface<vertex->num_faces; iface++){
    
    PetscInt face_id = vertex->face_ids[iface];
    TDy_face *face = &faces[face_id];
    
    if (face->cell_ids[0] == cell_id || face->cell_ids[1] == cell_id){
      
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

PetscErrorCode UpdateFaceOrientationAroundAVertex(TDy_cell *cell, TDy_face *faces,
                                                  TDy_vertex *vertex, PetscInt dim,
                                                  PetscInt f_idx[3]) {
  
  PetscFunctionBegin;
  
  PetscInt d;
  PetscErrorCode ierr;

  PetscReal a[3],b[3],c[3],axb[3],dot_prod;
  
  for (d=0; d<dim; d++) {
    a[d] = faces[f_idx[1]].centroid.X[d] - faces[f_idx[0]].centroid.X[d];
    b[d] = faces[f_idx[2]].centroid.X[d] - faces[f_idx[0]].centroid.X[d];
    c[d] = cell->centroid.X[d] - vertex->coordinate.X[d];
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
PetscErrorCode SetupCell2CellConnectivity(TDy_vertex *vertex, TDy_cell *cells, TDy_face *faces, PetscInt **cell2cell_conn) {

  /*
   
    cell2cell_conn[i][j]:  0 if i-th and j-th cell are unconnected
                        :  1 if i-th and j-th cell are connected
                        : -1 if i-th and j-th are connected but j-th is a face (note: i-th could itself be a face)

    dimensions: (0:ncells-1)               corresponds to cells, while
                (ncells:ncells+ncells_bnd) corresponds to faces.
   
*/

  PetscFunctionBegin;

  PetscInt icell, isubcell, iface, ncells, cell_id, ncells_bnd, bnd_count;

  ncells = vertex->num_internal_cells;
  ncells_bnd = vertex->num_boundary_cells;
  bnd_count = 0;

  for (icell=0; icell<ncells; icell++) {

    TDy_cell    *cell;
    TDy_subcell *subcell;

    // Determine the cell and subcell id
    cell_id  = vertex->internal_cell_ids[icell];
    isubcell = vertex->subcell_ids[icell];

    // Get access to the cell and subcell
    cell    = &cells[cell_id];
    subcell = &cell->subcells[isubcell];

    // Loop over all faces of the subcell
    for (iface=0;iface<subcell->num_faces;iface++) {

      TDy_face *face = &faces[subcell->face_ids[iface]];

      // Skip boundary face
      if (face->cell_ids[0] == -1 || face->cell_ids[1] == -1) {
        PetscInt cell_1, cell_2;
        if (face->cell_ids[0] == -1) cell_1 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[1]);
        else                         cell_1 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[0]);
      
        cell_2 = ncells + bnd_count;
        vertex->boundary_face_ids[bnd_count] = face->id;
        bnd_count++;

        // Add 1 to indicate cell_1 and cell_2 are connected
        cell2cell_conn[cell_1][cell_2] = 1;
        cell2cell_conn[cell_2][cell_1] = 1;

      } else {
      // Find the index of cells given by face->cell_ids[0:1] within the cell id list given
      // vertex->internal_cell_ids
      PetscInt cell_1 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[0]);
      PetscInt cell_2 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[1]);

      // Add 1 to indicate cell_1 and cell_2 are connected
      cell2cell_conn[cell_1][cell_2] = 1;
      cell2cell_conn[cell_2][cell_1] = 1;
      }
    }
  }

  PetscInt ii,jj;
  for (ii=0;ii<ncells_bnd-1;ii++) {
    for (jj=ii+1;jj<ncells_bnd;jj++) {
      TDy_face *face_1, *face_2;
      face_1 = &faces[vertex->boundary_face_ids[ii]];
      face_2 = &faces[vertex->boundary_face_ids[jj]];
      if (AreFacesNeighbors(face_1,face_2)) {
        cell2cell_conn[ncells+ii][ncells+jj] = -1;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ExtractCentroidForIthJthTraversalOrder(TDy_vertex *vertex, TDy_face *faces, TDy_cell *cells, PetscInt cell_traversal_ij, PetscReal cen[3]) {

  PetscFunctionBegin;
  PetscInt dim = 3;
  PetscErrorCode ierr;

  if (cell_traversal_ij>=0) { // Is a cell?
    PetscInt cell_id;
    TDy_cell *cell;

    cell_id = vertex->internal_cell_ids[cell_traversal_ij];
    cell = &cells[cell_id];
    ierr = TDyCell_GetCentroid(cell, dim, &cen[0]); CHKERRQ(ierr);

  } else { // is a face
  
    PetscInt face_id, idx;
    TDy_face *face;

    idx = -cell_traversal_ij - vertex->num_internal_cells;
    face_id = vertex->boundary_face_ids[idx];
    face = &faces[face_id];

    ierr = TDyFace_GetCentroid(face, dim, &cen[0]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateIthTraversalOrder(TDy_vertex *vertex, TDy_face *faces, TDy_cell *cells, PetscInt i, PetscBool flip_if_dprod_is_negative, PetscInt **cell_traversal) {

  PetscFunctionBegin;

  PetscInt dim, d, j;
  PetscReal v1[3],v2[3],v3[3],v4[3],c2v_vec[3],normal[3];
  PetscReal dot_product;
  PetscErrorCode ierr;

  dim = 3;

  // Get pointers to the four cells
  j = 0; ierr = ExtractCentroidForIthJthTraversalOrder(vertex, faces, cells, cell_traversal[i][j], v1);
  j = 1; ierr = ExtractCentroidForIthJthTraversalOrder(vertex, faces, cells, cell_traversal[i][j], v2);
  j = 2; ierr = ExtractCentroidForIthJthTraversalOrder(vertex, faces, cells, cell_traversal[i][j], v3);
  j = 3; ierr = ExtractCentroidForIthJthTraversalOrder(vertex, faces, cells, cell_traversal[i][j], v4);

  // Save (x,y,z) of the four cells, and
  // a vector joining centroid of four cells and the vertex (c2v_vec)
  for (d=0; d<dim; d++) {
    c2v_vec[d] = vertex->coordinate.X[d] - (v1[d] + v2[d] + v3[d] + v4[d])/4.0;
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
PetscErrorCode ComputeFirstTraversalOrder(TDy_vertex *vertex, TDy_face *faces, TDy_cell *cells, PetscInt **cell2cell_conn, PetscInt **cell_traversal)
{
  PetscFunctionBegin;

  PetscInt i, j, m, ncells, ncells_bnd;
  PetscErrorCode ierr;

  // Objective to find: c0 --> c2 --> c5 --> c4

  ncells    = vertex->num_internal_cells;
  ncells_bnd= vertex->num_boundary_cells;

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
  ierr = UpdateIthTraversalOrder(vertex, faces, cells, i, flip_if_dprod_is_negative, cell_traversal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeSecondTraversalOrder(TDy_vertex *vertex, TDy_face *faces, TDy_cell *cells, PetscInt **cell2cell_conn, PetscInt **cell_traversal) {

  PetscFunctionBegin;

  PetscInt i, j, k, m, ncells, count, cell_1, cell_2;
  PetscBool found;
  PetscErrorCode ierr;

  // Objective to find: c3 <-- c7 <-- c6 <-- c1

  i = 1;
  ncells = vertex->num_internal_cells;

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
  ierr = UpdateIthTraversalOrder(vertex, faces, cells, i, flip_if_dprod_is_negative, cell_traversal); CHKERRQ(ierr);

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
PetscErrorCode SetupUpwindFacesForSubcell(TDy_vertex *vertex, TDy_cell *cells, TDy_face *faces, PetscInt **cell_up2dw) {

  PetscFunctionBegin;

  PetscInt icell, isubcell, iface, ncells, cell_id, nflux_in;
  PetscInt ii, boundary_cell_count;

  ncells = vertex->num_internal_cells;
  boundary_cell_count = 0;

  switch (vertex->num_internal_cells) {
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

  for (icell=0; icell<ncells; icell++) {

    TDy_cell    *cell;
    TDy_subcell *subcell;

    // Determine the cell and subcell id
    cell_id  = vertex->internal_cell_ids[icell];
    isubcell = vertex->subcell_ids[icell];

    // Get access to the cell and subcell
    cell    = &cells[cell_id];
    subcell = &cell->subcells[isubcell];

    // Loop over all faces of the subcell
    for (iface=0;iface<subcell->num_faces;iface++) {

      TDy_face *face = &faces[subcell->face_ids[iface]];

      // Skip boundary face
      if (face->cell_ids[0] == -1 || face->cell_ids[1] == -1) {
        subcell->face_unknown_idx[iface] = boundary_cell_count+nflux_in;
        for (ii=0; ii<12; ii++) {
          if (cell_up2dw[ii][0] == subcell->face_unknown_idx[iface]){ subcell->is_face_up[iface] = PETSC_FALSE; break;}
          if (cell_up2dw[ii][1] == subcell->face_unknown_idx[iface]){ subcell->is_face_up[iface] = PETSC_TRUE ; break;}
        }
        boundary_cell_count++;
        continue;
      }

      // Find the index of cells given by face->cell_ids[0:1] within the cell id list given
      // vertex->internal_cell_ids
      PetscInt cell_1 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[0]);
      PetscInt cell_2 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[1]);

      for (ii=0; ii<12; ii++) {
        if (cell_up2dw[ii][0] == cell_1 && cell_up2dw[ii][1] == cell_2) {

          subcell->face_unknown_idx[iface] = ii;
          if (cell->id == face->cell_ids[0])  subcell->is_face_up[iface] = PETSC_TRUE;
          else                                subcell->is_face_up[iface] = PETSC_FALSE;
          break;
        } else if (cell_up2dw[ii][0] == cell_2 && cell_up2dw[ii][1] == cell_1) {

          subcell->face_unknown_idx[iface] = ii;
          if (cell->id == face->cell_ids[1]) subcell->is_face_up[iface] = PETSC_TRUE;
          else                               subcell->is_face_up[iface] = PETSC_FALSE;
          break;
        }
      }
    }
  }

  PetscInt nup_bnd_flux=0, ndn_bnd_flux=0;
  PetscInt nflux_bc = 0;

  nflux_bc = vertex->num_boundary_cells/2;

  // Save the face index that corresponds to the flux in transmissibility matrix
  for (icell=0; icell<ncells; icell++) {

    TDy_cell    *cell;
    TDy_subcell *subcell;

    // Determine the cell and subcell id
    cell_id  = vertex->internal_cell_ids[icell];
    isubcell = vertex->subcell_ids[icell];

    // Get access to the cell and subcell
    cell    = &cells[cell_id];
    subcell = &cell->subcells[isubcell];

    // Loop over all faces of the subcell
    for (iface=0; iface<subcell->num_faces; iface++) {
      TDy_face *face = &faces[subcell->face_ids[iface]];

      PetscInt idx_flux = subcell->face_unknown_idx[iface];
      if (face->is_internal) {
        vertex->trans_row_face_ids[idx_flux] = face->id;
      } else {
        if (subcell->is_face_up[iface]) {
          vertex->trans_row_face_ids[nflux_in+nup_bnd_flux] = face->id;
          nup_bnd_flux++;
        } else {
          vertex->trans_row_face_ids[nflux_in+nflux_bc+ndn_bnd_flux] = face->id;
          ndn_bnd_flux++;
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode DetermineUpwindFacesForSubcell(TDy tdy, TDy_vertex *vertex) {

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

  PetscInt ncells,ncells_bnd;
  PetscInt i, j, k;
  PetscInt **cell_traversal;
  PetscInt max_faces = 2;
  PetscInt max_cells_per_face = 4;
  PetscInt **cell2cell_conn;
  PetscInt count;
  PetscInt **cell_up2dw;
  PetscErrorCode ierr;

  cells = tdy->mesh->cells;
  faces = tdy->mesh->faces;

  ncells    = vertex->num_internal_cells;
  ncells_bnd= vertex->num_boundary_cells;

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
  ierr = SetupCell2CellConnectivity(vertex, cells, faces, cell2cell_conn); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_2D(&cell_traversal, max_faces, max_cells_per_face); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_2D(cell_traversal, max_faces, max_cells_per_face, -(ncells+ncells_bnd)); CHKERRQ(ierr);

  // First traversal:
  //   c0 --> c2 --> c5 --> c4, or
  //   c1 --> c6 --> f7 --> f3, or
  //   c3 --> c1 --> c6 --> c7
  ierr = ComputeFirstTraversalOrder(vertex,faces,cells,cell2cell_conn,cell_traversal); CHKERRQ(ierr);

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
    ierr = ComputeSecondTraversalOrder(vertex,faces,cells,cell2cell_conn,cell_traversal); CHKERRQ(ierr);
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
  
  ierr = SetupUpwindFacesForSubcell(vertex,cells,faces,cell_up2dw); CHKERRQ(ierr);

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
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces, *face;
  PetscInt       cStart, cEnd, num_subcells;
  PetscInt       icell, isubcell, ivertex;
  PetscInt       dim, d;
  PetscReal      cell_cen[3], v_c[3];
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  faces    = mesh->faces;
  vertices = mesh->vertices;

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<cEnd-cStart; icell++) {

    // set pointer to cell
    cell = &cells[icell];

    // save cell centroid
    ierr = TDyCell_GetCentroid(cell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cell->num_subcells;

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      vertex  = &vertices[cell->vertex_ids[isubcell]];
      subcell = &cell->subcells[isubcell];

      // save coorindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertex, dim, &v_c[0]); CHKERRQ(ierr);

      PetscInt num_shared_faces;

      // For a given cell, find all face ids that are share a vertex
      ierr = FindFaceIDsOfACellCommonToAVertex(cell, faces, vertex, subcell->face_ids, &num_shared_faces);

      // Update order of faces in f_idx so (face_ids[0], face_ids[1], face_ids[2])
      // form a plane such that normal to plane points toward the cell centroid
      ierr = UpdateFaceOrientationAroundAVertex(cell, faces, vertex, dim, subcell->face_ids);
      
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

        face = &faces[subcell->face_ids[iface]];

        ierr = TDyFace_GetCentroid(face, dim, &face_cen[iface][0]); CHKERRQ(ierr);

        PetscInt neighboring_vertex_ids[2];

        // Find 'n0' and 'n1'
        ierr = FindNeighboringVerticesOfAFace(face,vertex->id,neighboring_vertex_ids);
        
        PetscReal edge0_cen[3], edge1_cen[3];

        for (d=0; d<dim; d++) {
          edge0_cen[d] = (v_c[d] + vertices[neighboring_vertex_ids[0]].coordinate.X[d])/2.0;
          edge1_cen[d] = (v_c[d] + vertices[neighboring_vertex_ids[1]].coordinate.X[d])/2.0;
          subcell->face_centroid[iface].X[d] = (v_c[d] + edge0_cen[d] + face_cen[iface][d] + edge1_cen[d])/4.0;
        }

        // area of face
        ierr = TDyQuadrilateralArea(v_c, edge0_cen, face_cen[iface], edge1_cen, &subcell->face_area[iface]);

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
        TDy_face *face1, *face2;
        face1 = &faces[subcell->face_ids[f1_idx]];
        face2 = &faces[subcell->face_ids[f2_idx]];

        ierr = TDyFace_GetCentroid(face1, dim, &f1[0]); CHKERRQ(ierr);
        ierr = TDyFace_GetCentroid(face2, dim, &f2[0]); CHKERRQ(ierr);

        // Compute (x_{f1_idx } - x_{cell_centroid}) x (x_{f2_idx } - x_{cell_centroid})
        ierr = TDyNormalToTriangle(cell_cen, f1, f2, f_normal);

        // Save the data
        for (d=0; d<dim; d++) subcell->nu_vector[iface].V[d] = f_normal[d];

      }

      PetscReal volume;
      ierr = TDyComputeVolumeOfTetrahedron(cell_cen, face_cen[0], face_cen[1], face_cen[2], &volume); CHKERRQ(ierr);
      subcell->T = volume*6.0;
    }
    PetscReal normal[3], centroid;
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cell->volume), &centroid,
                                        &normal[0]); CHKERRQ(ierr);

  }

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    vertex = &vertices[ivertex];
    if (vertex->num_internal_cells > 1) {
      ierr = DetermineUpwindFacesForSubcell(tdy, vertex );
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateCellOrientationAroundAFace3DMesh(TDy tdy) {

  /*
  
  Ensure the order of cell_ids for a given face is such that:
    Vector from face->cell_ids[0] to face->cell_ids[1] points in the direction
    of the normal vector to the face.
  
  */

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  PetscInt       iface, dim;
  TDy_vertex     *vertices, *vertex;
  TDy_cell       *cells, *cell;
  TDy_face       *faces, *face;
  PetscErrorCode ierr;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = mesh->cells;
  faces = mesh->faces;
  vertices = mesh->vertices;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {
    face = &(faces[iface]);

    PetscReal v1[3], v2[3], v3[3], v4[3], normal[3];
    PetscReal f_cen[3], c_cen[3], f2c[3], dot_prod;
    PetscInt d;

    vertex = &vertices[face->vertex_ids[0]]; ierr = TDyVertex_GetCoordinate(vertex, dim, &v1[0]); CHKERRQ(ierr);
    vertex = &vertices[face->vertex_ids[1]]; ierr = TDyVertex_GetCoordinate(vertex, dim, &v2[0]); CHKERRQ(ierr);
    vertex = &vertices[face->vertex_ids[2]]; ierr = TDyVertex_GetCoordinate(vertex, dim, &v3[0]); CHKERRQ(ierr);
    vertex = &vertices[face->vertex_ids[3]]; ierr = TDyVertex_GetCoordinate(vertex, dim, &v4[0]); CHKERRQ(ierr);

    ierr = TDyFace_GetCentroid(face, dim, &f_cen[0]); CHKERRQ(ierr);

    cell = &cells[face->cell_ids[0]];
    ierr = TDyCell_GetCentroid(cell, dim, &c_cen[0]); CHKERRQ(ierr);

    ierr = TDyNormalToQuadrilateral(v1, v2, v3, v4, normal); CHKERRQ(ierr);
    for (d=0; d<dim; d++){ face->normal.V[d] = normal[d];}

    ierr = TDyCreateVecJoiningTwoVertices(f_cen, c_cen, f2c); CHKERRQ(ierr);
    ierr = TDyDotProduct(normal,f2c,&dot_prod); CHKERRQ(ierr);
    if ( dot_prod > 0.0 ) {
      PetscInt tmp;
      tmp = face->cell_ids[0];
      face->cell_ids[0] = face->cell_ids[1];
      face->cell_ids[1] = tmp;
    }
  }
  
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputCells2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  PetscInt       dim;
  PetscInt       icell, d, k;
  PetscErrorCode ierr;

  dm = tdy->dm;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  cells    = mesh->cells;

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

    // set pointer to cell
    cell = &cells[icell];

    // save centroid
    ierr = TDyCell_GetCentroid(cell, dim, &cell_cen_v[icell*dim]); CHKERRQ(ierr);

    // save volume
    cell_vol_v[icell] = cell->volume;

    for (k=0; k<4; k++) {
      neigh_id_v [icell*4 + k] = cell->neighbor_ids[k];
      vertex_id_v[icell*4 + k] = cell->vertex_ids[k];
      edge_id_v  [icell*4 + k] = cell->edge_ids[k];

      subcell = &cell->subcells[k];

      scell_vol_v[icell*4 + k] = subcell->T;

      scell_gmatrix_v[icell*4*4 + k*4 + 0] = tdy->subc_Gmatrix[icell][k][0][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 1] = tdy->subc_Gmatrix[icell][k][0][1];
      scell_gmatrix_v[icell*4*4 + k*4 + 2] = tdy->subc_Gmatrix[icell][k][1][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 3] = tdy->subc_Gmatrix[icell][k][1][1];

      ierr = TDySubCell_GetIthNuVector(subcell, 0, dim, &scell_nu_v[count]); CHKERRQ(ierr);
      for (d=0; d<dim; d++) {
        scell_cp_v[count] = subcell->variable_continuity_coordinates[0].X[d];
        count++;
      }

      ierr = TDySubCell_GetIthNuVector(subcell, 1, dim, &scell_nu_v[count]); CHKERRQ(ierr);
      for (d=0; d<dim; d++) {
        scell_cp_v[count] = subcell->variable_continuity_coordinates[1].X[d];
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
  TDy_edge       *edges, *edge;
  PetscInt       dim;
  PetscInt       iedge;
  PetscErrorCode ierr;

  dm = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh  = tdy->mesh;
  edges = mesh->edges;

  Vec edge_cen, edge_nor;
  PetscScalar *edge_cen_v, *edge_nor_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_cen);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_nor);
  CHKERRQ(ierr);

  ierr = VecGetArray(edge_cen, &edge_cen_v); CHKERRQ(ierr);
  ierr = VecGetArray(edge_nor, &edge_nor_v); CHKERRQ(ierr);

  for (iedge=0; iedge<mesh->num_edges; iedge++) {
    edge = &edges[iedge];
    ierr = TDyEdge_GetCentroid(edge, dim, &edge_cen_v[iedge*dim]); CHKERRQ(ierr);
    ierr = TDyEdge_GetNormal(  edge, dim, &edge_nor_v[iedge*dim]); CHKERRQ(ierr);
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
  TDy_vertex     *vertices, *vertex;
  PetscInt       dim;
  PetscInt       ivertex, i;
  PetscErrorCode ierr;

  dm = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  vertices = mesh->vertices;

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
    vertex = &vertices[ivertex];
    ierr = TDyVertex_GetCoordinate(vertex, dim, &vert_coord_v[ivertex*dim]); CHKERRQ(ierr);
    for (i=0; i<4; i++) {
      vert_icell_ids_v[ivertex*4 + i] = vertex->internal_cell_ids[i];
      vert_edge_ids_v[ivertex*4 + i] = vertex->edge_ids[i];
      vert_subcell_ids_v[ivertex*4 + i] = vertex->subcell_ids[i];
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
    break;

  case 3:
    ierr = SaveMeshGeometricAttributes(tdy); CHKERRQ(ierr);
    ierr = SaveMeshConnectivityInfo(   tdy); CHKERRQ(ierr);
    ierr = UpdateFaceOrderAroundAVertex3DMesh(tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAFace3DMesh(  tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor3DMesh     (tdy); CHKERRQ(ierr);
    break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in TDyBuildMesh");
    break;
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDy_cell *cell, TDy_vertex *vertex, TDy_subcell** subcell) {

  PetscFunctionBegin;

  PetscInt i, isubcell = -1;

  for (i=0; i<vertex->num_internal_cells;i++){
    if (vertex->internal_cell_ids[i] == cell->id) {
      isubcell = vertex->subcell_ids[i];
      break;
    }
  }

  if (isubcell == -1) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a subcell of a given cell that includes the given vertex");
  }
  
  *subcell = &cell->subcells[isubcell];

  PetscFunctionReturn(0);
}
