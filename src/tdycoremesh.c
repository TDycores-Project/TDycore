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

  ierr = Allocate_TDyVector_1D(num_nu_vectors, &subcell->nu_vector); CHKERRQ(ierr);
  ierr = Allocate_TDyCoordinate_1D(num_nu_vectors, &subcell->variable_continuity_coordinates); CHKERRQ(ierr);
  ierr = Allocate_TDyCoordinate_1D(num_vertices, &subcell->vertices_cordinates); CHKERRQ(ierr);

  ierr = Allocate_IntegerArray_1D(&subcell->face_ids,num_faces); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&subcell->face_area,num_faces); CHKERRQ(ierr);

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

  ierr = Allocate_IntegerArray_1D(&cell->vertex_ids  ,num_vertices ); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&cell->edge_ids    ,num_edges    ); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&cell->neighbor_ids,num_neighbors); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&cell->face_ids    ,num_faces    ); CHKERRQ(ierr);

  Initialize_IntegerArray_1D(cell->face_ids, num_faces, 0);

  cell->num_subcells = num_subcells;

  ierr = Allocate_TDySubcell_1D(num_subcells, &cell->subcells); CHKERRQ(ierr);

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

  ierr = Allocate_IntegerArray_1D(&vertex->edge_ids         ,num_edges ); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&vertex->face_ids         ,num_faces ); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&vertex->internal_cell_ids,num_internal_cells ); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&vertex->subcell_ids      ,num_internal_cells ); CHKERRQ(ierr);

  Initialize_IntegerArray_1D(vertex->face_ids, num_faces, 0);

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

  ierr = Allocate_IntegerArray_1D(&edge->cell_ids,num_cells); CHKERRQ(ierr);

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

  ierr = Allocate_IntegerArray_1D(&face->cell_ids,face->num_cells); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&face->edge_ids,face->num_edges); CHKERRQ(ierr);
  ierr = Allocate_IntegerArray_1D(&face->vertex_ids,face->num_vertices); CHKERRQ(ierr);

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
PetscErrorCode AllocateMemoryForMesh(DM dm, TDy_mesh *mesh) {

  PetscFunctionBegin;

  PetscInt nverts_per_cell;
  PetscInt cStart, cEnd, cNum;
  PetscInt vStart, vEnd, vNum;
  PetscInt eStart, eEnd, eNum;
  PetscInt fNum;
  PetscInt dim;
  TDyCellType cell_type;

  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (dim!= 2 && dim!=3 ) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only 2D and 3D grids are supported");
  }

  /* compute number of vertices per grid cell */
  nverts_per_cell = GetNumberOfCellVertices(dm);
  cell_type = GetCellType(dim, nverts_per_cell);

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

  ierr = Allocate_TDyCell_1D(cNum, &mesh->cells); CHKERRQ(ierr);
  ierr = Allocate_TDyFace_1D(fNum, &mesh->faces); CHKERRQ(ierr);
  ierr = Allocate_TDyEdge_1D(eNum, &mesh->edges); CHKERRQ(ierr);
  ierr = Allocate_TDyVertex_1D(vNum, &mesh->vertices); CHKERRQ(ierr);

  ierr = AllocateMemoryForCells(cNum, cell_type, mesh->cells); CHKERRQ(ierr);
  ierr = AllocateMemoryForVertices(vNum, cell_type, mesh->vertices);
  CHKERRQ(ierr);

  ierr = AllocateMemoryForEdges(eNum, cell_type, mesh->edges);
  CHKERRQ(ierr);

  ierr = AllocateMemoryForFaces(fNum, cell_type, mesh->faces);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode SaveTwoDimMeshGeometricAttributes(TDy tdy) {

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
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode SaveTwoDimMeshConnectivityInfo(TDy tdy) {

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
  PetscInt       closureSize, supportSize, coneSize;
  PetscInt       *closure;
  const PetscInt *support, *cone;
  PetscInt       c2vCount, c2eCount, c2fCount;
  PetscInt       nverts_per_cell;
  PetscInt       i,j,e,v,s;
  PetscReal      v_1[3], v_2[3];
  PetscInt       d;
  PetscBool      use_cone;
  PetscErrorCode ierr;

  dm = tdy->dm;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  nverts_per_cell = GetNumberOfCellVertices(dm);

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
  use_cone = PETSC_TRUE;
  for (icell=cStart; icell<cEnd; icell++) {
    closure  = NULL;
    ierr = DMPlexGetTransitiveClosure(dm, icell, use_cone, &closureSize, &closure);
    CHKERRQ(ierr);

    c2vCount = 0;
    c2eCount = 0;
    c2fCount = 0;

    for (i=0; i<closureSize*2; i+=2)  {

      if (IsClosureWithinBounds(closure[i], vStart,
                                vEnd)) { /* Is the closure a vertex? */
        PetscInt ivertex = closure[i] - vStart;
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
      } else if (IsClosureWithinBounds(closure[i], eStart,
                                       eEnd)) { /* Is the closure an edge? */
        PetscInt iedge = closure[i] - eStart;
        cells[icell].edge_ids[c2eCount] = iedge;
        for (j=0; j<2; j++) {
          if (edges[iedge].cell_ids[j] == -1) {
            edges[iedge].cell_ids[j] = icell;
            break;
          }
        }
        c2eCount++;
      } else if (IsClosureWithinBounds(closure[i], fStart,
                                       fEnd)) { /* Is the closure a face? */
        PetscInt iface = closure[i] - fStart;
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

    ierr = DMPlexRestoreTransitiveClosure(dm, icell, use_cone, &closureSize, &closure);
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

    for (d=0; d<dim; d++) {
      v_1[d] = vertices[edges[iedge].vertex_ids[0]].coordinate.X[d];
      v_2[d] = vertices[edges[iedge].vertex_ids[1]].coordinate.X[d];
      edges[iedge].centroid.X[d] = (v_1[d] + v_2[d])/2.0;
    }

    ierr = ComputeLength(v_1, v_2, dim, &(edges[iedge].length)); CHKERRQ(ierr);
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

    for (PetscInt c=0;c<coneSize;c++) {
      faces[iface].edge_ids[c] = cone[c]-eStart;

      PetscInt iedge = faces[iface].edge_ids[c];
      for (v=0;v<2;v++){
        PetscInt ivertex = edges[iedge].vertex_ids[v];
        vertex = &vertices[ivertex];
        PetscBool found = PETSC_FALSE;
        for (PetscInt ii=0; ii<vertex->num_faces; ii++) {
          if (vertex->face_ids[ii] == iface) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          vertex->face_ids[vertex->num_faces] = iface;
          vertex->num_faces++;

          face->vertex_ids[face->num_vertices] = vertex->id;
          face->num_vertices++;

          found = PETSC_TRUE;
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
        for (PetscInt ii=0; ii<cell->num_faces; ii++) {
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

PetscErrorCode UpdateCellOrientationAroundAVertexTwoDimMesh(TDy tdy) {

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
    PetscInt count=0;

    // First find all internal faces (i.e. face shared by two cells)
    for (PetscInt iface=0;iface<vertex->num_faces;iface++) {
      PetscInt face_id = vertex->face_ids[iface];
      if (faces[face_id].num_cells==2) {
        face_ids_sorted[count] = face_id;
        count++;
      }
    }
    
    // Now find all boundary faces (i.e. face shared by a single cell)
    for (PetscInt iface=0;iface<vertex->num_faces;iface++) {
      PetscInt face_id = vertex->face_ids[iface];
      if (faces[face_id].num_cells==1) {
        face_ids_sorted[count] = face_id;
        count++;
      }
    }

    // Save the sorted faces
    for (PetscInt iface=0;iface<vertex->num_faces;iface++) {
      vertex->face_ids[iface] = face_ids_sorted[iface];
    }
    
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAEdgeTwoDimMesh(TDy tdy) {

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

PetscErrorCode SetupSubcellsForTwoDimMesh(DM dm, TDy tdy) {

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
  PetscReal      normal, centroid;
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
    for (d=0; d<dim; d++) cell_cen[d] = cell->centroid.X[d];

    num_subcells = cell->num_subcells;

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      vertex  = &vertices[cell->vertex_ids[isubcell]];
      subcell = &cell->subcells[isubcell];

      // save coorindates of vertex that is part of the subcell
      for (d=0; d<dim; d++) v_c[d] = vertex->coordinate.X[d];

      // determine ids of up & down edges
      e_idx_up = cells[icell].edge_ids[isubcell];
      if (isubcell == 0) e_idx_dn = cells[icell].edge_ids[num_subcells-1];
      else               e_idx_dn = cells[icell].edge_ids[isubcell    -1];

      // set points to up/down edges
      edge_up = &edges[e_idx_up];
      edge_dn = &edges[e_idx_dn];

      // save centroids of up/down edges
      for (d=0; d<dim; d++) {
        e_cen_up[d] = edge_up->centroid.X[d];
        e_cen_dn[d] = edge_dn->centroid.X[d];
      }

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
      ierr = ComputeLength(cp_up, cell_cen, dim, &len_dn); CHKERRQ(ierr);
      ierr = ComputeLength(cp_dn, cell_cen, dim, &len_up); CHKERRQ(ierr);

      // save nu-vectors
      // note: length of nu-vectors is equal to length of edge diagonally
      //       opposite to the vector
      for (d=0; d<dim; d++) {
        subcell->nu_vector[0].V[d] = nu_vec_up[d]*len_up;
        subcell->nu_vector[1].V[d] = nu_vec_dn[d]*len_dn;
      }

      ierr = ComputeAreaOf2DTriangle(cp_up, cell_cen, cp_dn, &subcell->volume);

    }
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cell->volume), &centroid,
                                        &normal); CHKERRQ(ierr);
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
                                                  TDy_vertex *vertex, PetscInt f_idx[3],
                                                  PetscInt dim) {
  
  PetscFunctionBegin;
  
  PetscInt d;
  PetscErrorCode ierr;

  PetscReal a[3],b[3],c[3],axb[3],dot_prod;
  
  for (d=0; d<dim; d++) {
    a[d] = faces[f_idx[1]].centroid.X[d] - faces[f_idx[0]].centroid.X[d];
    b[d] = faces[f_idx[2]].centroid.X[d] - faces[f_idx[0]].centroid.X[d];
    c[d] = cell->centroid.X[d] - vertex->coordinate.X[d];
  }
  
  ierr = CrossProduct(a,b,axb); CHKERRQ(ierr);
  ierr = DotProduct(axb,c,&dot_prod);

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

PetscErrorCode SetupSubcellsFor3DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces, *face;
  PetscInt       cStart, cEnd, num_subcells;
  PetscInt       icell, isubcell;
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
    for (d=0; d<dim; d++) cell_cen[d] = cell->centroid.X[d];

    num_subcells = cell->num_subcells;

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      vertex  = &vertices[cell->vertex_ids[isubcell]];
      subcell = &cell->subcells[isubcell];

      // save coorindates of vertex that is part of the subcell
      for (d=0; d<dim; d++) v_c[d] = vertex->coordinate.X[d];

      PetscInt num_shared_faces;

      ierr = FindFaceIDsOfACellCommonToAVertex(cell, faces, vertex, subcell->face_ids, &num_shared_faces);

      // Update order of faces in f_idx so (face_ids[0], face_ids[1], face_ids[2])
      // form a plane such that normal to plane points toward the cell centroid
      ierr = UpdateFaceOrientationAroundAVertex(cell, faces, vertex, subcell->face_ids, dim);
      
      PetscReal face_cen[3][3];

      PetscInt iface;
      for (iface=0; iface<num_shared_faces; iface++){

        PetscReal f_normal[3];

        face = &faces[subcell->face_ids[iface]];

        for (d=0; d<dim; d++) face_cen[iface][d] = face->centroid.X[d];

        PetscInt neighboring_vertex_ids[2];

        ierr = FindNeighboringVerticesOfAFace(face,vertex->id,neighboring_vertex_ids);
        
        PetscReal edge0_cen[3], edge1_cen[3];

        for (d=0; d<dim; d++) {
          edge0_cen[d] = (v_c[d] + vertices[neighboring_vertex_ids[0]].coordinate.X[d])/2.0;
          edge1_cen[d] = (v_c[d] + vertices[neighboring_vertex_ids[1]].coordinate.X[d])/2.0;
        }

        // area of face
        ierr = QuadrilateralArea(v_c, edge0_cen, face_cen[iface], edge1_cen, &subcell->face_area[iface]);

        // normal to face
        if (face->cell_ids[0] == cell->id) {
          ierr = NormalToQuadrilateral(v_c, edge0_cen, face_cen[iface], edge1_cen, f_normal); CHKERRQ(ierr);
        } else {
          ierr = NormalToQuadrilateral(v_c, edge1_cen, face_cen[iface], edge0_cen, f_normal); CHKERRQ(ierr);
        }

        // nu vec
        PetscInt f1_idx, f2_idx;
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
        
        TDy_face *face1, *face2;
        face1 = &faces[subcell->face_ids[f1_idx]];
        face2 = &faces[subcell->face_ids[f2_idx]];

        for (d=0; d<dim; d++) {
          f1[d] = face1->centroid.X[d];
          f2[d] = face2->centroid.X[d];
        }

        ierr = NormalToTriangle(cell_cen, f1, f2, f_normal);
        for (d=0; d<dim; d++) subcell->nu_vector[iface].V[d] = f_normal[d];

      }

      ierr = ComputeVolumeOfTetrahedron(cell_cen, face_cen[0], face_cen[1], face_cen[2], &subcell->volume); CHKERRQ(ierr);
    }

  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateCellOrientationAroundAFace3DMesh(TDy tdy) {

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

    vertex = &vertices[face->vertex_ids[0]]; for (d=0; d<dim; d++){ v1[d] = vertex->coordinate.X[d];}
    vertex = &vertices[face->vertex_ids[1]]; for (d=0; d<dim; d++){ v2[d] = vertex->coordinate.X[d];}
    vertex = &vertices[face->vertex_ids[2]]; for (d=0; d<dim; d++){ v3[d] = vertex->coordinate.X[d];}
    vertex = &vertices[face->vertex_ids[3]]; for (d=0; d<dim; d++){ v4[d] = vertex->coordinate.X[d];}

    for (d=0; d<dim; d++){ f_cen[d] = face->centroid.X[d];}
    cell = &cells[face->cell_ids[0]]; for (d=0; d<dim; d++){ c_cen[d] = cell->centroid.X[d];}

    ierr = NormalToQuadrilateral(v1, v2, v3, v4, normal); CHKERRQ(ierr);
    for (d=0; d<dim; d++){ face->normal.V[d] = normal[d];}

    ierr = CreateVecJoiningTwoVertices(f_cen, c_cen, f2c); CHKERRQ(ierr);
    ierr = DotProduct(normal,f2c,&dot_prod); CHKERRQ(ierr);
    if ( dot_prod < 0.0 ) {
      PetscInt tmp;
      tmp = face->cell_ids[0];
      face->cell_ids[0] = face->cell_ids[1];
      face->cell_ids[1] = tmp;
    }
  }
  
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode SavePetscVecAsBinary(Vec vec, const char filename[]) {

  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE,
                               &viewer); CHKERRQ(ierr);
  ierr = VecView(vec, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputCellsTwoDimMesh(TDy tdy) {

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
    for (d=0; d<dim; d++) cell_cen_v[icell*dim + d] = cell->centroid.X[d];

    // save volume
    cell_vol_v[icell] = cell->volume;

    for (k=0; k<4; k++) {
      neigh_id_v [icell*4 + k] = cell->neighbor_ids[k];
      vertex_id_v[icell*4 + k] = cell->vertex_ids[k];
      edge_id_v  [icell*4 + k] = cell->edge_ids[k];

      subcell = &cell->subcells[k];

      scell_vol_v[icell*4 + k] = subcell->volume;

      scell_gmatrix_v[icell*4*4 + k*4 + 0] = tdy->subc_Gmatrix[icell][k][0][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 1] = tdy->subc_Gmatrix[icell][k][0][1];
      scell_gmatrix_v[icell*4*4 + k*4 + 2] = tdy->subc_Gmatrix[icell][k][1][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 3] = tdy->subc_Gmatrix[icell][k][1][1];

      for (d=0; d<dim; d++) {
        scell_nu_v[count] = subcell->nu_vector[0].V[d];
        scell_cp_v[count] = subcell->variable_continuity_coordinates[0].X[d];
        count++;
      }
      for (d=0; d<dim; d++) {
        scell_nu_v[count] = subcell->nu_vector[1].V[d];
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

  ierr = SavePetscVecAsBinary(cell_cen, "cell_cen.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_vol, "cell_vol.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_neigh_ids, "cell_neigh_ids.bin");
  CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_vertex_ids, "cell_vertex_ids.bin");
  CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_edge_ids, "cell_edge_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_nu, "subcell_nu.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_cp, "subcell_cp.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_vol, "subcell_vol.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_gmatrix, "subcell_gmatrix.bin");
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
PetscErrorCode OutputEdgesTwoDimMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_edge       *edges, *edge;
  PetscInt       dim;
  PetscInt       iedge, d;
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
    for (d=0; d<dim; d++) edge_cen_v[iedge*dim + d] = edge->centroid.X[d];
    for (d=0; d<dim; d++) edge_nor_v[iedge*dim + d] = edge->normal.V[d];
  }

  ierr = VecRestoreArray(edge_cen, &edge_cen_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(edge_nor, &edge_nor_v); CHKERRQ(ierr);

  ierr = SavePetscVecAsBinary(edge_cen, "edge_cen.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(edge_nor, "edge_nor.bin"); CHKERRQ(ierr);

  ierr = VecDestroy(&edge_cen); CHKERRQ(ierr);
  ierr = VecDestroy(&edge_nor); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputVerticesTwoDimMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_vertex     *vertices, *vertex;
  PetscInt       dim;
  PetscInt       ivertex, i, d;
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
    for (d=0; d<dim;
         d++) vert_coord_v[ivertex*dim + d] = vertex->coordinate.X[d];
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

  ierr = SavePetscVecAsBinary(vert_coord, "vert_coord.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(vert_icell_ids, "vert_icell_ids.bin");
  CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(vert_edge_ids, "vert_edge_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(vert_subcell_ids, "vert_subcell_ids.bin");
  CHKERRQ(ierr);

  ierr = VecDestroy(&vert_coord); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputTransmissibilityMatrixTwoDimMesh(TDy tdy) {

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

  ierr = SavePetscVecAsBinary(tmat, "trans_matrix.bin"); CHKERRQ(ierr);

  ierr = VecDestroy(&tmat); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputTwoDimMesh(TDy tdy) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = OutputCellsTwoDimMesh(tdy); CHKERRQ(ierr);
  ierr = OutputVerticesTwoDimMesh(tdy); CHKERRQ(ierr);
  ierr = OutputEdgesTwoDimMesh(tdy); CHKERRQ(ierr);
  ierr = OutputTransmissibilityMatrixTwoDimMesh(tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputMesh(TDy tdy) {

  PetscErrorCode ierr;
  PetscInt dim;

  PetscFunctionBegin;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);
  switch(dim) {
    case 2:
      ierr = OutputTwoDimMesh(tdy); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Output of mesh only supported for 2D meshes");
      break;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode BuildMesh(TDy tdy) {

  PetscInt dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;


  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = SaveTwoDimMeshGeometricAttributes(tdy); CHKERRQ(ierr);
    ierr = SaveTwoDimMeshConnectivityInfo(   tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAVertexTwoDimMesh(tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsForTwoDimMesh     (  tdy->dm, tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAEdgeTwoDimMesh(  tdy); CHKERRQ(ierr);
    break;

  case 3:
    ierr = SaveTwoDimMeshGeometricAttributes(tdy); CHKERRQ(ierr);
    ierr = SaveTwoDimMeshConnectivityInfo(   tdy); CHKERRQ(ierr);
    ierr = UpdateFaceOrderAroundAVertex3DMesh(tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAFace3DMesh(  tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor3DMesh     (tdy); CHKERRQ(ierr);
    break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in BuildMesh");
    break;
  }

  PetscFunctionReturn(0);

}