#include "tdycore.h"
#include "tdycoremesh.h"
#include "tdyutils.h"
#include <petscblaslapack.h>

/* ---------------------------------------------------------------- */

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start, PetscInt end){
  return (closure >= start) && (closure < end);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Initialize_IntegerArray_1D(PetscInt *array_1D, PetscInt ndim_1, PetscInt init_value){

  PetscFunctionBegin;

  for(int i=0; i<ndim_1; i++)
    array_1D[i] = init_value;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Initialize_RealArray_1D(PetscReal *array_1D, PetscInt ndim_1, PetscReal value){

  PetscFunctionBegin;

  for(int i=0; i<ndim_1; i++) {
    array_1D[i] = value;
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Initialize_RealArray_2D(PetscReal **array_2D, PetscInt ndim_1, PetscInt ndim_2, PetscReal value){

  PetscFunctionBegin;

  for(int i=0; i<ndim_1; i++) {
    for (int j=0; j<ndim_2; j++) {
      array_2D[i][j] = value;
    }
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Initialize_RealArray_3D(PetscReal ***array_3D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3, PetscReal value){

  PetscFunctionBegin;

  for(int i=0; i<ndim_1; i++) {
    for (int j=0; j<ndim_2; j++) {
      for (int k=0; k<ndim_3; k++) {
	array_3D[i][j][k] = value;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Initialize_RealArray_4D(PetscReal ****array_4D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4, PetscReal value){

  PetscFunctionBegin;

  for (int i=0; i<ndim_1; i++) {
    for (int j=0; j<ndim_2; j++) {
      for (int k=0; k<ndim_3; k++) {
	for (int l=0; l<ndim_4; l++) {
	  array_4D[i][j][k][l] = value;
	}
      }
    }
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Allocate_RealArray_1D(PetscReal **array_1D, PetscInt ndim_1){

  PetscErrorCode ierr;

  PetscFunctionBegin;

  *array_1D = (PetscReal *)malloc(ndim_1*sizeof(PetscReal ));

  ierr = Initialize_RealArray_1D(*array_1D, ndim_1, 0.0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Allocate_RealArray_2D(PetscReal ***array_2D, PetscInt ndim_1, PetscInt ndim_2){

  PetscErrorCode ierr;

  PetscFunctionBegin;

  *array_2D = (PetscReal **)malloc(ndim_1*sizeof(PetscReal *));
  for(int i=0; i<ndim_1; i++)
    (*array_2D)[i] = (PetscReal *)malloc(ndim_2*sizeof(PetscReal ));

  ierr = Initialize_RealArray_2D(*array_2D, ndim_1, ndim_2, 0.0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Deallocate_RealArray_2D(PetscReal **array_2D, PetscInt ndim_1){

  PetscFunctionBegin;

  for(int i=0; i<ndim_1; i++){
    free(array_2D[i]);
  }
  free(array_2D);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Allocate_RealArray_3D(PetscReal ****array_3D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3){

  PetscErrorCode ierr;

  PetscFunctionBegin;

  *array_3D = (PetscReal ***)malloc(ndim_1*sizeof(PetscReal **));
  for(int i=0; i<ndim_1; i++){
    (*array_3D)[i] = (PetscReal **)malloc(ndim_2*sizeof(PetscReal *));
    for(int j=0; j<ndim_2; j++){
      (*array_3D)[i][j] = (PetscReal *)malloc(ndim_3*sizeof(PetscReal));
    }
  }

  ierr = Initialize_RealArray_3D(*array_3D, ndim_1, ndim_2, ndim_3, 0.0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------- */
PetscErrorCode Allocate_RealArray_4D(PetscReal *****array_4D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4){

  PetscErrorCode ierr;

  PetscFunctionBegin;

  *array_4D = (PetscReal ****)malloc(ndim_1*sizeof(PetscReal ***));
  for(int i=0; i<ndim_1; i++){
    (*array_4D)[i] = (PetscReal ***)malloc(ndim_2*sizeof(PetscReal **));
    for(int j=0; j<ndim_2; j++){
      (*array_4D)[i][j] = (PetscReal **)malloc(ndim_3*sizeof(PetscReal *));
      for(int k=0; k<ndim_3; k++){
        (*array_4D)[i][j][k] = (PetscReal *)malloc(ndim_4*sizeof(PetscReal));
      }
    }
  }

  ierr = Initialize_RealArray_4D(*array_4D, ndim_1, ndim_2, ndim_3, ndim_4, 0.0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForASubcell(
    TDy_subcell    *subcell,
    TDySubcellType subcell_type) {

  PetscFunctionBegin;

  PetscInt num_nu_vectors;
  PetscInt num_vertices;

  switch (subcell_type){
  case SUBCELL_QUAD_TYPE:
    num_nu_vectors = 2;
    num_vertices   = 4;
    break;
  case SUBCELL_HEX_TYPE:
    num_nu_vectors = 3;
    num_vertices   = 8;
    break;
  }

  subcell->type           = subcell_type;
  subcell->num_nu_vectors = num_nu_vectors;
  subcell->num_vertices   = num_vertices;

  subcell->nu_vector                       = (TDy_vector     *) malloc(num_nu_vectors * sizeof(TDy_vector    ));
  subcell->variable_continuity_coordinates = (TDy_coordinate *) malloc(num_nu_vectors * sizeof(TDy_coordinate));
  subcell->vertices_cordinates             = (TDy_coordinate *) malloc(num_vertices   * sizeof(TDy_coordinate));

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForACell(
    TDy_cell       *cell,
    PetscInt       num_vertices,
    PetscInt       num_edges,
    PetscInt       num_neighbors,
    TDySubcellType subcell_type) {
  
  PetscFunctionBegin;

  PetscErrorCode ierr;
  PetscInt       num_subcells;
  TDy_subcell    *subcells;

  cell->num_vertices  = num_vertices;
  cell->num_edges     = num_edges;
  cell->num_neighbors = num_neighbors;

  cell->vertex_ids   = (PetscInt *) malloc(num_vertices  * sizeof(PetscInt));
  cell->edge_ids     = (PetscInt *) malloc(num_edges     * sizeof(PetscInt));
  cell->neighbor_ids = (PetscInt *) malloc(num_neighbors * sizeof(PetscInt));

  Initialize_IntegerArray_1D(cell->vertex_ids  , num_vertices , -1);
  Initialize_IntegerArray_1D(cell->edge_ids    , num_edges    , -1);
  Initialize_IntegerArray_1D(cell->neighbor_ids, num_neighbors, -1);

  switch (subcell_type){
  case SUBCELL_QUAD_TYPE:
    num_subcells = 4;
    break;
  case SUBCELL_HEX_TYPE:
    num_subcells = 8;
    break;
  }

  cell->num_subcells = num_subcells;
  cell->subcells     = (TDy_subcell *) malloc(num_subcells * sizeof(TDy_subcell));

  subcells = cell->subcells;
  for (int isubcell=0; isubcell<num_subcells; isubcell++){

    subcells[isubcell].id      = isubcell;
    subcells[isubcell].cell_id = cell->id;

    ierr = AllocateMemoryForASubcell(&subcells[isubcell], subcell_type); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForCells(
    PetscInt       num_cells,
    PetscInt       nverts_per_cell,
    TDySubcellType subcell_type,
    TDy_cell       *cells) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscInt num_vertices  = nverts_per_cell;
  PetscInt num_edges     = nverts_per_cell;
  PetscInt num_neighbors = nverts_per_cell;

  /* allocate memory for cells within the mesh*/
  for (int icell=0; icell<num_cells; icell++){
    cells[icell].id = icell;
    ierr = AllocateMemoryForACell(&cells[icell], num_vertices, num_edges,
                                  num_neighbors, subcell_type); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForAVertex(
    TDy_vertex     *vertex,
    PetscInt       num_internal_cells,
    PetscInt       num_edges,
    PetscInt       num_boundary_cells) {
  
  PetscFunctionBegin;

  vertex->num_internal_cells = 0;
  vertex->num_edges          = num_edges;
  vertex->num_boundary_cells = 0;

  vertex->edge_ids          = (PetscInt *) malloc(num_edges          * sizeof(PetscInt));
  vertex->internal_cell_ids = (PetscInt *) malloc(num_internal_cells * sizeof(PetscInt));
  vertex->subcell_ids       = (PetscInt *) malloc(num_internal_cells * sizeof(PetscInt));

  Initialize_IntegerArray_1D(vertex->edge_ids          , num_edges         , -1);
  Initialize_IntegerArray_1D(vertex->internal_cell_ids , num_internal_cells, -1);
  Initialize_IntegerArray_1D(vertex->subcell_ids       , num_internal_cells, -1);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForVertices(
    PetscInt       num_vertices,
    PetscInt       nverts_per_cell,
    TDy_vertex     *vertices) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscInt num_internal_cells = nverts_per_cell;
  PetscInt num_edges          = nverts_per_cell;
  PetscInt num_boundary_cells = 0;

  /* allocate memory for vertices within the mesh*/
  for (int ivertex=0; ivertex<num_vertices; ivertex++){
    vertices[ivertex].id = ivertex;
    ierr = AllocateMemoryForAVertex(&vertices[ivertex], num_internal_cells,
                                    num_edges, num_boundary_cells); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForAEdge(
    TDy_edge *edge,
    PetscInt num_cells) {
  
  PetscFunctionBegin;

  edge->num_cells = num_cells;

  edge->cell_ids = (PetscInt *) malloc(num_cells * sizeof(PetscInt));

  Initialize_IntegerArray_1D(edge->cell_ids, num_cells, -1);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForEdges(
    PetscInt num_edges,
    PetscInt ncells_per_edge,
    TDy_edge *edges) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  /* allocate memory for edges within the mesh*/
  for (int iedge=0; iedge<num_edges; iedge++){
    edges[iedge].id = iedge;
    ierr = AllocateMemoryForAEdge(&edges[iedge], ncells_per_edge); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForMesh(DM dm, TDy_mesh *mesh) {

  PetscFunctionBegin;

  PetscInt nverts_per_cell;
  PetscInt ncells_per_edge;
  PetscInt cStart, cEnd, cNum;
  PetscInt vStart, vEnd, vNum;
  PetscInt eStart, eEnd, eNum;

  PetscErrorCode ierr;

  /* compute number of vertices per grid cell */
  nverts_per_cell = GetNumberOfCellVertices(dm);
  ncells_per_edge = 2;

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd);CHKERRQ(ierr);

  cNum = cEnd - cStart;
  eNum = eEnd - eStart;
  vNum = vEnd - vStart;

  mesh->num_cells    = cNum;
  mesh->num_faces    = 0;
  mesh->num_edges    = eNum;
  mesh->num_vertices = vNum;

  mesh->cells    = (TDy_cell   *) malloc(cNum * sizeof(TDy_cell   ));
  mesh->edges    = (TDy_edge   *) malloc(eNum * sizeof(TDy_edge   ));
  mesh->vertices = (TDy_vertex *) malloc(vNum * sizeof(TDy_vertex ));

  ierr = AllocateMemoryForCells(cNum, nverts_per_cell, SUBCELL_QUAD_TYPE, mesh->cells); CHKERRQ(ierr);
  ierr = AllocateMemoryForVertices(vNum, nverts_per_cell, mesh->vertices); CHKERRQ(ierr);
  ierr = AllocateMemoryForEdges(eNum, ncells_per_edge, mesh->edges); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode SaveTwoDimMeshGeometricAttributes(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  PetscInt       dim;
  PetscInt       cStart, cEnd;
  PetscInt       vStart, vEnd;
  PetscInt       eStart, eEnd;
  PetscInt       pStart, pEnd;
  PetscInt       icell, iedge, ivertex;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(        dm, &pStart, &pEnd); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  for (int ielement=pStart; ielement<pEnd; ielement++){

    if (IsClosureWithinBounds(ielement, vStart, vEnd)) { // is the element a vertex?
      ivertex = ielement - vStart;
      for (int d=0; d<dim; d++) {
        vertices[ivertex].coordinate.X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, eStart, eEnd)) { // is the element an edge?
      iedge = ielement - eStart;
      for (int d=0; d<dim; d++) {
        edges[iedge].centroid.X[d] = tdy->X[ielement*dim + d];
        edges[iedge].normal.V[d]   = tdy->N[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, cStart, cEnd)) { // is the elment a cell?
      icell = ielement - cStart;
      for (int d=0; d<dim; d++) {
        cells[icell].centroid.X[d] = tdy->X[ielement*dim + d];
      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode SaveTwoDimMeshConnectivityInfo(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_edge       *edges;
  PetscInt       dim;
  PetscInt       cStart, cEnd;
  PetscInt       vStart, vEnd;
  PetscInt       eStart, eEnd;
  PetscInt       pStart, pEnd;
  PetscInt       icell, iedge, ivertex;
  PetscInt       closureSize, supportSize, coneSize;
  PetscInt       *closure;
  const PetscInt *support, *cone;
  PetscInt       c2vCount, c2eCount;
  PetscInt       nverts_per_cell;
  PetscBool      use_cone;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  nverts_per_cell = GetNumberOfCellVertices(dm);

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(        dm, &pStart, &pEnd); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  // cell--to--vertex
  // edge--to--cell
  // cell--to--edge
  // edge--to--cell
  use_cone = PETSC_TRUE;
  for (icell=cStart; icell<cEnd; icell++){
    closure  = NULL;
    ierr = DMPlexGetTransitiveClosure(dm, icell, use_cone, &closureSize, &closure); CHKERRQ(ierr);

    c2vCount = 0;
    c2eCount = 0;

    for (int i=0; i<closureSize*2; i+=2)  {

      if (IsClosureWithinBounds(closure[i], vStart, vEnd)) { /* Is the closure a vertex? */
        ivertex = closure[i] - vStart;
        cells[icell].vertex_ids[c2vCount] = ivertex ;
        for (int j=0; j<nverts_per_cell; j++){
          if (vertices[ivertex].internal_cell_ids[j] == -1){
            vertices[ivertex].num_internal_cells++;
            vertices[ivertex].internal_cell_ids[j] = icell;
            vertices[ivertex].subcell_ids[j]       = c2vCount;
            break;
          }
        }
        c2vCount++;
      } else if (IsClosureWithinBounds(closure[i], eStart, eEnd)){ /* Is the closure an edge? */
        iedge = closure[i] - eStart;
        cells[icell].edge_ids[c2eCount] = iedge;
        for (int j=0; j<2; j++){
          if (edges[iedge].cell_ids[j] == -1){
            edges[iedge].cell_ids[j] = icell;
            break;
          }
        }
        c2eCount++;
      }
    }
  }

  // edge--to--vertex
  for (int e=eStart; e<eEnd; e++) {
    ierr = DMPlexGetConeSize(dm, e, &coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, e, &cone); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, e, &supportSize); CHKERRQ(ierr);
    iedge = e-eStart;

    if (supportSize == 1) edges[iedge].is_internal = PETSC_FALSE;
    else                  edges[iedge].is_internal = PETSC_TRUE;

    edges[iedge].vertex_ids[0] = cone[0]-vStart;
    edges[iedge].vertex_ids[1] = cone[1]-vStart;
  }

  // vertex--to--edge
  for (int v=vStart; v<vEnd; v++){
    ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, v, &supportSize); CHKERRQ(ierr);
    ivertex = v - vStart;
    vertices[ivertex].num_edges = supportSize;
    for (int s=0; s<supportSize; s++) {
      iedge = support[s] - eStart;
      vertices[ivertex].edge_ids[s] = iedge;
      if (!edges[iedge].is_internal) vertices[ivertex].num_boundary_cells++;
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode ComputeTheta(PetscReal x, PetscReal y, PetscReal *theta){

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

PetscErrorCode UpdateCellOrientationAroundAVertex(TDy tdy, PetscInt ivertex){

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
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertex   = &(mesh->vertices[ivertex]);

  ncells = vertex->num_internal_cells;
  nedges = vertex->num_edges;
  count  = 0;

  // compute angle to all cell centroids w.r.t. the shared vertix
  for (PetscInt i=0; i<ncells; i++) {
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

  for (PetscInt i=0; i<nedges; i++) {
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
  if (boundary_edge_present) {
    // for a boundary vertex, find the last boundary edge in the
    // anitclockwise direction around the vertex
    for (PetscInt i=0; i<count; i++){
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
  for (PetscInt i=start_idx+1; i<count; i++) {

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
  for (PetscInt i=0; i<=start_idx; i++) {
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
  for (PetscInt i=0; i<tmp_ncells; i++) {
    vertex->internal_cell_ids[i] = tmp_cell_ids[i];
    vertex->subcell_ids[i]       = tmp_subcell_ids[i];
  }

  // save information about sorted edge ids
  for (PetscInt i=0; i<tmp_nedges; i++) {
    vertex->edge_ids[i] = tmp_edge_ids[i];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAVertexTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  PetscInt       vStart, vEnd;
  TDy_vertex     *vertices, *vertex;
  PetscInt       ivertex;
  PetscInt       edge_id_1, edge_id_2;
  TDy_edge       *edges;
  PetscReal      x,y, theta_1, theta_2;
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd);CHKERRQ(ierr);

  for (ivertex=0; ivertex<vEnd-vStart; ivertex++){

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

PetscErrorCode UpdateCellOrientationAroundAEdgeTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  PetscReal      dot_product;
  PetscInt       eStart, eEnd;
  PetscInt       iedge;
  TDy_cell       *cells, *cell_from, *cell_to;
  TDy_edge       *edges, *edge;
  PetscErrorCode ierr;

  mesh  = tdy->mesh;
  cells = mesh->cells;
  edges = mesh->edges;

  ierr = DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);

  for (iedge=0; iedge<eEnd-eStart; iedge++){
    edge = &(edges[iedge]);
    if (edge->is_internal) {
      cell_from = &cells[edge->cell_ids[0]];
      cell_to   = &cells[edge->cell_ids[1]];
      
      dot_product = (cell_to->centroid.X[0] - cell_from->centroid.X[0]) * edge->normal.V[0] +
                    (cell_to->centroid.X[1] - cell_from->centroid.X[1]) * edge->normal.V[1];
      if (dot_product < 0.0) {
        edge->cell_ids[0] = cell_to->id;
        edge->cell_ids[1] = cell_from->id;
      }
    } else {
      cell_from = &cells[edge->cell_ids[0]];

      dot_product = (edge->centroid.X[0] - cell_from->centroid.X[0]) * edge->normal.V[0] +
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
PetscErrorCode ComputeVariableContinuityPoint(PetscReal vertex[3], PetscReal edge[3], PetscReal alpha, PetscInt dim, PetscReal *point){

  PetscFunctionBegin;

  for (int d=0; d<dim; d++) point[d] = (1.0 - alpha)*vertex[d] + edge[d]*alpha;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeRightNormalVector(PetscReal v1[3], PetscReal v2[3], PetscInt dim, PetscReal *normal){

  PetscReal vec_from_1_to_2[3];
  PetscReal norm;

  PetscFunctionBegin;

  if (dim != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"ComputeRightNormalVector only support 2D grids");

  norm = 0.0;

  for (int d=0; d<dim; d++) {
    vec_from_1_to_2[d] = v2[d] - v1[d];
    norm += pow(vec_from_1_to_2[d], 2.0);
  }
  norm = pow(norm, 0.5);

  normal[0] =  vec_from_1_to_2[1]/norm;
  normal[1] = -vec_from_1_to_2[0]/norm;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim, PetscReal *length){

  PetscFunctionBegin;

  *length = 0.0;

  for (int d=0; d<dim; d++) *length += pow(v1[d] - v2[d], 2.0);

  *length = pow(*length, 0.5);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeAreaOf2DTriangle(PetscReal v1[3], PetscReal v2[3], PetscReal v3[3], PetscReal *area){

  PetscFunctionBegin;

  /*
   *
   *  v1[0] v1[1] 1.0
   *  v2[0] v2[1] 1.0
   *  v3[0] v3[1] 1.0
   *
  */

  *area = fabs(v1[0]*(v2[1] - v3[1]) - v1[1]*(v2[0] - v3[0]) + 1.0*(v2[0]*v3[1] - v2[1]*v3[0]))/2.0;

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
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_up, alpha, dim, cp_up); CHKERRQ(ierr);
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_dn, alpha, dim, cp_dn); CHKERRQ(ierr);

      // save continuity point
      for (d=0; d<dim; d++) {
        subcell->variable_continuity_coordinates[0].X[d] = cp_up[d];
        subcell->variable_continuity_coordinates[1].X[d] = cp_dn[d];
      }

      // compute the 'direction' of nu-vector
      ierr = ComputeRightNormalVector(cp_up   , cell_cen, dim, nu_vec_dn); CHKERRQ(ierr);
      ierr = ComputeRightNormalVector(cell_cen, cp_dn   , dim, nu_vec_up); CHKERRQ(ierr);

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
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cell->volume), &centroid, &normal); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode SavePetscVecAsBinary(Vec vec, const char filename[]) {

  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(vec, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputCellsTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  PetscInt       dim;
  PetscInt       icell;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  cells    = mesh->cells;

  Vec cell_cen, cell_vol;
  Vec cell_neigh_ids, cell_vertex_ids, cell_edge_ids;
  Vec scell_nu, scell_cp, scell_vol, scell_gmatrix;

  PetscScalar *cell_cen_v, *cell_vol_v;
  PetscScalar *neigh_id_v, *vertex_id_v, *edge_id_v;
  PetscScalar *scell_nu_v, *scell_cp_v, *scell_vol_v, *scell_gmatrix_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*dim     , &cell_cen); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells         , &cell_vol);  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4       , &cell_neigh_ids); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4       , &cell_vertex_ids); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4       , &cell_edge_ids); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4*dim*2 , &scell_nu); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4*dim*2 , &scell_cp); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4       , &scell_vol); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_cells*4*4     , &scell_gmatrix); CHKERRQ(ierr);

  ierr = VecGetArray(cell_cen       , &cell_cen_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_vol       , &cell_vol_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_neigh_ids , &neigh_id_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_vertex_ids, &vertex_id_v); CHKERRQ(ierr);
  ierr = VecGetArray(cell_edge_ids  , &edge_id_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_nu       , &scell_nu_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_cp       , &scell_cp_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_vol      , &scell_vol_v); CHKERRQ(ierr);
  ierr = VecGetArray(scell_gmatrix  , &scell_gmatrix_v); CHKERRQ(ierr);

  PetscInt count = 0;
  for (icell = 0; icell < mesh->num_cells; icell++){

    // set pointer to cell
    cell = &cells[icell];

    // save centroid
    for (PetscInt d=0; d<dim; d++) cell_cen_v[icell*dim + d] = cell->centroid.X[d];

    // save volume
    cell_vol_v[icell] = cell->volume;

    for (PetscInt k=0; k<4; k++){
      neigh_id_v [icell*4 + k] = cell->neighbor_ids[k];
      vertex_id_v[icell*4 + k] = cell->vertex_ids[k];
      edge_id_v  [icell*4 + k] = cell->edge_ids[k];

      subcell = &cell->subcells[k];

      scell_vol_v[icell*4 + k] = subcell->volume;

      scell_gmatrix_v[icell*4*4 + k*4 + 0] = tdy->subc_Gmatrix[icell][k][0][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 1] = tdy->subc_Gmatrix[icell][k][0][1];
      scell_gmatrix_v[icell*4*4 + k*4 + 2] = tdy->subc_Gmatrix[icell][k][1][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 3] = tdy->subc_Gmatrix[icell][k][1][1];

      for (PetscInt d=0; d<dim; d++) {
        scell_nu_v[count] = subcell->nu_vector[0].V[d];
        scell_cp_v[count] = subcell->variable_continuity_coordinates[0].X[d];
        count++;
      }
      for (PetscInt d=0; d<dim; d++) {
        scell_nu_v[count] = subcell->nu_vector[1].V[d];
        scell_cp_v[count] = subcell->variable_continuity_coordinates[1].X[d];
        count++;
      }
    }

  }

  ierr = VecRestoreArray(cell_cen       , &cell_cen_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_vol       , &cell_vol_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_neigh_ids , &neigh_id_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_vertex_ids, &vertex_id_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(cell_edge_ids  , &edge_id_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_nu       , &scell_nu_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_cp       , &scell_cp_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_vol      , &scell_vol_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(scell_gmatrix  , &scell_gmatrix_v); CHKERRQ(ierr);

  ierr = SavePetscVecAsBinary(cell_cen       , "cell_cen.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_vol       , "cell_vol.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_neigh_ids , "cell_neigh_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_vertex_ids, "cell_vertex_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(cell_edge_ids  , "cell_edge_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_nu       , "subcell_nu.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_cp       , "subcell_cp.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_vol      , "subcell_vol.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(scell_gmatrix  , "subcell_gmatrix.bin"); CHKERRQ(ierr);

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
PetscErrorCode OutputEdgesTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_edge       *edges, *edge;
  PetscInt       dim;
  PetscInt       iedge;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh  = tdy->mesh;
  edges = mesh->edges;

  Vec edge_cen, edge_nor;
  PetscScalar *edge_cen_v, *edge_nor_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_cen); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_nor); CHKERRQ(ierr);

  ierr = VecGetArray(edge_cen, &edge_cen_v); CHKERRQ(ierr);
  ierr = VecGetArray(edge_nor, &edge_nor_v); CHKERRQ(ierr);

  for (iedge=0; iedge<mesh->num_edges; iedge++) {
    edge = &edges[iedge];
    for (PetscInt d=0; d<dim; d++) edge_cen_v[iedge*dim + d] = edge->centroid.X[d];
    for (PetscInt d=0; d<dim; d++) edge_nor_v[iedge*dim + d] = edge->normal.V[d];
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
PetscErrorCode OutputVerticesTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_vertex     *vertices, *vertex;
  PetscInt       dim;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  vertices = mesh->vertices;

  Vec vert_coord, vert_icell_ids, vert_edge_ids, vert_subcell_ids;
  PetscScalar *vert_coord_v, *vert_icell_ids_v, *vert_edge_ids_v, *vert_subcell_ids_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*dim, &vert_coord); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*4, &vert_icell_ids); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*4, &vert_edge_ids); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*4, &vert_subcell_ids); CHKERRQ(ierr);

  ierr = VecGetArray(vert_coord, &vert_coord_v); CHKERRQ(ierr);
  ierr = VecGetArray(vert_icell_ids, &vert_icell_ids_v); CHKERRQ(ierr);
  ierr = VecGetArray(vert_edge_ids, &vert_edge_ids_v); CHKERRQ(ierr);
  ierr = VecGetArray(vert_subcell_ids, &vert_subcell_ids_v); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    vertex = &vertices[ivertex];
    for (PetscInt d=0; d<dim; d++) vert_coord_v[ivertex*dim + d] = vertex->coordinate.X[d];
    for (PetscInt i=0; i<4; i++) {
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
  ierr = SavePetscVecAsBinary(vert_icell_ids, "vert_icell_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(vert_edge_ids, "vert_edge_ids.bin"); CHKERRQ(ierr);
  ierr = SavePetscVecAsBinary(vert_subcell_ids, "vert_subcell_ids.bin"); CHKERRQ(ierr);

  ierr = VecDestroy(&vert_coord); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode OutputTransmissibilityMatrixTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  PetscInt       dim;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  mesh     = tdy->mesh;

  Vec tmat;
  PetscScalar *tmat_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*5*5, &tmat); CHKERRQ(ierr);

  ierr = VecGetArray(tmat, &tmat_v); CHKERRQ(ierr);

  PetscInt count;

  count = 0;
  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    for (PetscInt i=0; i<5; i++) {
      for (PetscInt j=0; j<5; j++) {
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
PetscErrorCode OutputTwoDimMesh(DM dm, TDy tdy) {

  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = OutputCellsTwoDimMesh(dm, tdy); CHKERRQ(ierr);
  ierr = OutputVerticesTwoDimMesh(dm, tdy); CHKERRQ(ierr);
  ierr = OutputEdgesTwoDimMesh(dm, tdy); CHKERRQ(ierr);
  ierr = OutputTransmissibilityMatrixTwoDimMesh(dm, tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode BuildTwoDimMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  ierr = SaveTwoDimMeshGeometricAttributes(dm, tdy); CHKERRQ(ierr);
  ierr = SaveTwoDimMeshConnectivityInfo(   dm, tdy); CHKERRQ(ierr);
  ierr = UpdateCellOrientationAroundAVertexTwoDimMesh(  dm, tdy); CHKERRQ(ierr);
  ierr = SetupSubcellsForTwoDimMesh     (  dm, tdy); CHKERRQ(ierr);
  ierr = UpdateCellOrientationAroundAEdgeTwoDimMesh(  dm, tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeEntryOfGMatrix(PetscReal edge_len, PetscReal n[3], PetscReal K[3][3], PetscReal v[3], PetscReal area, PetscInt dim, PetscReal *g){

  PetscFunctionBegin;

  PetscReal Kv[3];

  *g = 0.0;

  for (PetscInt i=0; i<dim; i++){
    Kv[i] = 0.0;
    for (PetscInt j=0; j<dim; j++){
      Kv[i] += K[i][j] * v[j];
    }
  }

  for (PetscInt i=0; i<dim; i++){
    (*g) += n[i] * Kv[i];
  }
  (*g) *= -1.0/(2.0*area)*edge_len;

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGMatrix(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges, *edge_up, *edge_dn;
  PetscInt       num_subcells;
  PetscInt       icell, isubcell;
  PetscInt       ii,jj;
  PetscInt       dim, d;
  PetscInt       e_idx_up, e_idx_dn;
  PetscReal      n_up[3], n_dn[3];
  PetscReal      e_cen_up[3], e_cen_dn[3], v_c[3];
  PetscReal      e_len_dn, e_len_up;
  PetscReal      K[3][3], nu_up[3], nu_dn[3];
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // set pointer to cell
    cell = &cells[icell];

    // extract permeability tensor
    for (ii=0; ii<dim; ii++) {
      for (jj=0; jj<dim; jj++) {
        K[ii][jj] = tdy->K[icell*dim*dim + ii*dim + jj];
      }
    }

    num_subcells = cell->num_subcells;

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      vertex  = &vertices[cell->vertex_ids[isubcell]];
      subcell = &cell->subcells[isubcell];

      // determine ids of up & down edges
      e_idx_up = cells[icell].edge_ids[isubcell];
      if (isubcell == 0) e_idx_dn = cells[icell].edge_ids[num_subcells-1];
      else               e_idx_dn = cells[icell].edge_ids[isubcell    -1];

      // set points to up/down edges
      edge_up = &edges[e_idx_up];
      edge_dn = &edges[e_idx_dn];

      for (d=0; d<dim; d++){

        // extract nu-vectors
        nu_up[d]    = subcell->nu_vector[0].V[d];
        nu_dn[d]    = subcell->nu_vector[1].V[d];

        // extract face centroid of edges
        e_cen_dn[d] = edge_dn->centroid.X[d];
        e_cen_up[d] = edge_up->centroid.X[d];

        // extract normal to edges
        n_dn[d] = edge_dn->normal.V[d];
        n_up[d] = edge_up->normal.V[d];

        // extract coordinate of the vertex
        v_c[d] = vertex->coordinate.X[d];
      }

      //
      ierr = ComputeLength(v_c, e_cen_dn, dim, &e_len_dn);
      ierr = ComputeLength(v_c, e_cen_up, dim, &e_len_up);

      //                               _         _   _           _
      //                              |           | |             |
      //                              | L_up*n_up | | K_xx   K_xy |  _             _
      // Gmatrix =        -1          |           | |             | |               |
      //             -----------      |           | |             | | nu_up   nu_dn |
      //              2*A_{subcell}   | L_dn*n_dn | | K_yx   K_yy | |_             _|
      //                              |           | |             |
      //                              |_         _| |_           _|
      //
      ComputeEntryOfGMatrix(e_len_up, n_up, K, nu_up, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][0][0]));
      ComputeEntryOfGMatrix(e_len_up, n_up, K, nu_dn, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][0][1]));
      ComputeEntryOfGMatrix(e_len_dn, n_dn, K, nu_up, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][1][0]));
      ComputeEntryOfGMatrix(e_len_dn, n_dn, K, nu_dn, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][1][1]));
    }
  }

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */

PetscErrorCode ExtractSubGmatrix(TDy tdy, PetscInt cell_id, PetscInt sub_cell_id, PetscInt dim, PetscReal **Gmatrix){

  PetscFunctionBegin;

  for (PetscInt i=0; i<dim; i++) {
    for (PetscInt j=0; j<dim; j++) {
      Gmatrix[i][j] = tdy->subc_Gmatrix[cell_id][sub_cell_id][i][j];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForInternalVertex(TDy tdy, TDy_vertex *vertex, TDy_cell *cells) {

  PetscInt       ncells, icell, isubcell;

  PetscReal **Fup, **Fdn;
  PetscReal **Cup, **Cdn;
  PetscReal **A, **B, **AinvB;
  PetscReal *A1d, *B1d, *Cup1d, *AinvB1d, *CuptimesAinvB1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscBLASInt info, *pivots;
  PetscInt n,m, ndim;
  PetscScalar zero = 0.0, one = 1.0;

  PetscFunctionBegin;

  ndim      = 2;
  ncells    = vertex->num_internal_cells;
  vertex_id = vertex->id;

  ierr = Allocate_RealArray_2D(&Gmatrix , ndim  , ndim  ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Fup     , ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Fdn     , ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Cup     , ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Cdn     , ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&A       , ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&B       , ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&AinvB   , ncells, ncells); CHKERRQ(ierr);

  ierr = Allocate_RealArray_1D(&A1d             , ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&B1d             , ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&Cup1d           , ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&AinvB1d         , ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&CuptimesAinvB1d , ncells*ncells); CHKERRQ(ierr);

  for (PetscInt i=0; i<ncells; i++) {
    icell    = vertex->internal_cell_ids[i];
    isubcell = vertex->subcell_ids[i];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    Fup[i][i] = Gmatrix[0][0] + Gmatrix[0][1];
    Cup[i][i] = Gmatrix[0][0];
    
    if (i<ncells-1) {
      Cup[i  ][i+1     ] = Gmatrix[0][1];
      Fdn[i+1][i       ] = Gmatrix[1][0] + Gmatrix[1][1];
      Cdn[i+1][i       ] = Gmatrix[1][0];
      Cdn[i+1][i+1     ] = Gmatrix[1][1];
    } else {
      Cup[i  ][0       ] = Gmatrix[0][1];
      Fdn[0  ][ncells-1] = Gmatrix[1][0] + Gmatrix[1][1];
      Cdn[0  ][ncells-1] = Gmatrix[1][0];
      Cdn[0  ][0       ] = Gmatrix[1][1];
    }
  }

  idx = 0;
  for (PetscInt j=0; j<ncells; j++){
    for (PetscInt i=0; i<ncells; i++){
      A[i][j] = -Cup[i][j] + Cdn[i][j];
      B[i][j] = -Fup[i][j] + Fdn[i][j];
      A1d[idx]= -Cup[i][j] + Cdn[i][j];
      B1d[idx]= -Fup[i][j] + Fdn[i][j];
      Cup1d[idx] = Cup[i][j];
      idx++;
    }
  }

  n = ncells; m = ncells;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);

  LAPACKgetrf_(&m, &n, A1d, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Bad LU factorization");

  ierr = PetscMemcpy(AinvB1d,B1d,sizeof(PetscScalar)*(n*m));CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1 * B) by back-substitution
  LAPACKgetrs_("N", &m, &n, A1d, &m, pivots, AinvB1d, &m, &info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (C * AinvB)
  BLASgemm_("N","N", &m, &m, &n, &one, Cup1d, &m, AinvB1d, &m, &zero, CuptimesAinvB1d, &m);

  idx = 0;
  for (PetscInt j=0; j<ncells; j++) {
    for (PetscInt i=0; i<ncells; i++) {
      AinvB[i][j] = AinvB1d[idx];
      tdy->Trans[vertex_id][i][j] = CuptimesAinvB1d[idx] - Fup[i][j];
      idx++;
    }
  }

  // Free up the memory
  ierr = Deallocate_RealArray_2D(Gmatrix , ndim   ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fup     , ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fdn     , ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cup     , ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cdn     , ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(A       , ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(B       , ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(AinvB   , ncells ); CHKERRQ(ierr);
  ierr = PetscFree(pivots                         ); CHKERRQ(ierr);

  free(A1d             );
  free(B1d             );
  free(Cup1d           );
  free(AinvB1d         );
  free(CuptimesAinvB1d );

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForBoundaryVertex(TDy tdy, TDy_vertex *vertex, TDy_cell *cells) {

  PetscInt       ncells_in, ncells_bc, icell, isubcell;

  PetscReal **Fup, **Cup, **Fdn, **Cdn;
  PetscReal **FupInxIn, **FdnInxIn; // InxIn: Internal flux with contribution from unknown internal pressure values
  PetscReal **FupBcxIn, **FdnBcxIn; // BcxIn: Boundary flux with contribution from unknown internal pressure values
  PetscReal **CupInxIn, **CdnInxIn; // InxIn: Internal flux with contribution from unknown internal pressure values
  PetscReal **CupInxBc, **CdnInxBc; // Inxbc: Internal flux with contribution from known boundary pressure values
  PetscReal **CupBcxIn, **CdnBcxIn; // BcxIn: Boundary flux with contribution from unknown internal pressure values
  PetscReal **CupBcxBc, **CdnBcxBc; // BcxIn: Boundary flux with contribution from known boundary pressure values

  PetscReal *AInxIninv_1d;
  PetscReal **AInxIn  , **BInxIn  , **AInxIninvBInxIn  ;
  PetscReal *AInxIn_1d, *BInxIn_1d, *DInxBc_1d, *AInxIninvBInxIn_1d, *AInxIninvDInxBc_1d;

  PetscReal *CupInxIn_1d, *CupInxIntimesAInxIninvBInxIn_1d, *CupInxIntimesAInxIninvDInxBc_1d;
  PetscReal *CupBcxIn_1d, *CdnBcxIn_1d;

  PetscReal *CupBcxIntimesAInxIninvBInxIn_1d, *CdnBcxIntimesAInxIninvBInxIn_1d;
  PetscReal *CupBcxIntimesAInxIninvDInxBc_1d, *CdnBcxIntimesAInxIninvDInxBc_1d;

  PetscReal *lapack_mem_1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscBLASInt info, *pivots;
  PetscInt n,m,k,ndim;
  PetscScalar zero = 0.0, one = 1.0;
  PetscInt i,j;

  PetscFunctionBegin;

  ndim      = 2;
  ncells_in = vertex->num_internal_cells;
  ncells_bc = vertex->num_boundary_cells;
  vertex_id = vertex->id;

  ierr = Allocate_RealArray_2D(&Gmatrix, ndim, ndim);
  
  ierr = Allocate_RealArray_2D(&Fup, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = Allocate_RealArray_2D(&Cup, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = Allocate_RealArray_2D(&Fdn, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = Allocate_RealArray_2D(&Cdn, ncells_in+ncells_bc, ncells_in+ncells_bc);
  
  ierr = Allocate_RealArray_2D(&FupInxIn, ncells_in-1, ncells_in);
  ierr = Allocate_RealArray_2D(&FupBcxIn, 1          , ncells_in);
  ierr = Allocate_RealArray_2D(&FdnInxIn, ncells_in-1, ncells_in);
  ierr = Allocate_RealArray_2D(&FdnBcxIn, 1          , ncells_in);
  
  ierr = Allocate_RealArray_2D(&CupInxIn, ncells_in-1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&CupInxBc, ncells_in-1, ncells_bc  );
  ierr = Allocate_RealArray_2D(&CupBcxIn, 1          , ncells_in-1);
  ierr = Allocate_RealArray_2D(&CupBcxBc, 1          , ncells_bc  );
  
  ierr = Allocate_RealArray_2D(&CdnInxIn, ncells_in-1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&CdnInxBc, ncells_in-1, ncells_bc  );
  ierr = Allocate_RealArray_2D(&CdnBcxIn, 1          , ncells_in-1);
  ierr = Allocate_RealArray_2D(&CdnBcxBc, 1          , ncells_bc  );
  
  ierr = Allocate_RealArray_2D(&AInxIn, ncells_in-1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&BInxIn, ncells_in-1, ncells_in  );
  
  ierr = Allocate_RealArray_2D(&AInxIninvBInxIn, ncells_in-1, ncells_in);
  
  ierr = Allocate_RealArray_1D(&AInxIn_1d                      , (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&lapack_mem_1d                  , (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&BInxIn_1d                      , (ncells_in-1)* ncells_in   );
  ierr = Allocate_RealArray_1D(&AInxIninv_1d                   , (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&AInxIninvBInxIn_1d             , (ncells_in-1)* ncells_in   );
  ierr = Allocate_RealArray_1D(&CupInxIn_1d                    , (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&CupInxIntimesAInxIninvBInxIn_1d, (ncells_in-1)*(ncells_in)  );
  ierr = Allocate_RealArray_1D(&CupBcxIn_1d                    , (1          )*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&CdnBcxIn_1d                    , (1          )*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&CupBcxIntimesAInxIninvBInxIn_1d, (1          )*(ncells_in)  );
  ierr = Allocate_RealArray_1D(&CdnBcxIntimesAInxIninvBInxIn_1d, (1          )*(ncells_in)  );
  
  ierr = Allocate_RealArray_1D(&DInxBc_1d                      , (ncells_in-1)* ncells_bc   );
  ierr = Allocate_RealArray_1D(&AInxIninvDInxBc_1d             , (ncells_in-1)* ncells_bc   );
  ierr = Allocate_RealArray_1D(&CupInxIntimesAInxIninvDInxBc_1d, (ncells_in-1)*(ncells_bc)  );
  ierr = Allocate_RealArray_1D(&CupBcxIntimesAInxIninvDInxBc_1d, (1          )*(ncells_bc)  );
  ierr = Allocate_RealArray_1D(&CdnBcxIntimesAInxIninvDInxBc_1d, (1          )*(ncells_bc)  );
  
  for (i=0; i<ncells_in; i++) {
    icell    = vertex->internal_cell_ids[i];
    isubcell = vertex->subcell_ids[i];
    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);
    
    Fup[i][i  ] = Gmatrix[0][0] + Gmatrix[0][1];
    Cup[i][i  ] = Gmatrix[0][0];
    Cup[i][i+1] = Gmatrix[0][1];
    
    Fdn[i+1][i  ] = Gmatrix[1][0] + Gmatrix[1][1];
    Cdn[i+1][i  ] = Gmatrix[1][0];
    Cdn[i+1][i+1] = Gmatrix[1][1];
  }
  
  i = 0        ; for (j=0; j<ncells_in; j++) FupBcxIn[0][j] = Fup[i][j];
  i = ncells_in; for (j=0; j<ncells_in; j++) FdnBcxIn[0][j] = Fdn[i][j];
  
  CupBcxBc[0][0] = Cup[0][0];
  CupBcxBc[0][1] = Cup[0][ncells_in];
  CdnBcxBc[0][0] = Cdn[ncells_in][0];
  CdnBcxBc[0][1] = Cdn[ncells_in][ncells_in];
  
  for (i=0; i<ncells_in-1; i++) {
    CupInxBc[i][0] = Cup[i+1][0        ];
    CupInxBc[i][1] = Cup[i+1][ncells_in];
    CdnInxBc[i][0] = Cdn[i+1][0        ];
    CdnInxBc[i][1] = Cdn[i+1][ncells_in];
    
    for (j=0; j<ncells_in-1; j++) {
      CupInxIn[i][j] = Cup[i+1][j+1];
      CdnInxIn[i][j] = Cdn[i+1][j+1];
    }
    
    for (j=0; j<ncells_in; j++) {
      FupInxIn[i][j] = Fup[i+1][j];
      FdnInxIn[i][j] = Fdn[i+1][j];
    }
    
  }
  
  idx = 0;
  for (PetscInt j=0; j<ncells_in-1; j++){
    for (PetscInt i=0; i<ncells_in-1; i++){
      AInxIn[i][j]     = -CupInxIn[i][j] + CdnInxIn[i][j];
      AInxIn_1d[idx]   = -CupInxIn[i][j] + CdnInxIn[i][j];
      CupInxIn_1d[idx] = CupInxIn[i][j];
      idx++;
    }
  }
  
  idx = 0;
  i   = 0;
  for (PetscInt j=0; j<ncells_in-1; j++){
    CupBcxIn_1d[idx] = Cup[i][j+1];
  }
  
  idx = 0;
  i   = ncells_in;
  for (PetscInt j=0; j<ncells_in-1; j++){
    CdnBcxIn_1d[idx] = Cdn[i][j+1];
  }
  
  idx = 0;
  for (PetscInt j=0; j<ncells_bc; j++){
    for (PetscInt i=0; i<ncells_in-1; i++){
      //DInxBc[i][j]     = CupInxBc[i][j] - CdnInxBc[i][j];
      DInxBc_1d[idx]   = CupInxBc[i][j] - CdnInxBc[i][j];
      idx++;
    }
  }
  
  idx = 0;
  for (PetscInt j=0; j<ncells_in; j++){
    for (PetscInt i=0; i<ncells_in-1; i++){
      BInxIn[i][j]     = -FupInxIn[i][j] + FdnInxIn[i][j];
      BInxIn_1d[idx]   = -FupInxIn[i][j] + FdnInxIn[i][j];
      idx++;
    }
  }
  
  n = ncells_in-1; m = ncells_in-1;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);
  
  LAPACKgetrf_(&m, &n, AInxIn_1d, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Bad LU factorization");

  ierr = PetscMemcpy(AInxIninvBInxIn_1d, BInxIn_1d,sizeof(PetscScalar)*(n*(m+1)));CHKERRQ(ierr); // AinvB in col major
  ierr = PetscMemcpy(AInxIninv_1d      , AInxIn_1d,sizeof(PetscScalar)*(n*m    ));CHKERRQ(ierr); // AinvB in col major

  ierr = PetscMemcpy(AInxIninvDInxBc_1d, DInxBc_1d,sizeof(PetscScalar)*(ncells_bc*(ncells_in-1)));CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1) by back-substitution
  PetscInt nn = n*n;
  LAPACKgetri_(&n, AInxIninv_1d, &n, pivots, lapack_mem_1d, &nn, &info);

  // Compute (Ainv*B)
  m = ncells_in-1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, AInxIninv_1d, &m, BInxIn_1d, &k, &zero, AInxIninvBInxIn_1d, &m);

  // Compute (Ainv*D)
  m = ncells_in-1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, AInxIninv_1d, &m, DInxBc_1d, &k, &zero, AInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*B)
  m = ncells_in-1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupInxIn_1d, &m, AInxIninvBInxIn_1d, &k, &zero, CupInxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*B) for up boundary flux
  m = 1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupBcxIn_1d, &m, AInxIninvBInxIn_1d, &k, &zero, CupBcxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*B) for down boundary flux
  m = 1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CdnBcxIn_1d, &m, AInxIninvBInxIn_1d, &k, &zero, CdnBcxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*D)
  m = ncells_in-1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupInxIn_1d, &m, AInxIninvDInxBc_1d, &k, &zero, CupInxIntimesAInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*D) for up boundary flux
  m = 1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupBcxIn_1d, &m, AInxIninvDInxBc_1d, &k, &zero, CupBcxIntimesAInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*D) for down boundary flux
  m = 1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CdnBcxIn_1d, &m, AInxIninvDInxBc_1d, &k, &zero, CdnBcxIntimesAInxIninvDInxBc_1d, &m);

  idx = 0;
  for (j=0; j<ncells_in; j++){
    for (i=0; i<ncells_in-1; i++){
      tdy->Trans[vertex_id][i][j] = CupInxIntimesAInxIninvBInxIn_1d[idx] - FupInxIn[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells_bc; j++){
    for (i=0; i<ncells_in-1; i++){
      tdy->Trans[vertex_id][i][j+ncells_in] = CupInxIntimesAInxIninvDInxBc_1d[idx] + CupInxBc[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells_in; j++){
    i = ncells_in-1; tdy->Trans[vertex_id][i][j] = CupBcxIntimesAInxIninvBInxIn_1d[idx] - FupBcxIn[0][j];
    i = ncells_in  ; tdy->Trans[vertex_id][i][j] = CdnBcxIntimesAInxIninvBInxIn_1d[idx] - FdnBcxIn[0][j];
    idx++;
  }

  idx = 0;
  for (j=0; j<ncells_bc; j++){
    i = ncells_in-1; tdy->Trans[vertex_id][i][j+ncells_in] = CupBcxIntimesAInxIninvDInxBc_1d[idx] + CupBcxBc[0][j];
    i = ncells_in  ; tdy->Trans[vertex_id][i][j+ncells_in] = CdnBcxIntimesAInxIninvDInxBc_1d[idx] + CdnBcxBc[0][j];
    idx++;
  }


  ierr = Deallocate_RealArray_2D(Gmatrix, ndim);

  ierr = Deallocate_RealArray_2D(Fup, ncells_in+ncells_bc);
  ierr = Deallocate_RealArray_2D(Cup, ncells_in+ncells_bc);
  ierr = Deallocate_RealArray_2D(Fdn, ncells_in+ncells_bc);
  ierr = Deallocate_RealArray_2D(Cdn, ncells_in+ncells_bc);

  ierr = Deallocate_RealArray_2D(FupInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(FupBcxIn, 1          );
  ierr = Deallocate_RealArray_2D(FdnInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(FdnBcxIn, 1          );

  ierr = Deallocate_RealArray_2D(CupInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CupInxBc, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CupBcxIn, 1          );
  ierr = Deallocate_RealArray_2D(CupBcxBc, 1          );

  ierr = Deallocate_RealArray_2D(CdnInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CdnInxBc, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CdnBcxIn, 1          );
  ierr = Deallocate_RealArray_2D(CdnBcxBc, 1          );

  ierr = Deallocate_RealArray_2D(AInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(BInxIn, ncells_in-1);

  ierr = Deallocate_RealArray_2D(AInxIninvBInxIn, ncells_in-1);

  free(AInxIn_1d                      );
  free(lapack_mem_1d                  );
  free(BInxIn_1d                      );
  free(AInxIninv_1d                   );
  free(AInxIninvBInxIn_1d             );
  free(CupInxIn_1d                    );
  free(CupInxIntimesAInxIninvBInxIn_1d);
  free(CupBcxIn_1d                    );
  free(CdnBcxIn_1d                    );
  free(CupBcxIntimesAInxIninvBInxIn_1d);
  free(CdnBcxIntimesAInxIninvBInxIn_1d);
  free(DInxBc_1d                      );
  free(AInxIninvDInxBc_1d             );
  free(CupInxIntimesAInxIninvDInxBc_1d);
  free(CupBcxIntimesAInxIninvDInxBc_1d);
  free(CdnBcxIntimesAInxIninvDInxBc_1d);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix(DM dm, TDy tdy) {

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    vertex = &vertices[ivertex];
    if (vertex->num_boundary_cells == 0) {

      ierr = ComputeTransmissibilityMatrixForInternalVertex(tdy, vertex, cells); CHKERRQ(ierr);
    } else {
      if (vertex->num_internal_cells > 1) {
        ierr = ComputeTransmissibilityMatrixForBoundaryVertex(tdy, vertex, cells); CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOInitialize(TDy tdy){

  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       dim;
  DM             dm;

  PetscFunctionBegin;

  dm = tdy->dm;

  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (dim != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"TDyMPFAOInitialize only support 2D grids");

  tdy->mesh = (TDy_mesh *) malloc(sizeof(TDy_mesh));

  ierr = AllocateMemoryForMesh(dm, tdy->mesh); CHKERRQ(ierr);

  ierr = Allocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells, tdy->mesh->num_vertices, 3, 3);CHKERRQ(ierr);
  ierr = Allocate_RealArray_3D(&tdy->Trans       , tdy->mesh->num_vertices, 5, 5);CHKERRQ(ierr);

  ierr = BuildTwoDimMesh(dm, tdy); CHKERRQ(ierr);

  ierr = ComputeGMatrix(dm, tdy); CHKERRQ(ierr);

  ierr = ComputeTransmissibilityMatrix(dm, tdy); CHKERRQ(ierr);

  //ierr = OutputTwoDimMesh(dm, tdy);

  /* Setup the section, 1 dof per cell */
  PetscSection sec;
  PetscInt p, pStart, pEnd;
  ierr = PetscSectionCreate(comm, &sec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0, 1);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++){
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view");CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem(TDy tdy,Mat K,Vec F){

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges, *edge;
  PetscInt       ivertex, icell, icell_from, icell_to, isubcell;
  PetscInt       icol, row, col, vertex_id, iedge;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscReal      sign;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = MatZeroEntries(K);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

  
  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;

    if (vertex->num_boundary_cells == 0) {
      for (icell=0; icell<vertex->num_internal_cells; icell++){
       
        if (icell==0) edge = &edges[vertex->edge_ids[vertex->num_internal_cells-1]];
        else          edge = &edges[vertex->edge_ids[icell-1]];

        icell_from = edge->cell_ids[0];
        icell_to   = edge->cell_ids[1];

        
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = vertex->internal_cell_ids[icol];
          value = tdy->Trans[vertex_id][icell][icol];
          row = icell_from; ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
          row = icell_to  ; ierr = MatSetValue(K, row, col,  value, ADD_VALUES);CHKERRQ(ierr);
        }
      }

    } else {

      // Vertex is on the boundary

      PetscScalar pBoundary[4];
      PetscInt cell_ids_from_to[4][2];
      PetscInt numBoundary;
      
      // For boundary edges, save following information:
      //  - Dirichlet pressure value
      //  - Cell IDs connecting the boundary edge in the direction of unit normal
      
      numBoundary = 0;
      for (iedge=0; iedge<vertex->num_edges; iedge++) {
        
        if (iedge==0) edge = &edges[vertex->edge_ids[vertex->num_edges-1]];
        else          edge = &edges[vertex->edge_ids[iedge-1]];
        
        if (edge->is_internal == 0) {
          
          PetscInt f;
          f = edge->id + fStart;
          (*tdy->dirichlet)(&(tdy->X[f*dim]), &pBoundary[numBoundary]);
          cell_ids_from_to[numBoundary][0] = edge->cell_ids[0];
          cell_ids_from_to[numBoundary][1] = edge->cell_ids[1];
          numBoundary++;
        }
      }

      if (vertex->num_internal_cells > 1) {


        // and vector
        for (icell=0; icell<vertex->num_internal_cells-1; icell++){
          iedge = vertex->edge_ids[icell];
          
          edge = &edges[vertex->edge_ids[icell]];

          icell_from = edge->cell_ids[0];
          icell_to   = edge->cell_ids[1];

          row = icell_from;
          if (row>-1) {
            for (icol=0; icol<vertex->num_internal_cells; icol++) {
              col   = vertex->internal_cell_ids[icol];
              value = tdy->Trans[vertex_id][icell][icol];
              ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
            }

            for (icol=0; icol<vertex->num_boundary_cells; icol++) {
              value = tdy->Trans[vertex_id][icell][icol + vertex->num_internal_cells] * pBoundary[icol];
              ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
            }
          }

          row = icell_to;
          if (row>-1) {
            for (icol=0; icol<vertex->num_internal_cells; icol++) {
              col = vertex->internal_cell_ids[icol];
              value = tdy->Trans[vertex_id][icell][icol];
              ierr = MatSetValue(K, row, col,  value, ADD_VALUES);CHKERRQ(ierr);
            }

            for (icol=0; icol<vertex->num_boundary_cells; icol++) {
              value = tdy->Trans[vertex_id][icell][icol + vertex->num_internal_cells] * pBoundary[icol];
              ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
            }
          }
        }

        // For fluxes through boundary edges, only add contribution to the vector
        for (icell=0; icell<vertex->num_boundary_cells; icell++){
          row = cell_ids_from_to[icell][0];
          if (row>-1) {
            for (icol=0; icol<vertex->num_boundary_cells; icol++) {
              value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol + vertex->num_internal_cells] * pBoundary[icol];
              ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
            }
            
            for (icol=0; icol<vertex->num_internal_cells; icol++){
              col   = vertex->internal_cell_ids[icol];
              value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol];
              ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
            }
          }

          row = cell_ids_from_to[icell][1];
          if (row>-1) {
            for (icol=0; icol<vertex->num_boundary_cells; icol++) {
              value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol + vertex->num_internal_cells] * pBoundary[icol];
              ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
            }
            for (icol=0; icol<vertex->num_internal_cells; icol++){
              col   = vertex->internal_cell_ids[icol];
              value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol];
              ierr = MatSetValue(K, row, col,  value, ADD_VALUES); CHKERRQ(ierr);
            }
          }
        }

      } else {
        icell    = vertex->internal_cell_ids[0];
        isubcell = vertex->subcell_ids[0];
        ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);
        value = 0.0;
        for (PetscInt i=0; i<dim; i++) {
          row = cell_ids_from_to[i][0];
          if (row>-1) sign = -1.0;
          else        sign = +1.0;
          for (PetscInt j=0; j<dim; j++) {
            value += sign*Gmatrix[i][j];
          }
        }
        row = icell; col = icell;
        ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);


        // For fluxes through boundary edges, only add contribution to the vector
        for (icell=0; icell<vertex->num_boundary_cells; icell++){
          row = cell_ids_from_to[icell][0];
          if (row>-1) {
            for (icol=0; icol<vertex->num_boundary_cells; icol++) {
              value = Gmatrix[icell][icol] * pBoundary[icol];
              ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
            }
          }
          
          row = cell_ids_from_to[icell][1];
          if (row>-1) {
            for (icol=0; icol<vertex->num_boundary_cells; icol++) {
              value = Gmatrix[icell][icol] * pBoundary[icol];
              ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
            }
          }
          
        }

      }
    }
  }

  PetscReal f;
  if (tdy->forcing) {
    for (icell=0; icell<tdy->mesh->num_cells; icell++){
      (*tdy->forcing)(&(tdy->X[icell*dim]), &f);
      value = f * cells[icell].volume;
      ierr = VecSetValue(F, icell, value, ADD_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);


}
/* -------------------------------------------------------------------------- */
