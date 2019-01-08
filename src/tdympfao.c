#include "tdycore.h"
#include "tdycoremesh.h"
#include "tdyutils.h"
#include <petscblaslapack.h>

/* ---------------------------------------------------------------- */

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start, PetscInt end){
  return (closure >= start) && (closure < end);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Allocate_RealArray_4D(PetscReal *****array_4D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4){

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

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Initialize_IntegerArray_1D(PetscInt *array_1D, PetscInt ndim_1, PetscInt init_value){

  PetscFunctionBegin;

  for(int i=0; i<ndim_1; i++)
    array_1D[i] = init_value;

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
PetscErrorCode Save2DMeshGeometricAttributes(DM dm, TDy tdy) {

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

PetscErrorCode Save2DMeshConnectivityInfo(DM dm, TDy tdy) {

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

  // vertex--to--edge
  for (int v=vStart; v<vEnd; v++){
    ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, v, &supportSize); CHKERRQ(ierr);
    ivertex = v - vStart;
    vertices[ivertex].num_edges = supportSize;
    for (int s=0; s<supportSize; s++) {
      iedge = support[s] - eStart;
      vertices[ivertex].edge_ids[s] = iedge;
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

  // compute angle to cell centroid w.r.t. vertix
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

  // compute angle to face centroid w.r.t. vertix
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

PetscErrorCode UpdateCellOrientation2DMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDy_mesh       *mesh;
  PetscInt       vStart, vEnd;
  TDy_vertex     *vertices;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  vertices = mesh->vertices;

  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd);CHKERRQ(ierr);

  for (ivertex=0; ivertex<vEnd-vStart; ivertex++){

    if (vertices[ivertex].num_internal_cells > 1) {
      ierr = UpdateCellOrientationAroundAVertex(tdy, ivertex); CHKERRQ(ierr);
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
        subcell->variable_continuity_coordinates[0].X[d] = cp_dn[d];
        subcell->variable_continuity_coordinates[1].X[d] = cp_up[d];
      }

      // compute the 'direction' of nu-vector
      ierr = ComputeRightNormalVector(cp_up   , cell_cen, dim, nu_vec_up); CHKERRQ(ierr);
      ierr = ComputeRightNormalVector(cell_cen, cp_dn   , dim, nu_vec_dn); CHKERRQ(ierr);

      // compute length of nu-vectors
      ierr = ComputeLength(cp_up, cell_cen, dim, &len_up); CHKERRQ(ierr);
      ierr = ComputeLength(cp_up, cell_cen, dim, &len_dn); CHKERRQ(ierr);

      // save nu-vectors
      // note: length of nu-vectors is equal to length of edge diagonally
      //       opposite to the vector
      for (d=0; d<dim; d++) {
        subcell->nu_vector[0].V[d] = nu_vec_dn[d]*len_up;
        subcell->nu_vector[1].V[d] = nu_vec_up[d]*len_dn;
      }

      ierr = ComputeAreaOf2DTriangle(cp_up, cell_cen, cp_dn, &subcell->volume);

    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode Build2DMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  ierr = Save2DMeshGeometricAttributes(dm, tdy); CHKERRQ(ierr);
  ierr = Save2DMeshConnectivityInfo(   dm, tdy); CHKERRQ(ierr);
  ierr = UpdateCellOrientation2DMesh(  dm, tdy); CHKERRQ(ierr);
  ierr = SetupSubcellsFor2DMesh     (  dm, tdy); CHKERRQ(ierr);

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

  for (icell=0; icell<mesh->num_cells*0+1; icell++) {

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
        nu_dn[d]    = subcell->nu_vector[0].V[d];
        nu_up[d]    = subcell->nu_vector[1].V[d];

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
      //                              | L_dn*n_dn | | K_xx   K_xy |  _             _
      // Gmatrix =        -1          |           | |             | |               |
      //             -----------      |           | |             | | nu_up   nu_dn |
      //              2*A_{subcell}   | L_up*n_up | | K_yx   K_yy | |_             _|
      //                              |           | |             |
      //                              |_         _| |_           _|
      //
      ComputeEntryOfGMatrix(e_len_dn, n_dn, K, nu_dn, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][0][0]));
      ComputeEntryOfGMatrix(e_len_dn, n_dn, K, nu_up, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][0][1]));
      ComputeEntryOfGMatrix(e_len_dn, n_up, K, nu_dn, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][1][0]));
      ComputeEntryOfGMatrix(e_len_dn, n_up, K, nu_up, subcell->volume, dim, &(tdy->subc_Gmatrix[icell][isubcell][1][1]));

    }
  }

  PetscFunctionReturn(0);

}
/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOInitialize(DM dm,TDy tdy){

  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       dim;

  PetscFunctionBegin;

  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (dim != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"TDyMPFAOInitialize only support 2D grids");

  tdy->mesh = (TDy_mesh *) malloc(sizeof(TDy_mesh));

  ierr = AllocateMemoryForMesh(dm, tdy->mesh); CHKERRQ(ierr);

  ierr = Allocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells, tdy->mesh->num_vertices, 3, 3);CHKERRQ(ierr);

  ierr = Build2DMesh(dm, tdy); CHKERRQ(ierr);

  ierr = ComputeGMatrix(dm, tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
