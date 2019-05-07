#include <petsc.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymemoryimpl.h>
#include "tdyutils.h"

/* ---------------------------------------------------------------- */

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start,
                                PetscInt end) {
  return (closure >= start) && (closure < end);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForASubcell(
  TDy_subcell    *subcell,
  TDySubcellType subcell_type) {

  PetscFunctionBegin;

  PetscInt num_nu_vectors;
  PetscInt num_vertices;

  switch (subcell_type) {
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

  subcell->nu_vector                       = (TDy_vector *) malloc(
        num_nu_vectors * sizeof(TDy_vector    ));
  subcell->variable_continuity_coordinates = (TDy_coordinate *) malloc(
        num_nu_vectors * sizeof(TDy_coordinate));
  subcell->vertices_cordinates             = (TDy_coordinate *) malloc(
        num_vertices   * sizeof(TDy_coordinate));

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
  PetscInt       isubcell;
  PetscInt       num_subcells;
  TDy_subcell    *subcells;

  cell->is_local      = PETSC_FALSE;
  cell->num_vertices  = num_vertices;
  cell->num_edges     = num_edges;
  cell->num_neighbors = num_neighbors;

  cell->vertex_ids   = (PetscInt *) malloc(num_vertices  * sizeof(PetscInt));
  cell->edge_ids     = (PetscInt *) malloc(num_edges     * sizeof(PetscInt));
  cell->neighbor_ids = (PetscInt *) malloc(num_neighbors * sizeof(PetscInt));

  Initialize_IntegerArray_1D(cell->vertex_ids, num_vertices, -1);
  Initialize_IntegerArray_1D(cell->edge_ids, num_edges, -1);
  Initialize_IntegerArray_1D(cell->neighbor_ids, num_neighbors, -1);

  switch (subcell_type) {
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
  PetscInt       num_cells,
  PetscInt       nverts_per_cell,
  TDySubcellType subcell_type,
  TDy_cell       *cells) {

  PetscFunctionBegin;

  PetscInt icell;
  PetscErrorCode ierr;

  PetscInt num_vertices  = nverts_per_cell;
  PetscInt num_edges     = nverts_per_cell;
  PetscInt num_neighbors = nverts_per_cell;

  /* allocate memory for cells within the mesh*/
  for (icell=0; icell<num_cells; icell++) {
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

  vertex->is_local           = PETSC_FALSE;
  vertex->num_internal_cells = 0;
  vertex->num_edges          = num_edges;
  vertex->num_boundary_cells = 0;

  vertex->edge_ids          = (PetscInt *) malloc(num_edges          * sizeof(
                                PetscInt));
  vertex->internal_cell_ids = (PetscInt *) malloc(num_internal_cells * sizeof(
                                PetscInt));
  vertex->subcell_ids       = (PetscInt *) malloc(num_internal_cells * sizeof(
                                PetscInt));

  Initialize_IntegerArray_1D(vertex->edge_ids, num_edges, -1);
  Initialize_IntegerArray_1D(vertex->internal_cell_ids, num_internal_cells, -1);
  Initialize_IntegerArray_1D(vertex->subcell_ids, num_internal_cells, -1);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForVertices(
  PetscInt       num_vertices,
  PetscInt       nverts_per_cell,
  TDy_vertex     *vertices) {

  PetscFunctionBegin;

  PetscInt ivertex;
  PetscErrorCode ierr;

  PetscInt num_internal_cells = nverts_per_cell;
  PetscInt num_edges          = nverts_per_cell;
  PetscInt num_boundary_cells = 0;

  /* allocate memory for vertices within the mesh*/
  for  (ivertex=0; ivertex<num_vertices; ivertex++) {
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

  edge->is_local = PETSC_FALSE;

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

  PetscInt iedge;
  PetscErrorCode ierr;

  /* allocate memory for edges within the mesh*/
  for (iedge=0; iedge<num_edges; iedge++) {
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
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  cNum = cEnd - cStart;
  eNum = eEnd - eStart;
  vNum = vEnd - vStart;

  mesh->num_cells    = cNum;
  mesh->num_faces    = 0;
  mesh->num_edges    = eNum;
  mesh->num_vertices = vNum;

  mesh->cells    = (TDy_cell *) malloc(cNum * sizeof(TDy_cell   ));
  mesh->edges    = (TDy_edge *) malloc(eNum * sizeof(TDy_edge   ));
  mesh->vertices = (TDy_vertex *) malloc(vNum * sizeof(TDy_vertex ));

  ierr = AllocateMemoryForCells(cNum, nverts_per_cell, SUBCELL_QUAD_TYPE,
                                mesh->cells); CHKERRQ(ierr);
  ierr = AllocateMemoryForVertices(vNum, nverts_per_cell, mesh->vertices);
  CHKERRQ(ierr);
  ierr = AllocateMemoryForEdges(eNum, ncells_per_edge, mesh->edges);
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
  PetscInt       dim;
  PetscInt       cStart, cEnd;
  PetscInt       vStart, vEnd;
  PetscInt       eStart, eEnd;
  PetscInt       pStart, pEnd;
  PetscInt       icell, iedge, ivertex, ielement, d;
  PetscErrorCode ierr;

  dm       = tdy->dm;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  /* Determine the number of cells, edges, and vertices of the mesh */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 1, &eStart, &eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetChart(        dm, &pStart, &pEnd); CHKERRQ(ierr);

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

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
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode SaveTwoDimMeshConnectivityInfo(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
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

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

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

    for (i=0; i<closureSize*2; i+=2)  {

      if (IsClosureWithinBounds(closure[i], vStart,
                                vEnd)) { /* Is the closure a vertex? */
        ivertex = closure[i] - vStart;
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
        iedge = closure[i] - eStart;
        cells[icell].edge_ids[c2eCount] = iedge;
        for (j=0; j<2; j++) {
          if (edges[iedge].cell_ids[j] == -1) {
            edges[iedge].cell_ids[j] = icell;
            break;
          }
        }
        c2eCount++;
      }
    }
  }

  // edge--to--vertex
  for (e=eStart; e<eEnd; e++) {
    ierr = DMPlexGetConeSize(dm, e, &coneSize); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, e, &cone); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, e, &supportSize); CHKERRQ(ierr);
    iedge = e-eStart;

    if (supportSize == 1) edges[iedge].is_internal = PETSC_FALSE;
    else                  edges[iedge].is_internal = PETSC_TRUE;

    edges[iedge].vertex_ids[0] = cone[0]-vStart;
    edges[iedge].vertex_ids[1] = cone[1]-vStart;

    for (d=0; d<dim; d++) {
      v_1[d] = vertices[edges[iedge].vertex_ids[0]].coordinate.X[d];
      v_2[d] = vertices[edges[iedge].vertex_ids[1]].coordinate.X[d];
    }
    ierr = ComputeLength(v_1, v_2, dim, &(edges[iedge].length)); CHKERRQ(ierr);
  }

  // vertex--to--edge
  for (v=vStart; v<vEnd; v++) {
    ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, v, &supportSize); CHKERRQ(ierr);
    ivertex = v - vStart;
    vertices[ivertex].num_edges = supportSize;
    for (s=0; s<supportSize; s++) {
      iedge = support[s] - eStart;
      vertices[ivertex].edge_ids[s] = iedge;
      if (!edges[iedge].is_internal) vertices[ivertex].num_boundary_cells++;
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

PetscErrorCode SetupSubcellsForTwoDimMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
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

  dm       = tdy->dm;
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
PetscErrorCode BuildTwoDimMesh(TDy tdy) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  ierr = SaveTwoDimMeshGeometricAttributes(tdy); CHKERRQ(ierr);
  ierr = SaveTwoDimMeshConnectivityInfo(   tdy); CHKERRQ(ierr);
  ierr = UpdateCellOrientationAroundAVertexTwoDimMesh(tdy); CHKERRQ(ierr);
  ierr = SetupSubcellsForTwoDimMesh     (  tdy); CHKERRQ(ierr);
  ierr = UpdateCellOrientationAroundAEdgeTwoDimMesh(  tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
