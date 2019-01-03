#include "tdycore.h"
#include "tdycoremesh.h"
#include "tdyutils.h"
#include <petscblaslapack.h>

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

  vertex->num_internal_cells = num_internal_cells;
  vertex->num_edges          = num_edges;
  vertex->num_boundary_cells = num_boundary_cells;

  vertex->edge_ids          = (PetscInt *) malloc(num_edges          * sizeof(PetscInt));
  vertex->internal_cell_ids = (PetscInt *) malloc(num_internal_cells * sizeof(PetscInt));
  vertex->subcell_ids       = (PetscInt *) malloc(num_internal_cells * sizeof(PetscInt));

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
PetscErrorCode AllocateMemoryForMesh(
    PetscInt cNum,
    PetscInt eNum,
    PetscInt vNum,
    PetscInt nverts_per_cell,
    PetscInt ncells_per_edge,
    TDy_mesh *mesh) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

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

PetscErrorCode TDyMPFAOInitialize(DM dm,TDy tdy){

  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       dim;

  PetscInt nverts_per_cell;
  PetscInt ncells_per_edge;
  PetscInt cStart, cEnd, cNum;
  PetscInt vStart, vEnd, vNum;
  PetscInt eStart, eEnd, eNum;

  PetscFunctionBegin;

  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  if (dim != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"TDyMPFAOInitialize only support 2D grids");

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

  tdy->mesh = (TDy_mesh *) malloc(sizeof(TDy_mesh));

  ierr = AllocateMemoryForMesh(cNum, eNum, vNum, nverts_per_cell, ncells_per_edge, tdy->mesh); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
