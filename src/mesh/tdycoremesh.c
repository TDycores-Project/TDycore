#include <petsc.h>
#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdyregionimpl.h>

/* ---------------------------------------------------------------- */

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start,
                                PetscInt end) {
  return (closure >= start) && (closure < end);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForCells(
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

   cells->is_local = (PetscBool *)malloc(num_cells*sizeof(PetscBool));

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

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForSubcells(
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
  subcells->type = (TDySubcellType *)malloc(num_cells*num_subcells_per_cell*sizeof(TDySubcellType));

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

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForVertices(
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

  vertices->is_local = (PetscBool *)malloc(num_vertices*sizeof(PetscBool));

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

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForEdges(
  PetscInt num_edges,
  TDyCellType cell_type,
  TDyEdge *edges) {

  PetscFunctionBegin;

  PetscInt num_cells = GetNumCellsPerEdgeForCellType(cell_type);

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_1D(&edges->id,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->global_id,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->num_cells,num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&edges->vertex_ids,num_edges*2); CHKERRQ(ierr);

  edges->is_local = (PetscBool *)malloc(num_edges*sizeof(PetscBool));
  edges->is_internal = (PetscBool *)malloc(num_edges*sizeof(PetscBool));

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

/* -------------------------------------------------------------------------- */
PetscErrorCode AllocateMemoryForFaces(
  PetscInt num_faces,
  TDyCellType cell_type,
  TDyFace *faces) {

  PetscFunctionBegin;

  PetscInt num_cells    = GetNumCellsPerFaceForCellType(cell_type);
  PetscInt num_edges    = GetNumOfEdgesFormingAFaceForCellType(cell_type);
  PetscInt num_vertices = GetNumOfVerticesFormingAFaceForCellType(cell_type);

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_1D(&faces->id,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_vertices,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_edges,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->num_cells,num_faces); CHKERRQ(ierr);

  faces->is_local = (PetscBool *)malloc(num_faces*sizeof(PetscBool));
  faces->is_internal = (PetscBool *)malloc(num_faces*sizeof(PetscBool));

  ierr = TDyAllocate_IntegerArray_1D(&faces->vertex_offset,num_faces+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->cell_offset,num_faces+1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->edge_offset,num_faces+1); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_1D(&faces->cell_ids,num_faces*num_cells); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->edge_ids,num_faces*num_edges); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&faces->vertex_ids,num_faces*num_vertices); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&faces->area,num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyCoordinate_1D(num_faces, &faces->centroid); CHKERRQ(ierr);
  ierr = TDyAllocate_TDyVector_1D(num_faces, &faces->normal); CHKERRQ(ierr);

  for (PetscInt iface=0; iface<num_faces; iface++) {
    faces->id[iface] = iface;
    faces->is_local[iface]    = PETSC_FALSE;
    faces->is_internal[iface] = PETSC_FALSE;

    faces->num_edges[iface] = num_edges;

    faces->num_cells[iface] = 0;
    faces->num_vertices[iface] = 0;
  }

  for (PetscInt iface=0; iface<=num_faces; iface++) {
    faces->cell_offset[iface] = iface*num_cells;
    faces->edge_offset[iface] = iface*num_edges;
    faces->vertex_offset[iface] = iface*num_vertices;
  }

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyAllocateMemoryForMesh(TDy tdy) {

  PetscFunctionBegin;

  DM dm = tdy->dm;
  TDyMesh *mesh = tdy->mesh;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (dim!= 2 && dim!=3 ) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only 2D and 3D grids are supported");
  }

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  PetscInt num_cells = c_end - c_start;

  PetscInt e_start, e_end;
  ierr = DMPlexGetDepthStratum( dm, 1, &e_start, &e_end); CHKERRQ(ierr);
  PetscInt num_edges = e_end - e_start;

  PetscInt v_start, v_end;
  ierr = DMPlexGetDepthStratum( dm, 0, &v_start, &v_end); CHKERRQ(ierr);
  PetscInt num_vertices = v_end - v_start;

  PetscInt num_faces;
  if (dim == 3) {
    PetscInt f_start, f_end;
    ierr = DMPlexGetDepthStratum( dm, 2, &f_start, &f_end); CHKERRQ(ierr);
    num_faces = f_end - f_start;
  } else {
    num_faces = 0;
  }

  mesh->num_cells    = num_cells;
  mesh->num_faces    = num_faces;
  mesh->num_edges    = num_edges;
  mesh->num_vertices = num_vertices;

  tdy->maxClosureSize = 27*4*4;
  ierr = TDyAllocate_IntegerArray_1D(&tdy->closureSize, num_cells+num_faces+num_edges+num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&tdy->closure, num_cells+num_faces+num_edges+num_vertices, 2*tdy->maxClosureSize); CHKERRQ(ierr);
  ierr = TDySaveClosures(dm, tdy->closureSize, tdy->closure, &tdy->maxClosureSize); CHKERRQ(ierr);

  // compute number of vertices per grid cell
  PetscInt nverts_per_cell = TDyGetNumberOfCellVerticesWithClosures(dm, tdy->closureSize, tdy->closure);
  TDyCellType cell_type = GetCellType(dim, nverts_per_cell);

  ierr = AllocateMemoryForCells(num_cells, cell_type, &mesh->cells); CHKERRQ(ierr);
  ierr = AllocateMemoryForEdges(num_edges, cell_type, &mesh->edges); CHKERRQ(ierr);
  ierr = AllocateMemoryForFaces(num_faces, cell_type, &mesh->faces); CHKERRQ(ierr);

  PetscInt ncells_per_vertex = TDyMaxNumberOfCellsSharingAVertex(dm, tdy->closureSize, tdy->closure);
  PetscInt nfaces_per_vertex = TDyMaxNumberOfFacesSharingAVertex(dm, tdy->closureSize, tdy->closure);
  PetscInt nedges_per_vertex = TDyMaxNumberOfEdgesSharingAVertex(dm, tdy->closureSize, tdy->closure);
  tdy->ncv = ncells_per_vertex;
  tdy->nfv = nfaces_per_vertex;

  ierr = AllocateMemoryForVertices(num_vertices, ncells_per_vertex, nfaces_per_vertex, nedges_per_vertex, cell_type, &mesh->vertices); CHKERRQ(ierr);

  TDySubcellType subcell_type = GetSubcellTypeForCellType(cell_type);
  PetscInt num_subcells  = GetNumSubcellsForSubcellType(subcell_type);
  mesh->num_subcells = num_cells*num_subcells;
  ierr = AllocateMemoryForSubcells(num_cells, num_subcells, subcell_type, &mesh->subcells); CHKERRQ(ierr);

  ierr = TDyRegionCreate(&mesh->region_connected); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SaveMeshGeometricAttributes(TDy tdy) {

  PetscFunctionBegin;

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyEdge       *edges = &mesh->edges;
  TDyFace       *faces = &mesh->faces;
  PetscErrorCode ierr;

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);

  PetscInt e_start, e_end;
  ierr = DMPlexGetDepthStratum(dm, 1, &e_start, &e_end); CHKERRQ(ierr);

  PetscInt v_start, v_end;
  ierr = DMPlexGetDepthStratum(dm, 0, &v_start, &v_end); CHKERRQ(ierr);

  PetscInt p_start, pEnd;
  ierr = DMPlexGetChart(dm, &p_start, &pEnd); CHKERRQ(ierr);

  // Face indexes -- only relevant for 3D calculations.
  PetscInt dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  PetscInt f_start, f_end;
  if (dim == 3) {
    ierr = DMPlexGetDepthStratum( dm, 2, &f_start, &f_end); CHKERRQ(ierr);
  } else {
    f_start = 0; f_end = 0;
  }

  for (PetscInt ielement=p_start; ielement<pEnd; ielement++) {

    if (IsClosureWithinBounds(ielement, v_start, v_end)) { // is the element a vertex?
      PetscInt ivertex = ielement - v_start;
      for (PetscInt d=0; d<dim; d++) {
        vertices->coordinate[ivertex].X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, e_start,
                                     e_end)) { // is the element an edge?
      PetscInt iedge = ielement - e_start;
      for (PetscInt d=0; d<dim; d++) {
        edges->centroid[iedge].X[d] = tdy->X[ielement*dim + d];
        edges->normal[iedge].V[d]   = tdy->N[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, c_start,
                                     c_end)) { // is the element a cell?
      PetscInt icell = ielement - c_start;
      for (PetscInt d=0; d<dim; d++) {
        cells->centroid[icell].X[d] = tdy->X[ielement*dim + d];
      }
    } else if (IsClosureWithinBounds(ielement, f_start,
                                     f_end)) { // is the elment a face?
      PetscInt iface = ielement - f_start;
      for (PetscInt d=0; d<dim; d++) {
        faces->centroid[iface].X[d] = tdy->X[ielement*dim + d];
      }
      faces->area[iface] = tdy->V[ielement];
    }
  }

  // If we're using natural ordering, generate natural indices for each cell.
  PetscBool useNatural;
  ierr = DMGetUseNatural(dm, &useNatural); CHKERRQ(ierr);
  if (useNatural) {
    PetscInt num_fields;

    ierr = DMGetNumFields(dm, &num_fields); CHKERRQ(ierr);

    // Create the natural vector
    Vec natural;
    ierr = DMCreateGlobalVector(dm, &natural);
    PetscInt natural_size, cum_natural_size;
    ierr = VecGetLocalSize(natural, &natural_size);
    ierr = MPI_Scan(&natural_size, &cum_natural_size, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

    // Add entries in the natural vector
    PetscScalar *entries;
    ierr = VecGetArray(natural, &entries); CHKERRQ(ierr);
    for (PetscInt i = 0; i < natural_size; ++i) {
      if (i % num_fields == 0) {
        entries[i] = i + cum_natural_size/num_fields - natural_size/num_fields;
      }
      else {
        entries[i] = 0;
      }
    }
    ierr = VecRestoreArray(natural, &entries); CHKERRQ(ierr);

    // Map natural IDs in global order
    Vec global;
    ierr = DMCreateGlobalVector(dm, &global);CHKERRQ(ierr);
    ierr = DMPlexNaturalToGlobalBegin(dm, natural, global);CHKERRQ(ierr);
    ierr = DMPlexNaturalToGlobalEnd(dm, natural, global);CHKERRQ(ierr);

    // Map natural IDs in local order
    Vec local;
    ierr = DMCreateLocalVector(dm, &local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local); CHKERRQ(ierr);

    // Save natural IDs
    PetscInt local_size;
    ierr = VecGetLocalSize(local, &local_size);
    ierr = VecGetArray(local, &entries); CHKERRQ(ierr);
    for (PetscInt i = 0; i < local_size/num_fields; ++i) {
      cells->natural_id[i] = entries[i*num_fields];
    }
    ierr = VecRestoreArray(local, &entries); CHKERRQ(ierr);

    // Cleanup
    ierr = VecDestroy(&natural); CHKERRQ(ierr);
    ierr = VecDestroy(&global); CHKERRQ(ierr);
    ierr = VecDestroy(&local); CHKERRQ(ierr);

    char file[PETSC_MAX_PATH_LEN];
    PetscBool connected_region;
    ierr = PetscOptionsGetString(NULL,NULL,"-tdy_connected_region",file,sizeof(file),
                                 &connected_region); CHKERRQ(ierr);
    if (connected_region) {

      PetscViewer viewer;
      Vec region_id_nat_idx;

      ierr = VecCreate(PETSC_COMM_WORLD,&region_id_nat_idx); CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
      ierr = VecLoad(region_id_nat_idx,viewer); CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

      // Map natural IDs in global order
      ierr = DMCreateGlobalVector(dm, &global);CHKERRQ(ierr);
      ierr = DMPlexNaturalToGlobalBegin(dm, region_id_nat_idx, global);CHKERRQ(ierr);
      ierr = DMPlexNaturalToGlobalEnd(dm, region_id_nat_idx, global);CHKERRQ(ierr);

      // Map natural IDs in local order
      ierr = DMCreateLocalVector(dm, &local);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local); CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local); CHKERRQ(ierr);

      // Save the region ids
      ierr = VecGetLocalSize(local, &local_size);
      ierr = VecGetArray(local, &entries); CHKERRQ(ierr);

      PetscInt ncells = local_size/num_fields;
      PetscInt cell_ids[ncells];

      for (PetscInt i=0; i<ncells; i++) {
        cell_ids[i] = entries[i*num_fields];
      }

      ierr = TDyRegionAddCells(ncells, cell_ids, &mesh->region_connected);

      ierr = VecRestoreArray(local, &entries); CHKERRQ(ierr);

      // Cleanup
      ierr = VecDestroy(&region_id_nat_idx); CHKERRQ(ierr);
      ierr = VecDestroy(&global); CHKERRQ(ierr);
      ierr = VecDestroy(&local); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode SaveMeshConnectivityInfo(TDy tdy) {

  PetscFunctionBegin;

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyEdge       *edges = &mesh->edges;
  TDyFace       *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscInt nverts_per_cell = tdy->ncv;

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);

  PetscInt e_start, e_end;
  ierr = DMPlexGetDepthStratum( dm, 1, &e_start, &e_end); CHKERRQ(ierr);

  PetscInt v_start, v_end;
  ierr = DMPlexGetDepthStratum( dm, 0, &v_start, &v_end); CHKERRQ(ierr);

  PetscInt p_start, pEnd;
  ierr = DMPlexGetChart(dm, &p_start, &pEnd); CHKERRQ(ierr);

  // Faces -- only relevant in 3D calculations.
  PetscInt dim, f_start, f_end;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  if (dim == 3) {
    ierr = DMPlexGetDepthStratum( dm, 2, &f_start, &f_end); CHKERRQ(ierr);
  } else {
    f_start = 0; f_end = 0;
  }

  // cell--to--vertex
  // edge--to--cell
  // cell--to--edge
  // edge--to--cell
  for (PetscInt icell=c_start; icell<c_end; icell++) {

    PetscInt c2v_count, c2e_count, c2f_count;
    c2v_count = 0;
    c2e_count = 0;
    c2f_count = 0;

    for (PetscInt i=0; i<tdy->closureSize[icell]*2; i+=2)  {

      if (IsClosureWithinBounds(tdy->closure[icell][i], v_start,
                                v_end)) { /* Is the closure a vertex? */
        PetscInt ivertex = tdy->closure[icell][i] - v_start;
        PetscInt cOffsetVert = cells->vertex_offset[icell];
        cells->vertex_ids[cOffsetVert + c2v_count] = ivertex ;

        PetscInt vOffsetCell = vertices->internal_cell_offset[ivertex];
        PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

        PetscInt found = PETSC_FALSE;
        for (PetscInt j=0; j<nverts_per_cell; j++) {
          if (vertices->internal_cell_ids[vOffsetCell + j] == -1) {
            vertices->num_internal_cells[ivertex]++;
            vertices->internal_cell_ids[vOffsetCell + j] = icell;
            vertices->subcell_ids[vOffsetSubcell + j]    = c2v_count;
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,
                  "No empty space found in the vertex to save cell");
        }
        c2v_count++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], e_start,
                                       e_end)) { /* Is the closure an edge? */
        PetscInt iedge = tdy->closure[icell][i] - e_start;
        PetscInt cOffsetEdge = cells->edge_offset[icell];
        cells->edge_ids[cOffsetEdge + c2e_count] = iedge;
        PetscInt eOffsetCell = edges->cell_offset[iedge];
        for (PetscInt j=0; j<2; j++) {
          if (edges->cell_ids[eOffsetCell + j] == -1) {
            edges->cell_ids[eOffsetCell + j] = icell;
            break;
          }
        }

        c2e_count++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], f_start,
                                       f_end)) { /* Is the closure a face? */
        PetscInt iface = tdy->closure[icell][i] - f_start;
        PetscInt cOffsetFace = cells->face_offset[icell];
        PetscInt fOffsetCell = faces->cell_offset[iface];
        cells->face_ids[cOffsetFace + c2f_count] = iface;
        for (PetscInt j=0; j<2; j++) {
          if (faces->cell_ids[fOffsetCell + j] < 0) {
            faces->cell_ids[fOffsetCell + j] = icell;
            faces->num_cells[iface]++;
            break;
          }
        }
        c2f_count++;
      }
    }

  }


  // edge--to--vertex
  for (PetscInt e=e_start; e<e_end; e++) {
    const PetscInt* cone;
    PetscInt cone_size;
    ierr = DMPlexGetConeSize(dm, e, &cone_size); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, e, &cone); CHKERRQ(ierr);

    PetscInt support_size;
    ierr = DMPlexGetSupportSize(dm, e, &support_size); CHKERRQ(ierr);
    PetscInt iedge = e-e_start;

    if (support_size == 1) {
      edges->is_internal[iedge] = PETSC_FALSE;
    } else {
      edges->is_internal[iedge] = PETSC_TRUE;
    }

    edges->vertex_ids[iedge*2 + 0] = cone[0]-v_start;
    edges->vertex_ids[iedge*2 + 1] = cone[1]-v_start;

    PetscReal v_1[3], v_2[3];
    ierr = TDyVertex_GetCoordinate(vertices, edges->vertex_ids[iedge*2 + 0], dim, &v_1[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, edges->vertex_ids[iedge*2 + 1], dim, &v_2[0]); CHKERRQ(ierr);

    for (PetscInt d=0; d<dim; d++) {
      edges->centroid[iedge].X[d] = (v_1[d] + v_2[d])/2.0;
    }

    ierr = TDyComputeLength(v_1, v_2, dim, &(edges->length[iedge])); CHKERRQ(ierr);
  }

  // vertex--to--edge
  for (PetscInt v=v_start; v<v_end; v++) {
    const PetscInt *support;
    PetscInt support_size;
    ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, v, &support_size); CHKERRQ(ierr);
    PetscInt ivertex = v - v_start;
    vertices->num_edges[ivertex] = support_size;
    for (PetscInt s=0; s<support_size; s++) {
      PetscInt iedge = support[s] - e_start;
      PetscInt vOffsetEdge = vertices->edge_offset[ivertex];
      vertices->edge_ids[vOffsetEdge + s] = iedge;
      if (!edges->is_internal[iedge]) vertices->num_boundary_faces[ivertex]++;
    }
  }

  for (PetscInt f=f_start; f<f_end; f++){
    PetscInt iface = f-f_start;
    PetscInt fOffsetEdge = faces->edge_offset[iface];
    PetscInt fOffsetVertex = faces->vertex_offset[iface];

    // face--to--edge
    const PetscInt* cone;
    PetscInt cone_size;
    ierr = DMPlexGetConeSize(dm, f, &cone_size); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, f, &cone); CHKERRQ(ierr);

    for (PetscInt c=0;c<cone_size;c++) {
      faces->edge_ids[fOffsetEdge + c] = cone[c]-e_start;
    }

    // face--to-vertex
    for (PetscInt i=0; i<tdy->closureSize[f]*2; i+=2)  {
      if (IsClosureWithinBounds(tdy->closure[f][i],v_start,v_end)) {
        faces->vertex_ids[fOffsetVertex + faces->num_vertices[iface]] = tdy->closure[f][i]-v_start;
        faces->num_vertices[iface]++;

        PetscBool found = PETSC_FALSE;
        PetscInt ivertex = tdy->closure[f][i]-v_start;
        PetscInt vOffsetFace = vertices->face_offset[ivertex];
        for (PetscInt ii=0; ii<vertices->num_faces[ivertex]; ii++) {
          if (vertices->face_ids[vOffsetFace+ii] == iface) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          vertices->face_ids[vOffsetFace+vertices->num_faces[ivertex]] = iface;
          vertices->subface_ids[vOffsetFace+vertices->num_faces[ivertex]] = faces->num_vertices[iface] - 1;
          vertices->num_faces[ivertex]++;
        }
      }
    }

    // face--to--cell
    const PetscInt *support;
    PetscInt support_size;
    ierr = DMPlexGetSupportSize(dm, f, &support_size); CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, f, &support); CHKERRQ(ierr);

    if (support_size == 2) {
      faces->is_internal[iface] = PETSC_TRUE;
    } else {
      faces->is_internal[iface] = PETSC_FALSE;
    }

    for (PetscInt s=0; s<support_size; s++) {
      PetscInt icell = support[s] - c_start;
        PetscBool found = PETSC_FALSE;
        PetscInt cOffsetFace = cells->face_offset[icell];
        for (PetscInt ii=0; ii<cells->num_faces[icell]; ii++) {
          if (cells->face_ids[cOffsetFace+ii] == f-f_start) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          cells->face_ids[cOffsetFace + cells->num_faces[icell]] = f-f_start;
          cells->num_faces[icell]++;
          found = PETSC_TRUE;
        }
    }

    // If it is a boundary face, increment the number of boundary
    // cells by 1 for all vertices that form the face
    if (!faces->is_internal[iface]) {
      for (PetscInt v=0; v<faces->num_vertices[iface]; v++) {
        PetscInt vertex_id = faces->vertex_ids[fOffsetVertex + v];
        PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[vertex_id];

        vertices->boundary_face_ids[vOffsetBoundaryFace + vertices->num_boundary_faces[vertex_id] ] = iface;
        vertices->num_boundary_faces[vertex_id]++;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts an integer datatype of TDy mesh element in compressed format
///
/// @param [in] tdy                  A TDy struct
/// @param [in] num_elements         Number of elements
/// @param [in] default_offset_size  Default offset size for elements
/// @param [in] update_offset        Determines if subelement_offset should be updated
/// @param [inout] subelement_num    Number of subelements for a given element
/// @param [inout] subelement_offset Offset of subelements for a given element
/// @param [inout] subelement_id     Subelement ids
/// @returns 0                       on success, or a non-zero error code on failure
PetscErrorCode ConvertMeshElementToCompressedFormatIntegerValues(TDy tdy, PetscInt num_element, PetscInt default_offset_size, PetscInt update_offset,
  PetscInt **subelement_num, PetscInt **subelement_offset, PetscInt **subelement_id) {

  PetscFunctionBegin;

  PetscInt count = 0, new_offset = 0;

  count = (*subelement_num)[0];
  for (PetscInt ielem=1; ielem<num_element; ielem++) {

    new_offset += (*subelement_num)[ielem-1];
    PetscInt old_offset = (*subelement_offset)[ielem];

    for (PetscInt isubelem=0; isubelem<(*subelement_num)[ielem]; isubelem++){

      (*subelement_id)[new_offset + isubelem] = (*subelement_id)[old_offset + isubelem];
      count++;

    }
    if (update_offset) {
      (*subelement_offset)[ielem] = new_offset;
    }
  }

  if (update_offset) {
    new_offset += (*subelement_num)[num_element-1];
    (*subelement_offset)[num_element] = new_offset;
  }

  for (PetscInt ii = count; ii < default_offset_size*num_element; ii++) {
    (*subelement_id)[ii] = -1;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts TDyVector datatype of TDy mesh element in compressed format
///
/// @param [in] tdy                  A TDy struct
/// @param [in] num_elements         Number of elements
/// @param [in] default_offset_size  Default offset size for elements
/// @param [in] update_offset        Determines if subelement_offset should be updated
/// @param [inout] subelement_num    Number of subelements for a given element
/// @param [inout] subelement_offset Offset of subelements for a given element
/// @param [inout] subelement_id     Subelement ids
/// @returns 0                       on success, or a non-zero error code on failure
PetscErrorCode ConvertMeshElementToCompressedFormatTDyVectorValues(TDy tdy, PetscInt num_element, PetscInt default_offset_size, PetscInt update_offset,
  PetscInt **subelement_num, PetscInt **subelement_offset, TDyVector **subelement_value) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  PetscInt count = 0, new_offset = 0;

  count = (*subelement_num)[0];
  for (PetscInt ielem=1; ielem<num_element; ielem++) {

    new_offset += (*subelement_num)[ielem-1];
    PetscInt old_offset = (*subelement_offset)[ielem];

    for (PetscInt isubelem=0; isubelem<(*subelement_num)[ielem]; isubelem++){

      for (PetscInt d=0; d<dim; d++) {
        (*subelement_value)[new_offset + isubelem].V[d] = (*subelement_value)[old_offset + isubelem].V[d];
      }
      count++;

    }
    if (update_offset) {
      (*subelement_offset)[ielem] = new_offset;
    }
  }

  if (update_offset) {
    new_offset += (*subelement_num)[num_element-1];
    (*subelement_offset)[num_element] = new_offset;
  }

  for (PetscInt ii = count; ii < default_offset_size*num_element; ii++) {
    for (PetscInt d=0; d<dim; d++) {
      (*subelement_value)[ii].V[d] = -1;
    }
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/// Converts TDyCoordinate datatype of TDy mesh element in compressed format
///
/// @param [in] tdy                  A TDy struct
/// @param [in] num_elements         Number of elements
/// @param [in] default_offset_size  Default offset size for elements
/// @param [in] update_offset        Determines if subelement_offset should be updated
/// @param [inout] subelement_num    Number of subelements for a given element
/// @param [inout] subelement_offset Offset of subelements for a given element
/// @param [inout] subelement_id     Subelement ids
/// @returns 0                       on success, or a non-zero error code on failure
PetscErrorCode ConvertMeshElementToCompressedFormatTDyCoordinateValues(TDy tdy, PetscInt num_element, PetscInt default_offset_size, PetscInt update_offset,
  PetscInt **subelement_num, PetscInt **subelement_offset, TDyCoordinate **subelement_value) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  PetscInt count = 0, new_offset = 0;

  count = (*subelement_num)[0];
  for (PetscInt ielem=1; ielem<num_element; ielem++) {

    new_offset += (*subelement_num)[ielem-1];
    PetscInt old_offset = (*subelement_offset)[ielem];

    for (PetscInt isubelem=0; isubelem<(*subelement_num)[ielem]; isubelem++){

      for (PetscInt d=0; d<dim; d++) {
        (*subelement_value)[new_offset + isubelem].X[d] = (*subelement_value)[old_offset + isubelem].X[d];
      }
      count++;

    }
    if (update_offset) {
      (*subelement_offset)[ielem] = new_offset;
    }
  }

  if (update_offset) {
    new_offset += (*subelement_num)[num_element-1];
    (*subelement_offset)[num_element] = new_offset;
  }

  for (PetscInt ii = count; ii < default_offset_size*num_element; ii++) {
    for (PetscInt d=0; d<dim; d++) {
      (*subelement_value)[ii].X[d] = -1;
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts a real datatype of TDy mesh element in compressed format
///
/// @param [in] tdy                  A TDy struct
/// @param [in] num_elements         Number of elements
/// @param [in] default_offset_size  Default offset size for elements
/// @param [in] update_offset        Determines if subelement_offset should be updated
/// @param [inout] subelement_num    Number of subelements for a given element
/// @param [inout] subelement_offset Offset of subelements for a given element
/// @param [inout] subelement_id     Subelement ids
/// @returns 0                       on success, or a non-zero error code on failure
PetscErrorCode ConvertMeshElementToCompressedFormatRealValues(TDy tdy, PetscInt num_element, PetscInt default_offset_size, PetscInt update_offset,
  PetscInt **subelement_num, PetscInt **subelement_offset, PetscReal **subelement_value) {

  PetscFunctionBegin;

  PetscInt count = 0, new_offset = 0;

  count = (*subelement_num)[0];
  for (PetscInt ielem=1; ielem<num_element; ielem++) {

    new_offset += (*subelement_num)[ielem-1];
    PetscInt old_offset = (*subelement_offset)[ielem];

    for (PetscInt isubelem=0; isubelem<(*subelement_num)[ielem]; isubelem++){

      (*subelement_value)[new_offset + isubelem] = (*subelement_value)[old_offset + isubelem];
      count++;

    }
    if (update_offset) {
      (*subelement_offset)[ielem] = new_offset;
    }
  }

  if (update_offset) {
    new_offset += (*subelement_num)[num_element-1];
    (*subelement_offset)[num_element] = new_offset;
  }

  for (PetscInt ii = count; ii < default_offset_size*num_element; ii++) {
    (*subelement_value)[ii] = -1;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts all member variables of a TDyCell struct in compressed format
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ConvertCellsToCompressedFormat(TDy tdy) {

  PetscFunctionBegin;

  DM dm = tdy->dm;
  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  PetscErrorCode ierr;

  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum( dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  PetscInt num_cells = c_end - c_start;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  // compute number of vertices per grid cell
  PetscInt nverts_per_cell = TDyGetNumberOfCellVerticesWithClosures(dm, tdy->closureSize, tdy->closure);
  TDyCellType cell_type = GetCellType(dim, nverts_per_cell);

  PetscInt num_vertices  = GetNumVerticesForCellType(cell_type);
  PetscInt num_edges     = GetNumEdgesForCellType(cell_type);
  PetscInt num_neighbors = GetNumNeighborsForCellType(cell_type);
  PetscInt num_faces     = GetNumFacesForCellType(cell_type);

  PetscInt update_offset = 1;

  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_cells, num_vertices, update_offset,
    &cells->num_vertices, &cells->vertex_offset, &cells->vertex_ids); CHKERRQ(ierr);

  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_cells, num_edges, update_offset,
    &cells->num_edges, &cells->edge_offset, &cells->edge_ids); CHKERRQ(ierr);

  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_cells, num_neighbors, update_offset,
    &cells->num_neighbors, &cells->neighbor_offset, &cells->neighbor_ids); CHKERRQ(ierr);

  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_cells, num_faces, update_offset,
    &cells->num_faces, &cells->face_offset, &cells->face_ids); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts all member variables of a TDySubcell struct in compressed format
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ConvertSubcellsToCompressedFormat(TDy tdy) {

  PetscFunctionBegin;

  DM dm = tdy->dm;
  TDyMesh *mesh = tdy->mesh;
  TDySubcell *subcells = &mesh->subcells;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum( dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  PetscInt num_cells = c_end - c_start;

  PetscInt nverts_per_cell = TDyGetNumberOfCellVerticesWithClosures(dm, tdy->closureSize, tdy->closure);
  TDyCellType cell_type = GetCellType(dim, nverts_per_cell);
  TDySubcellType subcell_type = GetSubcellTypeForCellType(cell_type);

  PetscInt num_subcells   = GetNumSubcellsForSubcellType(subcell_type);
  PetscInt num_nu_vectors = GetNumOfNuVectorsForSubcellType(subcell_type);
  PetscInt num_vertices   = GetNumVerticesForSubcellType(subcell_type);
  PetscInt num_faces      = GetNumFacesForSubcellType(subcell_type);

  PetscInt num_subcells_per_cell = num_cells * num_subcells;

  PetscInt update_offset;

  /* Change variables that have subelement size of 'num_faces'*/
  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_subcells_per_cell, num_faces, update_offset,
    &subcells->num_faces, &subcells->face_offset, &subcells->face_ids); CHKERRQ(ierr);

  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_subcells_per_cell, num_faces, update_offset,
    &subcells->num_faces, &subcells->face_offset, &subcells->is_face_up); CHKERRQ(ierr);

  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_subcells_per_cell, num_faces, update_offset,
    &subcells->num_faces, &subcells->face_offset, &subcells->face_unknown_idx); CHKERRQ(ierr);

  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_subcells_per_cell, num_faces, update_offset,
    &subcells->num_faces, &subcells->face_offset, &subcells->face_flux_idx); CHKERRQ(ierr);

  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatRealValues(tdy, num_subcells_per_cell, num_faces, update_offset,
    &subcells->num_faces, &subcells->face_offset, &subcells->face_area); CHKERRQ(ierr);

  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_subcells_per_cell, num_faces, update_offset,
    &subcells->num_faces, &subcells->face_offset, &subcells->vertex_ids); CHKERRQ(ierr);

  /* Change variables that have subelement size of 'num_nu_vectors'*/
  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatTDyVectorValues(tdy, num_subcells_per_cell, num_nu_vectors, update_offset,
    &subcells->num_nu_vectors, &subcells->nu_vector_offset, &subcells->nu_vector); CHKERRQ(ierr);

  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatTDyVectorValues(tdy, num_subcells_per_cell, num_nu_vectors, update_offset,
    &subcells->num_nu_vectors, &subcells->nu_vector_offset, &subcells->nu_star_vector); CHKERRQ(ierr);

  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatTDyCoordinateValues(tdy, num_subcells_per_cell, num_nu_vectors, update_offset,
    &subcells->num_nu_vectors, &subcells->nu_vector_offset, &subcells->variable_continuity_coordinates); CHKERRQ(ierr);

  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatTDyCoordinateValues(tdy, num_subcells_per_cell, num_nu_vectors, update_offset,
    &subcells->num_nu_vectors, &subcells->nu_vector_offset, &subcells->face_centroid); CHKERRQ(ierr);

  /* Change variables that have subelement size of 'num_vertices'*/
  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatTDyCoordinateValues(tdy, num_subcells_per_cell, num_vertices, update_offset,
    &subcells->num_vertices, &subcells->vertex_offset, &subcells->vertices_coordinates); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts all member variables of a TDyVertex struct in compressed format
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ConvertVerticesToCompressedFormat(TDy tdy) {

  PetscFunctionBegin;

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyVertex     *vertices = &mesh->vertices;
  PetscErrorCode ierr;

  PetscInt v_start, v_end, num_vertices;
  ierr = DMPlexGetDepthStratum( dm, 0, &v_start, &v_end); CHKERRQ(ierr);
  num_vertices = v_end-v_start;

  PetscInt ncells_per_vertex = TDyMaxNumberOfCellsSharingAVertex(dm, tdy->closureSize, tdy->closure);
  PetscInt nfaces_per_vertex = TDyMaxNumberOfFacesSharingAVertex(dm, tdy->closureSize, tdy->closure);
  PetscInt nedges_per_vertex = TDyMaxNumberOfEdgesSharingAVertex(dm, tdy->closureSize, tdy->closure);

  PetscInt update_offset;

  /* Convert edge_ids */
  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_vertices, nedges_per_vertex, update_offset,
    &vertices->num_edges, &vertices->edge_offset, &vertices->edge_ids); CHKERRQ(ierr);

  /* Convert face_ids */

  // Note: face_id and subface_id use the same offset (vertices->face_offset)
  //       So, the offsets are not updated when updating face_ids.
  //       The vertices->face_offset are updated in the second round 
  //       i.e. when the subface_ids are updated.
  update_offset = 0;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_vertices, nfaces_per_vertex, update_offset,
    &vertices->num_faces, &vertices->face_offset, &vertices->face_ids); CHKERRQ(ierr);

  /* Convert subface_ids */
  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_vertices, nfaces_per_vertex, update_offset,
    &vertices->num_faces, &vertices->face_offset, &vertices->subface_ids); CHKERRQ(ierr);


  /* Convert internal_cell_ids */
  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_vertices, ncells_per_vertex, update_offset,
    &vertices->num_internal_cells, &vertices->internal_cell_offset, &vertices->internal_cell_ids); CHKERRQ(ierr);

  /* Convert subcell_ids */
  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_vertices, nfaces_per_vertex, update_offset,
    &vertices->num_internal_cells, &vertices->subcell_offset, &vertices->subcell_ids); CHKERRQ(ierr);

  /* Convert boundary_face_ids */
  update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_vertices, nfaces_per_vertex, update_offset,
    &vertices->num_boundary_faces, &vertices->boundary_face_offset, &vertices->boundary_face_ids); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Converts all member variables of a TDyFace struct in compressed format
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ConvertFacesToCompressedFormat(TDy tdy) {

  PetscFunctionBegin;

  DM dm = tdy->dm;
  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  PetscInt nverts_per_cell = TDyGetNumberOfCellVerticesWithClosures(dm, tdy->closureSize, tdy->closure);
  TDyCellType cell_type = GetCellType(dim, nverts_per_cell);
  PetscInt num_vertices_per_face = GetNumOfVerticesFormingAFaceForCellType(cell_type);

  /* Convert vertex_ids */
  PetscInt num_faces = mesh->num_faces;
  PetscInt update_offset = 1;
  ierr = ConvertMeshElementToCompressedFormatIntegerValues(tdy, num_faces, num_vertices_per_face, update_offset,
    &faces->num_vertices, &faces->vertex_offset, &faces->vertex_ids); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateCellOrientationAroundAVertex(TDy tdy, PetscInt ivertex) {

  PetscFunctionBegin;

  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyEdge       *edges = &mesh->edges;
  PetscErrorCode ierr;

  PetscInt ncells = vertices->num_internal_cells[ivertex];
  PetscInt nedges = vertices->num_edges[ivertex];

  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt vOffsetEdge    = vertices->edge_offset[ivertex];

  // compute angle to all cell centroids w.r.t. the shared vertix
  PetscReal theta[8];
  PetscInt idx[8], ids[8], subcell_ids[8];
  PetscBool is_cell[8];
  PetscBool is_internal_edge[8];
  PetscInt count = 0;
  for (PetscInt i=0; i<ncells; i++) {
    PetscInt icell = vertices->internal_cell_ids[vOffsetCell + i];

    PetscReal x = cells->centroid[icell].X[0] - vertices->coordinate[ivertex].X[0];
    PetscReal y = cells->centroid[icell].X[1] - vertices->coordinate[ivertex].X[1];

    ids[count]              = icell;
    idx[count]              = count;
    subcell_ids[count]      = vertices->subcell_ids[vOffsetSubcell + i];
    is_cell[count]          = PETSC_TRUE;
    is_internal_edge[count] = PETSC_FALSE;

    ierr = ComputeTheta(x, y, &theta[count]); CHKERRQ(ierr);
    count++;
  }

  // compute angle to face centroids w.r.t. the shared vertix
  PetscBool boundary_edge_present = PETSC_FALSE;

  for (PetscInt i=0; i<nedges; i++) {
    PetscInt iedge = vertices->edge_ids[vOffsetEdge + i];
    PetscReal x = edges->centroid[iedge].X[0] - vertices->coordinate[ivertex].X[0];
    PetscReal y = edges->centroid[iedge].X[1] - vertices->coordinate[ivertex].X[1];

    ids[count]              = iedge;
    idx[count]              = count;
    subcell_ids[count]      = -1;
    is_cell[count]          = PETSC_FALSE;
    is_internal_edge[count] = edges->is_internal[iedge];

    if (!edges->is_internal[iedge]) {
      boundary_edge_present = PETSC_TRUE;
    }

    ierr = ComputeTheta(x, y, &theta[count]); CHKERRQ(ierr);
    count++;
  }

  // sort the thetas in anti-clockwise direction
  PetscSortRealWithPermutation(count, theta, idx);

  // determine the starting sorted index
  PetscInt start_idx = -1;
  if (boundary_edge_present) {
    // for a boundary vertex, find the last boundary edge in the
    // anitclockwise direction around the vertex
    for (PetscInt i=0; i<count; i++) {
      if (!is_cell[idx[i]]) { // is this an edge?
        if (!is_internal_edge[idx[i]]) { // is this a boundary edge?
          start_idx = i;
        }
      }
    }
    // if the starting index is the last index AND the first index
    // is an edge index, reset the starting index to be the first index
    if (start_idx == count-1 && !is_cell[idx[0]]) {
      start_idx = 0;
    }

  } else {
    // For an internal vertex, the starting index should be a
    // face centroid
    if ( is_cell[idx[0]] ) {
      start_idx = count-1;
    }
    else {
      start_idx = 0; // assuming that is_cell[idx[1]] = TRUE
    }
  }

  PetscInt tmp_subcell_ids[4];
  PetscInt tmp_cell_ids[4], tmp_edge_ids[4], tmp_ncells, tmp_nedges;
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
    vertices->internal_cell_ids[vOffsetCell + i] = tmp_cell_ids[i];
    vertices->subcell_ids[vOffsetSubcell + i]    = tmp_subcell_ids[i];
  }

  // save information about sorted edge ids
  for (PetscInt i=0; i<tmp_nedges; i++) {
    vertices->edge_ids[vOffsetEdge + i] = tmp_edge_ids[i];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAVertex2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyVertex     *vertices = &mesh->vertices;
  TDyEdge       *edges = &mesh->edges;
  PetscErrorCode ierr;

  PetscInt v_start, v_end;
  ierr = DMPlexGetDepthStratum( dm, 0, &v_start, &v_end); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<v_end-v_start; ivertex++) {

    PetscInt vOffsetEdge = vertices->edge_offset[ivertex];

    if (vertices->num_internal_cells[ivertex] > 1) {
      ierr = UpdateCellOrientationAroundAVertex(tdy, ivertex); CHKERRQ(ierr);
    } else {

      PetscInt edge_id_1 = vertices->edge_ids[vOffsetEdge + 0];
      PetscInt edge_id_2 = vertices->edge_ids[vOffsetEdge + 1];

      PetscReal x = edges->centroid[edge_id_1].X[0] - vertices->coordinate[ivertex].X[0];
      PetscReal y = edges->centroid[edge_id_1].X[1] - vertices->coordinate[ivertex].X[1];
      PetscReal theta_1;
      ierr = ComputeTheta(x, y, &theta_1);

      x = edges->centroid[edge_id_2].X[0] - vertices->coordinate[ivertex].X[0];
      y = edges->centroid[edge_id_2].X[1] - vertices->coordinate[ivertex].X[1];
      PetscReal theta_2;
      ierr = ComputeTheta(x, y, &theta_2);

      if (theta_1 < theta_2) {
        if (theta_2 - theta_1 <= PETSC_PI) {
          vertices->edge_ids[vOffsetEdge + 0] = edge_id_2;
          vertices->edge_ids[vOffsetEdge + 1] = edge_id_1;
        } else {
          vertices->edge_ids[vOffsetEdge + 0] = edge_id_1;
          vertices->edge_ids[vOffsetEdge + 1] = edge_id_2;
        }
      } else {
        if (theta_1 - theta_2 <= PETSC_PI) {
          vertices->edge_ids[vOffsetEdge + 0] = edge_id_1;
          vertices->edge_ids[vOffsetEdge + 1] = edge_id_2;
        } else {
          vertices->edge_ids[vOffsetEdge + 0] = edge_id_2;
          vertices->edge_ids[vOffsetEdge + 1] = edge_id_1;
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// For each vertex, reorder faces and subfaces such that all internal faces/subfaces
/// are listed first followed by boundary faces
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode UpdateFaceOrderAroundAVertex3DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyFace       *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscInt v_start, v_end;
  ierr = DMPlexGetDepthStratum( dm, 0, &v_start, &v_end); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<v_end-v_start; ivertex++) {

    PetscInt *face_ids, num_faces;
    PetscInt *subface_ids, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);
    //num_faces = vertices->num_faces[ivertex];

    PetscInt face_ids_sorted[num_faces];
    PetscInt subface_ids_sorted[num_faces];
    PetscInt count=0;

    // First find all internal faces (i.e. face shared by two cells)
    // and the corresponding subface
    for (PetscInt iface=0;iface<num_faces;iface++) {
      PetscInt face_id = face_ids[iface];
      PetscInt subface_id = subface_ids[iface];
      if (faces->num_cells[face_id]==2) {
        face_ids_sorted[count] = face_id;
        subface_ids_sorted[count] = subface_id;
        count++;
      }
    }

    // Now find all boundary faces (i.e. face shared by a single cell)
    // and the corresponding subfaces
    for (PetscInt iface=0;iface<num_faces;iface++) {
      PetscInt face_id = face_ids[iface];
      PetscInt subface_id = subface_ids[iface];
      if (faces->num_cells[face_id]==1) {
        face_ids_sorted[count] = face_id;
        subface_ids_sorted[count] = subface_id;
        count++;
      }
    }

    // Save the sorted faces/subfaces
    for (PetscInt iface=0;iface<num_faces;iface++) {
      face_ids[iface] = face_ids_sorted[iface];
      subface_ids[iface] = subface_ids_sorted[iface];
    }

  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

PetscErrorCode UpdateCellOrientationAroundAEdge2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  PetscReal      dot_product;
  PetscInt       e_start, e_end;
  TDyCell       *cells = &mesh->cells;
  TDyEdge       *edges = &mesh->edges;
  PetscErrorCode ierr;

  ierr = DMPlexGetHeightStratum(dm, 1, &e_start, &e_end); CHKERRQ(ierr);

  for (PetscInt iedge=0; iedge<e_end-e_start; iedge++) {
    PetscInt eOffsetCell = edges->cell_offset[iedge];

    if (edges->is_internal[iedge]) {
      TDyCoordinate *cell_from_centroid = &cells->centroid[edges->cell_ids[eOffsetCell + 0]];
      TDyCoordinate *cell_to_centroid   = &cells->centroid[edges->cell_ids[eOffsetCell + 1]];

      dot_product = (cell_to_centroid->X[0] - cell_from_centroid->X[0]) * edges->normal[iedge].V[0] +
                    (cell_to_centroid->X[1] - cell_from_centroid->X[1]) * edges->normal[iedge].V[1];
      if (dot_product < 0.0) {
        PetscInt tmp = edges->cell_ids[eOffsetCell + 0];
        edges->cell_ids[eOffsetCell + 0] = edges->cell_ids[eOffsetCell + 1];
        edges->cell_ids[eOffsetCell + 1] = tmp;
      }
    } else {
      TDyCoordinate *cell_from_centroid = &cells->centroid[edges->cell_ids[eOffsetCell + 0]];

      dot_product = (edges->centroid[iedge].X[0] - cell_from_centroid->X[0]) *
                    edges->normal[iedge].V[0] +
                    (edges->centroid[iedge].X[1] - cell_from_centroid->X[1]) * edges->normal[iedge].V[1];
      if (dot_product < 0.0) {
        PetscInt tmp = edges->cell_ids[eOffsetCell + 0];
        edges->cell_ids[eOffsetCell + 0] = -1;
        edges->cell_ids[eOffsetCell + 1] = tmp;
      }

    }
  }

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeVariableContinuityPoint(PetscReal vertex[3],
    PetscReal edge[3], PetscReal alpha, PetscInt dim, PetscReal *point) {
  PetscFunctionBegin;

  for (PetscInt d=0; d<dim; d++) {
    point[d] = (1.0 - alpha)*vertex[d] + edge[d]*alpha;
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeRightNormalVector(PetscReal v1[3], PetscReal v2[3],
                                        PetscInt dim, PetscReal *normal) {

  PetscReal vec_from_1_to_2[3];
  PetscReal norm;

  PetscFunctionBegin;

  if (dim != 2) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "ComputeRightNormalVector only support 2D grids");
  }

  norm = 0.0;

  for (PetscInt d=0; d<dim; d++) {
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

PetscErrorCode SetupSubcellsFor2DMesh(DM dm, TDy tdy) {

  PetscFunctionBegin;

  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDySubcell    *subcells = &mesh->subcells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyEdge       *edges = &mesh->edges;
  PetscInt       c_start, c_end, num_subcells;
  PetscInt       dim;
  PetscInt       e_idx_up, e_idx_dn;
  PetscReal      cell_cen[3], e_cen_up[3], e_cen_dn[3], v_c[3];
  PetscReal      cp_up[3], cp_dn[3], nu_vec_up[3], nu_vec_dn[3];
  PetscReal      len_up, len_dn;
  PetscReal      alpha;
  PetscReal      normal[2], centroid;
  PetscErrorCode ierr;

  alpha = 1.0;

  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (PetscInt icell=0; icell<c_end-c_start; icell++) {

    PetscInt *vertex_ids, num_vertices;
    PetscInt *edge_ids, num_edges;
    ierr = TDyMeshGetCellVertices(mesh, icell, &vertex_ids, &num_vertices); CHKERRQ(ierr);
    ierr = TDyMeshGetCellEdges(mesh, icell, &edge_ids, &num_edges); CHKERRQ(ierr);

    // save cell centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cells->num_subcells[icell];

    for (PetscInt isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell

      PetscInt subcell_id = icell*num_subcells+isubcell;
      PetscInt sOffsetNuVectors = subcells->nu_vector_offset[subcell_id];

      // save coorindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertices, vertex_ids[isubcell], dim, &v_c[0]); CHKERRQ(ierr);

      // determine ids of up & down edges
      e_idx_up = edge_ids[isubcell];

      if (isubcell == 0) e_idx_dn = edge_ids[num_subcells-1];
      else               e_idx_dn = edge_ids[isubcell    -1];

      // save centroids of up/down edges
      ierr = TDyEdge_GetCentroid(edges, e_idx_up, dim, &e_cen_up[0]); CHKERRQ(ierr);
      ierr = TDyEdge_GetCentroid(edges, e_idx_dn, dim, &e_cen_dn[0]); CHKERRQ(ierr);

      // compute continuity point
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_up, alpha, dim, cp_up);
      CHKERRQ(ierr);
      ierr = ComputeVariableContinuityPoint(v_c, e_cen_dn, alpha, dim, cp_dn);
      CHKERRQ(ierr);

      // save continuity point
      for (PetscInt d=0; d<dim; d++) {
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
      for (PetscInt d=0; d<dim; d++) {
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

PetscErrorCode UpdateFaceOrientationAroundAVertex(TDyCoordinate *cell_centroid, TDyFace *faces,
                                                  TDyVertex *vertices, PetscInt ivertex, PetscInt dim,
                                                  PetscInt f_idx[3]) {

  PetscFunctionBegin;

  PetscReal a[3],b[3],c[3],axb[3],dot_prod;
  for (PetscInt d=0; d<dim; d++) {
    a[d] = faces->centroid[f_idx[1]].X[d] - faces->centroid[f_idx[0]].X[d];
    b[d] = faces->centroid[f_idx[2]].X[d] - faces->centroid[f_idx[0]].X[d];
    c[d] = cell_centroid->X[d] - vertices->coordinate[ivertex].X[d];
  }

  PetscErrorCode ierr;
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
PetscErrorCode ExtractCentroidForIthJthTraversalOrder(TDyVertex *vertices, PetscInt ivertex, TDyFace *faces, TDyCell *cells, PetscInt cell_traversal_ij, PetscReal cen[3]) {

  PetscFunctionBegin;
  PetscInt dim = 3;
  PetscErrorCode ierr;

  if (cell_traversal_ij>=0) { // Is a cell?
    PetscInt cell_id;
    PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];

    cell_id = vertices->internal_cell_ids[vOffsetIntCell + cell_traversal_ij];
    ierr = TDyCell_GetCentroid2(cells, cell_id, dim, &cen[0]); CHKERRQ(ierr);

  } else { // is a face

    PetscInt face_id, idx;
    PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

    idx = -cell_traversal_ij - vertices->num_internal_cells[ivertex];
    face_id = vertices->boundary_face_ids[vOffsetBoundaryFace + idx];

    ierr = TDyFace_GetCentroid(faces, face_id, dim, &cen[0]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode UpdateIthTraversalOrder(TDyVertex *vertices, PetscInt ivertex, TDyFace *faces, TDyCell *cells, PetscInt i, PetscBool flip_if_dprod_is_negative, PetscInt **cell_traversal) {

  PetscFunctionBegin;

  PetscInt dim = 3;
  PetscReal v1[3],v2[3],v3[3],v4[3],c2v_vec[3],normal[3];
  PetscReal dot_product;

  // Get pointers to the four cells
  PetscErrorCode ierr;
  ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][0], v1);
  ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][1], v2);
  ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][2], v3);
  ierr = ExtractCentroidForIthJthTraversalOrder(vertices, ivertex, faces, cells, cell_traversal[i][3], v4);

  // Save (x,y,z) of the four cells, and
  // a vector joining centroid of four cells and the vertex (c2v_vec)
  for (PetscInt d=0; d<dim; d++) {
    c2v_vec[d] = vertices->coordinate[ivertex].X[d] - (v1[d] + v2[d] + v3[d] + v4[d])/4.0;
  }

  // Compute the normal to the plane formed by four cells
  ierr = TDyNormalToQuadrilateral(v1, v2, v3, v4, normal); CHKERRQ(ierr);

  // Determine the dot product between normal and c2v_vec
  dot_product = 0.0;
  for (PetscInt d=0; d<dim; d++) {
    dot_product += normal[d]*c2v_vec[d];
  }

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
  for (PetscInt j=0;j<4;j++) if (cell_traversal[i][j]<0) num_faces++;
  if (num_faces>0) {
    if (num_faces != 2) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"UpdateIthTraversalOrder: Unsupported num_faces");
    }

    if (cell_traversal[i][0]>=0 && cell_traversal[i][1]>=0) {
      // do nothing
    } else if (cell_traversal[i][0]>=0 && cell_traversal[i][3]>=0) {
      PetscInt tmp[4];
      tmp[0] = cell_traversal[i][3];
      tmp[1] = cell_traversal[i][0];
      tmp[2] = cell_traversal[i][1];
      tmp[3] = cell_traversal[i][2];
      for (PetscInt j=0;j<4;j++) {
        cell_traversal[i][j] = tmp[j];
      }
    } else if (cell_traversal[i][1]>=0 && cell_traversal[i][2]>=0) {
      PetscInt tmp[4];
      tmp[0] = cell_traversal[i][1];
      tmp[1] = cell_traversal[i][2];
      tmp[2] = cell_traversal[i][3];
      tmp[3] = cell_traversal[i][0];
      for (PetscInt j=0;j<4;j++) {
        cell_traversal[i][j] = tmp[j];
      }
    } else {
      PetscInt tmp[4];
      tmp[0] = cell_traversal[i][2];
      tmp[1] = cell_traversal[i][3];
      tmp[2] = cell_traversal[i][0];
      tmp[3] = cell_traversal[i][1];
      for (PetscInt j=0;j<4;j++) {
        cell_traversal[i][j] = tmp[j];
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetupUpwindFacesForSubcell(TDyMesh *mesh, TDyVertex *vertices, PetscInt ivertex, TDyCell *cells, TDyFace *faces, TDySubcell *subcells, PetscInt **cell_up2dw, PetscInt nUp2Dw) {

  PetscFunctionBegin;

  PetscInt ncells = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];
  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  PetscErrorCode ierr;
  PetscInt *vertex_face_ids, vertex_num_faces;
  PetscInt *subface_ids, num_subfaces;
  ierr = TDyMeshGetVertexFaces(mesh, ivertex, &vertex_face_ids, &vertex_num_faces); CHKERRQ(ierr);
  ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

  PetscInt boundary_cell_count = 0;

  // Compute number of fluxes through internal faces
  PetscInt nflux_in = 0;
  for (PetscInt iface=0; iface<vertex_num_faces; iface++) {
    PetscInt faceID = vertex_face_ids[iface];
    if (faces->is_internal[faceID]) nflux_in++;
  }

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  for (PetscInt icell=0; icell<ncells; icell++) {

    // Determine the cell and subcell id
    PetscInt cell_id  = vertices->internal_cell_ids[vOffsetIntCell + icell];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + icell];

    // Get access to the cell and subcell
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;

    PetscInt *subcell_face_ids, *subcell_is_face_up, *subcell_face_unknown_idx, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellIsFaceUp(mesh, subcell_id, &subcell_is_face_up, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceUnknownIdxs(mesh, subcell_id, &subcell_face_unknown_idx, &subcell_num_faces); CHKERRQ(ierr);

    // Loop over all faces of the subcell
    for (PetscInt iface=0;iface<subcell_num_faces;iface++) {

      PetscInt face_id = subcell_face_ids[iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Boundary face
      if (faces->cell_ids[fOffsetCell + 0] < 0 || faces->cell_ids[fOffsetCell + 1] < 0) {
        subcell_face_unknown_idx[iface] = boundary_cell_count+nflux_in;

        PetscInt face_id_local;
        for (PetscInt ii=0; ii<nfaces_bnd; ii++) {
          if (vertices->boundary_face_ids[vOffsetBoundaryFace + ii] == face_id) {
            face_id_local = ncells + ii;
            break;
          }
        }

        for (PetscInt ii=0; ii<nUp2Dw; ii++) {
          if (cell_up2dw[ii][0] == face_id_local && cell_up2dw[ii][1] == icell){ subcell_is_face_up[iface] = PETSC_TRUE; break;}
          if (cell_up2dw[ii][1] == face_id_local && cell_up2dw[ii][0] == icell){ subcell_is_face_up[iface] = PETSC_FALSE; break;}
        }
        boundary_cell_count++;

      } else {

        // Find the index of cells given by faces->cell_ids[fOffsetCell + 0:1] within the cell id list given
        // vertex->internal_cell_ids
        PetscInt cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[vOffsetIntCell], ncells, faces->cell_ids[fOffsetCell + 0]);
        PetscInt cell_2 = TDyReturnIndexInList(&vertices->internal_cell_ids[vOffsetIntCell], ncells, faces->cell_ids[fOffsetCell + 1]);

        for (PetscInt ii=0; ii<nUp2Dw; ii++) {
          if (cell_up2dw[ii][0] == cell_1 && cell_up2dw[ii][1] == cell_2) {

            subcell_face_unknown_idx[iface] = ii;
            if (cells->id[cell_id] == faces->cell_ids[fOffsetCell + 0])  subcell_is_face_up[iface] = PETSC_TRUE;
            else                                                         subcell_is_face_up[iface] = PETSC_FALSE;
            break;
          } else if (cell_up2dw[ii][0] == cell_2 && cell_up2dw[ii][1] == cell_1) {

            subcell_face_unknown_idx[iface] = ii;
            if (cells->id[cell_id] == faces->cell_ids[fOffsetCell + 1]) subcell_is_face_up[iface] = PETSC_TRUE;
            else                                                        subcell_is_face_up[iface] = PETSC_FALSE;
            break;
          }
        }
      }
    }
  }

  PetscInt nup_bnd_flux=0, ndn_bnd_flux=0;

  PetscInt nflux_bc_up = 0;
  PetscInt nflux_bc_dn = 0;
  PetscInt vOffsetCell = vertices->internal_cell_offset[ivertex];
  for (PetscInt i=0; i<ncells; i++) {
    PetscInt icell    = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

    PetscInt *subcell_face_ids, *subcell_is_face_up, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellIsFaceUp(mesh, subcell_id, &subcell_is_face_up, &subcell_num_faces); CHKERRQ(ierr);

    for (PetscInt iface=0; iface<subcell_num_faces; iface++) {

      PetscInt faceID = subcell_face_ids[iface];
      if (faces->is_internal[faceID]) continue;

      PetscBool upwind_entries;
      upwind_entries = (subcell_is_face_up[iface]==1);

      if (upwind_entries) nflux_bc_up++;
      else                nflux_bc_dn++;
    }

  }


  // Save the face index that corresponds to the flux in transmissibility matrix
  for (PetscInt icell=0; icell<ncells; icell++) {

    // Determine the cell and subcell id
    PetscInt cell_id  = vertices->internal_cell_ids[vOffsetIntCell+icell];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell+icell];

    // Get access to the cell and subcell
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;

    PetscInt *subcell_face_ids, *subcell_is_face_up, *subcell_face_unknown_idx, *subcell_face_flux_idx, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellIsFaceUp(mesh, subcell_id, &subcell_is_face_up, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceUnknownIdxs(mesh, subcell_id, &subcell_face_unknown_idx, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceFluxIdxs(mesh, subcell_id, &subcell_face_flux_idx, &subcell_num_faces); CHKERRQ(ierr);

    // Loop over all faces of the subcell and
    // - Updates face_ids for a vertex such that first all
    //   internal faces are listed, followed by upwind boundary
    //   faces, and the downward boundary faces are last
    // - Similary, the index of the flux through the faces of a
    //   subcell are identifed. The internal fluxes are first,
    //   followed by upwind boundary and downwind boundary faces.
    for (PetscInt iface=0; iface<subcell_num_faces; iface++) {
      PetscInt face_id = subcell_face_ids[iface];

      PetscInt idx_flux = subcell_face_unknown_idx[iface];
      if (faces->is_internal[face_id]) {
        vertex_face_ids[idx_flux] = face_id;
        subcell_face_flux_idx[iface] = idx_flux;
      } else {
        if (subcell_is_face_up[iface]) {
          vertex_face_ids[nflux_in + nup_bnd_flux] = face_id;
          subcell_face_flux_idx[iface] = nflux_in+nup_bnd_flux;
          nup_bnd_flux++;
        } else {
          vertex_face_ids[nflux_in + nflux_bc_up + ndn_bnd_flux] = face_id;
          subcell_face_flux_idx[iface] = nflux_in+ndn_bnd_flux;
          ndn_bnd_flux++;
        }
      }

    }
  }

  // Since vertices->face_ids[] has been updated, now
  // update vertices->subface_ids[]
  for (PetscInt iface=0; iface<vertices->num_faces[ivertex]; iface++) {
    PetscInt face_id = vertex_face_ids[iface];
    PetscInt fOffsetVertex = faces->vertex_offset[face_id];
    PetscBool found = PETSC_FALSE;
    for (PetscInt ii=0; ii<faces->num_vertices[face_id]; ii++) {
      if (faces->vertex_ids[fOffsetVertex + ii] == ivertex) {
        subface_ids[iface] = ii;
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a vertex within a face");
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscBool VerticesHaveSameXYCoords(TDy tdy, PetscInt ivertex_1, PetscInt ivertex_2) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  PetscBool sameXY = PETSC_FALSE;
  PetscReal dist = 0.0, eps = 1.e-14;

  for (PetscInt d=0;d<2;d++)
    dist += PetscSqr(vertices->coordinate[ivertex_1].X[d] - vertices->coordinate[ivertex_2].X[d]);

  if (dist<eps) sameXY = PETSC_TRUE;

  PetscFunctionReturn(sameXY);
}


/* -------------------------------------------------------------------------- */
PetscInt VertexIdWithSameXYInACell(TDy tdy, PetscInt icell, PetscInt ivertex) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  PetscInt result;
  PetscBool found = PETSC_FALSE;

  PetscErrorCode ierr;

  PetscInt *vertex_ids, num_vertices;
  ierr = TDyMeshGetCellVertices(mesh, icell, &vertex_ids, &num_vertices); CHKERRQ(ierr);

  for (PetscInt iv=0; iv<num_vertices; iv++) {

    PetscInt ivertex_2 = vertex_ids[iv];

    if (ivertex != ivertex_2) {

      if (VerticesHaveSameXYCoords(tdy, ivertex, ivertex_2)) {
        found = PETSC_TRUE;
        result = ivertex_2;
        break;
      }
    }
  }

  if (!found) {
    char error_msg[100];
    sprintf(error_msg,"VertexIdWithSameXYInACell: Could not determine if cell_id %d is above or below vertex_id %d\n",icell,ivertex);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,error_msg);
  }

  PetscFunctionReturn(result);
}


/* -------------------------------------------------------------------------- */
PetscBool IsCellAboveTheVertex(TDy tdy, PetscInt icell, PetscInt ivertex) {

  PetscFunctionBegin;

  PetscBool is_above;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  PetscBool found = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscInt *vertex_ids, num_vertices;
  ierr = TDyMeshGetCellVertices(mesh, icell, &vertex_ids, &num_vertices); CHKERRQ(ierr);

  for (PetscInt iv=0; iv<num_vertices; iv++) {

    PetscInt ivertex_2 = vertex_ids[iv];

    if (ivertex != ivertex_2) {

      if (VerticesHaveSameXYCoords(tdy, ivertex, ivertex_2)) {
        found = PETSC_TRUE;
        is_above = (vertices->coordinate[ivertex_2].X[2]>vertices->coordinate[ivertex].X[2]);
        break;
      }
    }
  }

  if (!found) {
    char error_msg[100];
    sprintf(error_msg,"IsCellAboveTheVertex: Could not determine if cell_id %d is above or below vertex_id %d\n",icell,ivertex);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,error_msg);
  }

  PetscFunctionReturn(is_above);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode DetermineCellsAboveAndBelow(TDy tdy, PetscInt ivertex, PetscInt **cellsAbvBlw,
                PetscInt *ncells_abv, PetscInt *ncells_blw) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  PetscErrorCode ierr;

  PetscInt ncells_int = vertices->num_internal_cells[ivertex];

  *ncells_abv = 0;
  *ncells_blw = 0;

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
  for (PetscInt icell=0; icell<ncells_int; icell++) {
    PetscInt cellID = vertices->internal_cell_ids[vOffsetIntCell + icell];

    if (IsCellAboveTheVertex(tdy,cellID,ivertex)) {
      cellsAbvBlw[0][*ncells_abv] = cellID;
      (*ncells_abv)++;
    } else {
      cellsAbvBlw[1][*ncells_blw] = cellID;
      (*ncells_blw)++;
    }
  }

  // Rearrange the cells above/below the vertex such that the boundary cells are
  // first and last cells at each level
  PetscInt level, ncells_level;

  for (level=0; level<2; level++) {
    if (level==0) ncells_level = *ncells_abv;
    else          ncells_level = *ncells_blw;

    // Skip if there is only one cell
    if (ncells_level<2) continue;

    PetscInt ivertexAbvBlw[ncells_level];
    PetscInt IsBoundaryfaceIDsAbvBlw[ncells_level][2];

    PetscInt maxBndFaces = 0;
    for (PetscInt ii=0; ii<ncells_level; ii++) {
      PetscInt cellID = cellsAbvBlw[level][ii];
      PetscBool found = PETSC_FALSE;

      PetscInt *vertex_ids, num_vertices;
      ierr = TDyMeshGetCellVertices(mesh, cellID, &vertex_ids, &num_vertices); CHKERRQ(ierr);

      // Find the vertex that is above/below ivertex
      for (PetscInt iv=0; iv<num_vertices; iv++) {
        ivertexAbvBlw[level] = vertex_ids[iv];
        if (ivertex != ivertexAbvBlw[level]) {
          if (VerticesHaveSameXYCoords(tdy, ivertex, ivertexAbvBlw[level])) {
            found = PETSC_TRUE;
            break;
          }
        }
      }
      if (!found) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a vertex that has same XY coords");
      }

      // Find faces that are shared by (ivertex, ivertexAbvBlw)
      PetscInt *face_ids, num_faces;
      ierr = TDyMeshGetCellFaces(mesh, cellID, &face_ids, &num_faces); CHKERRQ(ierr);

      PetscInt faceCount = 0;
      for (PetscInt iface=0; iface<num_faces; iface++) {
        PetscInt faceID = face_ids[iface];
        PetscInt fOffsetVert = faces->vertex_offset[faceID];

        // Does the faceID-th face contains vertices corresponding to ivertex and ivertexAbvBlw?
        PetscInt count=0;
        for (PetscInt iv=0; iv<faces->num_vertices[faceID]; iv++) {
          if (faces->vertex_ids[fOffsetVert+iv] == ivertex) count++;
          if (faces->vertex_ids[fOffsetVert+iv] == ivertexAbvBlw[level]) count++;
        }

        if (count == 2) {
          // The faceID-th face contains BOTH vertices corresponding to ivertex and ivertexAbvBlw,
          // so save it
          IsBoundaryfaceIDsAbvBlw[ii][faceCount] = (faces->is_internal[faceID] == 0);
          faceCount++;
        }

      }
      if (IsBoundaryfaceIDsAbvBlw[ii][0]+IsBoundaryfaceIDsAbvBlw[ii][1] > maxBndFaces) {
        maxBndFaces = IsBoundaryfaceIDsAbvBlw[ii][0]+IsBoundaryfaceIDsAbvBlw[ii][1];
      }
    }

    // Rearrange the list only if there is a boundary face
    if (maxBndFaces>0) {
      PetscInt cellsAbvBlw_rearranged[ncells_level];
      PetscInt count_bnd=0,count_int=1;

      for (PetscInt icell=0; icell<ncells_level; icell++) {

        if ( IsBoundaryfaceIDsAbvBlw[icell][0]+IsBoundaryfaceIDsAbvBlw[icell][1] > 0) {
          // This cell has a boundary face

          if (count_bnd==0) {
            // Since this is the first boundary cell, put it at the beginning of the list
            cellsAbvBlw_rearranged[0] = cellsAbvBlw[level][icell];
            count_bnd++;
          } else {
            // Since this is the second boundary cell, put it at the end of the list
            cellsAbvBlw_rearranged[ncells_level-1] = cellsAbvBlw[level][icell];
          }
        } else {
          // This cell does not have a boundary face, so put it in the list excluding
          // the first and the last position
          cellsAbvBlw_rearranged[count_int] = cellsAbvBlw[level][icell];
          count_int++;
        }
      }

      // Now, copy the rearranged listed
      for (PetscInt icell=0; icell<ncells_level; icell++) {
        cellsAbvBlw[level][icell] = cellsAbvBlw_rearranged[icell];
      }
    } // maxBndFaces>0
  } // for (level=0; level<2; level++)

  if (*ncells_abv>0 && *ncells_blw>0) {
    if (*ncells_abv != *ncells_blw) {
      char error_msg[400];
      char fun_name[100] = "DetermineCellsAboveAndBelow";
      sprintf(error_msg,"%s: No. of cells above (=%d) and below (=%d) of the vertex_id %d are not same. Such a mesh is unsupported.\n",
        fun_name, *ncells_abv, *ncells_blw, ivertex);
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,error_msg);
    }
  } else if (*ncells_abv == 0 && *ncells_blw == 0) {
    char error_msg[200];
    char fun_name[100] = "DetermineCellsAboveAndBelow";
    sprintf(error_msg,"%s: Did not find any cells above and below vertex_id %d\n",fun_name,ivertex);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,error_msg);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscBool AreCellsNeighbors(TDy tdy, PetscInt cell_id_1, PetscInt cell_id_2) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  PetscErrorCode ierr;

  PetscInt *face_ids_1, num_faces_1;
  PetscInt *face_ids_2, num_faces_2;
  ierr = TDyMeshGetCellFaces(mesh, cell_id_1, &face_ids_1, &num_faces_1); CHKERRQ(ierr);
  ierr = TDyMeshGetCellFaces(mesh, cell_id_2, &face_ids_2, &num_faces_2); CHKERRQ(ierr);

  PetscBool are_neighbors = PETSC_FALSE;
  for (PetscInt ii=0; ii<num_faces_1; ii++) {
    for (PetscInt jj=0; jj<num_faces_2; jj++) {
      if (face_ids_1[ii] == face_ids_2[jj] ) {
        are_neighbors = PETSC_TRUE;
        break;
      }
    }
  }

  PetscFunctionReturn(are_neighbors);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ArrangeCellsInCircularOrder(TDy tdy, PetscInt *cellsAbvBlw, PetscInt ncells, PetscInt *cell_order) {

  PetscFunctionBegin;

  // 1. If all cells in cellsAbvBlw do not have same number of neighbors, then
  //    - Check to ensure only two cells have miniumum number of neighbors, and
  //    - Arrange cellsAbvBlw such that cells with minimum number of neighbors are
  //      at the start and end of the array.

  PetscInt nNeighbors[ncells];
  for (PetscInt ii = 0; ii < ncells; ii++) {
    nNeighbors[ii] = 0;
  }

  // Determine the number of neigbhors
  for (PetscInt ii = 0; ii < ncells; ii++){
    for (PetscInt jj = ii+1; jj < ncells; jj++){
      PetscInt result = AreCellsNeighbors(tdy, cellsAbvBlw[ii], cellsAbvBlw[jj]);

      nNeighbors[ii] += result;
      nNeighbors[jj] += result;
    }
  }

  // Determine (1) min/max number of neighbors, and (2) index of cell
  // with minimum number of cells
  PetscInt minIdx = -1, minNeighbors = ncells, maxNeighbors = 0;
  for (PetscInt ii = 0; ii < ncells; ii++) {
    if (nNeighbors[ii] < minNeighbors) {
      minNeighbors = nNeighbors[ii];
      minIdx = ii;
    }
    if (nNeighbors[ii] > maxNeighbors) {
      maxNeighbors = nNeighbors[ii];
    }
  }

  if (minNeighbors != maxNeighbors) {

    // Cells have different number of neighbors

    // Check only two cells have min number of neighbors
    PetscInt count;
    count = 0;
    for (PetscInt ii = 0; ii < ncells; ii++) {
      if (nNeighbors[ii] == minNeighbors) {
        count++;
      }
    }
    if (count != 2) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Number of cells with minimum number of neighbors is not TWO");
    }

    // Put the cells with min number of neigbhors at the start and end of the array
    PetscInt tmpOrder[ncells];
    count = 0;
    tmpOrder[0] = cellsAbvBlw[minIdx];
    for (PetscInt ii = 0; ii < ncells; ii++){
      if (ii != minIdx && nNeighbors[ii] > minNeighbors) {
        count++;
        tmpOrder[count] = cellsAbvBlw[ii];
      }
    }

    for (PetscInt ii = 0; ii < ncells; ii++){
      if (ii != minIdx && nNeighbors[ii] == minNeighbors) {
        count++;
        tmpOrder[count] = cellsAbvBlw[ii];
      }
    }

    // Copy the new cells in new order 
    for (PetscInt ii = 0; ii < ncells; ii++) {
      cellsAbvBlw[ii] = tmpOrder[ii];
    }

  }

  //
  // 2. Now rearrange the cells such that cellsAbvBlw[i] is neighbor to
  //    cellsAbvBlw[i+1].
  //

  // First put cells in an order that could be clockwise or anticlockwise
  PetscInt cell_used[ncells];

  for (PetscInt ii=0; ii<ncells; ii++) cell_used[ii] = 0;

  cell_order[0] = cellsAbvBlw[0];
  cell_used[0] = 1;

  for (PetscInt ii=0; ii<ncells-1; ii++) {
    // For ii-th cell find a neighboring cell that hasn't been
    // previously identified it's neigbhor.

    PetscBool found = PETSC_FALSE;

    for (PetscInt jj=0; jj<ncells; jj++) {

      if (cell_used[jj] == 0) {

        if (AreCellsNeighbors(tdy, cell_order[ii], cellsAbvBlw[jj])) {
          cell_order[ii+1] = cellsAbvBlw[jj];
          cell_used[jj] = 1;
          found = PETSC_TRUE;
          break;
        }
      }
    }
    if (!found) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a neighbor");
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscInt IDofFaceSharedByTwoCellsForACommonVertex(TDy tdy, PetscInt ivertex, PetscInt icell_1, PetscInt icell_2) {

  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscInt *face_ids, num_faces;
  ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);

  PetscInt face_id;
  PetscBool found;
  for (PetscInt iface=0; iface<num_faces; iface++) {
    face_id = face_ids[iface];

    found = PETSC_FALSE;
    PetscInt fOffsetCell = faces->cell_offset[face_id];

    if (
        (faces->cell_ids[fOffsetCell  ] == icell_1 || faces->cell_ids[fOffsetCell  ]  == icell_2 ) &&
        (faces->cell_ids[fOffsetCell+1] == icell_1 || faces->cell_ids[fOffsetCell+1]  == icell_2 )
       ) {
      found = PETSC_TRUE;
      break;
    }
  }

  if (!found) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a shared face");
  }

   PetscFunctionReturn(face_id);
 }

/* -------------------------------------------------------------------------- */
PetscBool PointsAreInAntiClockDirInXYPlane(PetscReal a[2], PetscReal b[2], PetscReal c[2]) {
  PetscFunctionBegin;

  PetscReal ba[3],ca[3],baXca[3];
  for (PetscInt d=0; d<2; d++) {
    ba[d] = b[d] - a[d];
    ca[d] = c[d] - a[d];
  }
  ba[2] = 0.0; ca[2] = 0.0;

  PetscErrorCode ierr = TDyCrossProduct(ba,ca,baXca); CHKERRQ(ierr);

  PetscBool result;
  if (baXca[2]>0.0) {
    result = PETSC_TRUE;
  } else {
    result = PETSC_FALSE;
  }

  PetscFunctionReturn(result);
}

/* -------------------------------------------------------------------------- */

  PetscErrorCode ArrangeCellsInAntiClockwiseDirection(TDy tdy, PetscInt ivertex, PetscInt *inp_cell_order, PetscInt ncells, PetscInt *out_cell_order) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyVertex *vertices = &mesh->vertices;
  TDyFace *faces = &mesh->faces;

  // Find the face that is shared by first and second cell
  PetscInt face_id = IDofFaceSharedByTwoCellsForACommonVertex(tdy, ivertex, inp_cell_order[0], inp_cell_order[1]);

  // Now rearrnge the cell order to be anticlockwise direction
  //   vec_a = (centroid_0 - ivertex)
  //   vec_b = (face_01    - ivertex)
  //
  // If dotprod(vec_a,vec_b) > 0, the cell_order is in anticlockwise
  // else flip the cell_order
  //

  PetscReal a2d[2],b2d[2],c2d[2];
  for (PetscInt d=0; d<2; d++) {
    a2d[d] = vertices->coordinate[ivertex     ].X[d];
    b2d[d] = cells->centroid[inp_cell_order[0]].X[d];
    c2d[d] = faces->centroid[face_id          ].X[d];
  }

  if (PointsAreInAntiClockDirInXYPlane(a2d,b2d,c2d)) {
    // Cells are in anticlockwise direction, so copy the ids
    for (PetscInt ii=0; ii<ncells; ii++) {
      out_cell_order[ii] = inp_cell_order[ii];
    }
  } else {
    // Cells are in clockwise direction, so flip the order
    for (PetscInt ii=0; ii<ncells; ii++) {
      out_cell_order[ii] = inp_cell_order[ncells-ii-1];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode RearrangeCellsInListAsNeighbors(TDy tdy, PetscInt ncells, PetscInt *cell_list_1, PetscInt *cell_list_2) {

  PetscFunctionBegin;

  PetscInt tmp_cell_order[ncells];
  for (PetscInt ii=0; ii<ncells; ii++) {
    PetscBool found = PETSC_FALSE;
    for (PetscInt jj=0; jj<ncells; jj++) {
      if (AreCellsNeighbors(tdy, cell_list_1[ii], cell_list_2[jj])) {
        tmp_cell_order[ii] = cell_list_2[jj];
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a corresponding cell below the given cell");
  }

  // Update the order of cells below the vertex
  for (PetscInt ii=0; ii<ncells; ii++) {
    cell_list_2[ii] = tmp_cell_order[ii];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode RearrangeCellsInAntiClockwiseDir(TDy tdy, PetscInt ivertex, PetscInt **cellsAbvBlw,
                PetscInt ncells_abv, PetscInt ncells_blw) {

  PetscFunctionBegin;

  PetscInt level, ncells;
  if (ncells_abv>0) {
    ncells = ncells_abv;
    level = 0;
  } else {
    ncells = ncells_blw;
    level = 1;
  }

  PetscInt cell_order[ncells];
  PetscErrorCode ierr;

  ierr = ArrangeCellsInCircularOrder(tdy, cellsAbvBlw[level], ncells, &cell_order[0]); CHKERRQ(ierr);

  ierr = ArrangeCellsInAntiClockwiseDirection(tdy, ivertex, cell_order, ncells, cellsAbvBlw[level]); CHKERRQ(ierr);

  if (ncells_abv>0 && ncells_blw>0) {
    // Cells are present above and below the ivertex and
    // only the cells above the vertex were sorted.
    // So, for each cell above the vertex, find the corresponding
    // cell below the vertex
    ierr = RearrangeCellsInListAsNeighbors(tdy, ncells, cellsAbvBlw[0], cellsAbvBlw[1]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AddTwoBndFacesOfTwoCellsInTraversalDirection(TDy tdy, PetscInt ivertex, PetscInt *cellsAbvBlw,
  PetscInt ncells_level, PetscInt *cell_traversal) {

  // For the last and first cell in cellsAbvBlw, add the face that
  // that includes a vertex directly above or below ivertex

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  TDyFace *faces = &mesh->faces;

  PetscInt ncells    = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];
  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  PetscInt bnd_faces_found[nfaces_bnd];
  for (PetscInt ii=0; ii<nfaces_bnd; ii++) bnd_faces_found[ii] = 0;

  // For the first and the last cell, check if there is a boundary face
  // that corresponds to the anticlockwise traversal direction.

  for (PetscInt kk=0; kk<2; kk++) { // For the first and last cell

    // Select the mm-th (first or last) cell
    PetscInt mm;
    if (kk == 0) {
      mm = ncells_level-1;
    } else {
      mm = 0;
    }

    PetscBool found = PETSC_FALSE;

    // Loop through all boundary faces of ivertex
    for (PetscInt ii=0; ii<nfaces_bnd; ii++) {

      // Skip the boundary face that has been previously identified
      if (bnd_faces_found[ii]) continue;

      PetscInt face_id = vertices->boundary_face_ids[vOffsetBoundaryFace + ii];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Check if the face belongs to the mm-th cell
      if ((cellsAbvBlw[mm] == faces->cell_ids[fOffsetCell]  )||
          (cellsAbvBlw[mm] == faces->cell_ids[fOffsetCell+1])) {

        PetscInt fOffsetVertex = faces->vertex_offset[face_id];

        // If the face has a vertex that is exactly above or below the
        // ivertex, add it to the traversal direction
        for (PetscInt jj=0; jj<faces->num_vertices[face_id]; jj++) {
          if ( (ivertex != faces->vertex_ids[fOffsetVertex + jj])
              && VerticesHaveSameXYCoords(tdy, ivertex, faces->vertex_ids[fOffsetVertex + jj])) {
            found = PETSC_TRUE;
            bnd_faces_found[ii] = 1;
            cell_traversal[ncells_level+kk] = ncells + ii;
            break;
          }
        }
        if (found) break;
      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode AddTwoBndFacesOfACellInTraversalDirection(TDy tdy, PetscInt ivertex, PetscInt *cellsAbvBlw,
  PetscInt ncells_level, PetscInt *cell_traversal) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;

  PetscInt ncells    = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];
  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  PetscInt mm = 0, count=0, face_id_1, ii_1, ii_2;
  for (PetscInt ii=0; ii<nfaces_bnd; ii++) {
    PetscInt face_id = vertices->boundary_face_ids[vOffsetBoundaryFace + ii];
    PetscInt fOffsetCell = faces->cell_offset[face_id];
    if ((cellsAbvBlw[mm] == faces->cell_ids[fOffsetCell]  )||
        (cellsAbvBlw[mm] == faces->cell_ids[fOffsetCell+1])) {
      if (count==0) {
        face_id_1 = face_id;
        ii_1 = ii + ncells;
        count++;
      } else {
        ii_2 = ii + ncells;
        count++;
      }
    }
  }
  if (count!=2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not two faces for the cell in the list");

  // Determine if cellID --> face_id_1 is in anticlockwise direction

  PetscReal a[3], b[3], axb[3];
  PetscInt dim=2;
  PetscErrorCode ierr;
  PetscInt cellID = cellsAbvBlw[mm];

  for (PetscInt d=0; d<dim; d++) {
    a[d] = cells->centroid[cellID   ].X[d] - vertices->coordinate[ivertex].X[d];
    b[d] = faces->centroid[face_id_1].X[d] - vertices->coordinate[ivertex].X[d];
  }
  a[2] = 0.0;
  b[2] = 0.0;

  ierr = TDyCrossProduct(a,b,axb); CHKERRQ(ierr);

  if (axb[2]>0) { // cellID --> face_id_1 is in anticlockwise direction
    cell_traversal[1] = ii_1;
    cell_traversal[2] = ii_2;
  } else {
    cell_traversal[1] = ii_2;
    cell_traversal[2] = ii_1;
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ConvertCellsAntiClockwiseDirInTraversalDir(TDy tdy, PetscInt ivertex, PetscInt *cell_ids_abv_blw,
  PetscInt ncells_level, PetscInt *cell_traversal){

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;

  PetscInt ncells,nfaces_bnd;

  if (ncells_level==0) PetscFunctionReturn(0);

  ncells    = vertices->num_internal_cells[ivertex];
  nfaces_bnd= vertices->num_boundary_faces[ivertex];

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];

  PetscBool found;

  // Values of cell_traversal[0:1][:] should corresponds to cell/face IDs in
  // local numbering

  // Add internal cells in the traversal direction.
  // Note: The travel direction of cells in cell_ids_abv_blw is in
  //       the local numbering corresponding to vertices->internal_cell_ids
  for (PetscInt ii=0; ii<ncells_level; ii++) {
    found = PETSC_FALSE;
    for (PetscInt jj=0; jj<ncells; jj++) {
      PetscInt cellID = vertices->internal_cell_ids[vOffsetIntCell + jj];
      if (cellID == cell_ids_abv_blw[ii]) {
        cell_traversal[ii] = jj;
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a the cell in the list");
  }

  if (nfaces_bnd == 0) PetscFunctionReturn(0);

  // Now add boundary faces in the traversal direction
  if (ncells_level>1) {
    AddTwoBndFacesOfTwoCellsInTraversalDirection(tdy, ivertex, cell_ids_abv_blw, ncells_level, cell_traversal);
  } else if (ncells_level == 1) {
    AddTwoBndFacesOfACellInTraversalDirection(tdy, ivertex, cell_ids_abv_blw, ncells_level, cell_traversal);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AddUpDownBndFacesOfCellsInTraversalDirection(TDy tdy, PetscInt ivertex, PetscInt **cell_ids_abv_blw,
  PetscInt ncells_level, PetscInt level, PetscInt **cell_traversal){

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  TDyFace *faces = &mesh->faces;

  PetscInt ncells    = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];

  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  PetscInt bnd_faces_found[nfaces_bnd];

  PetscInt level_faces;
  if (level == 0) level_faces = 1;
  else            level_faces = 0;

  // Identify the boundary faces that have already been used
  for (PetscInt ii=0; ii<nfaces_bnd; ii++) {
    bnd_faces_found[ii] = 0;
  }
  for (PetscInt ii=0; ii<ncells+nfaces_bnd; ii++) {
    PetscInt jj = cell_traversal[level][ii];
    if (jj >= ncells) bnd_faces_found[jj-ncells] = 1;
  }

  // For each cell at the given 'level', find a corresponding top/bottom face
  for (PetscInt ii=0; ii<ncells_level; ii++) {
    for (PetscInt jj=0; jj<nfaces_bnd; jj++) {
      if (bnd_faces_found[jj]) continue;
      PetscInt face_id = vertices->boundary_face_ids[vOffsetBoundaryFace + jj];
      PetscInt fOffsetCell = faces->cell_offset[face_id];
      if ((cell_ids_abv_blw[level][ii] == faces->cell_ids[fOffsetCell]  )||
          (cell_ids_abv_blw[level][ii] == faces->cell_ids[fOffsetCell+1])) {
          cell_traversal[level_faces][ii] = ncells+jj;
        bnd_faces_found[jj] = 1;
        break;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetUpToDownConnections(TDy tdy, PetscInt ivertex, PetscInt **cell_traversal, PetscInt **cell_up2dw, PetscInt *nUp2Dw) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;

  PetscInt ncells    = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];

  PetscInt aa,bb,l2r_dir;
  PetscInt count=0;
  // Set connection between cells that are at the same
  // level (=above/below) w.r.t. the vertex
  for (PetscInt level=0; level<2; level++) {
    if (level==0) l2r_dir = 1;  // x ---> y
    else          l2r_dir = 0;  // x <--- y
    for (PetscInt ii = 0; ii<ncells+nfaces_bnd; ii++) {
      aa = cell_traversal[level][ii];
      if (aa == -1) break;

      if (ii == ncells+nfaces_bnd-1) bb = cell_traversal[level][0];
      else                           bb = cell_traversal[level][ii+1];

      if (bb == -1) bb = cell_traversal[level][0];

      if (aa>=ncells && bb>=ncells) continue;

      if (l2r_dir) {
        cell_up2dw[count][0] = aa;
        cell_up2dw[count][1] = bb;
      } else {
        cell_up2dw[count][0] = bb;
        cell_up2dw[count][1] = aa;
      }
      count++;
    }
  }

  // Set up-to-dw connections between cells above and
  // below the vertex
  for (PetscInt ii=0; ii<ncells+nfaces_bnd; ii++) {
    aa = cell_traversal[0][ii];
    bb = cell_traversal[1][ii];
    if (aa == -1 || bb == -1) break;
    if (aa>=ncells && bb>=ncells) continue;

    if (ii%2 == 0) {
      cell_up2dw[count][0] = aa;
      cell_up2dw[count][1] = bb;
    } else {
      cell_up2dw[count][0] = bb;
      cell_up2dw[count][1] = aa;
    }
    count++;
  }

  // Rearrange cell_up2dw such that connections between cells are first
  // in the list followed by connections between cells and faces
  PetscInt tmp[count][2],found[count];
  for (PetscInt ii=0;ii<count;ii++) {
    found[ii] = 0;
  }
  PetscInt jj = 0;
  for (PetscInt ii=0;ii<count;ii++) {
    if (cell_up2dw[ii][0]<ncells && cell_up2dw[ii][1]<ncells) {
      tmp[jj][0] = cell_up2dw[ii][0];
      tmp[jj][1] = cell_up2dw[ii][1];
      found[ii] = 1;
      jj++;
    }
  }
  for (PetscInt ii=0;ii<count;ii++) {
    if (found[ii]==0) {
      tmp[jj][0] = cell_up2dw[ii][0];
      tmp[jj][1] = cell_up2dw[ii][1];
      jj++;
    }
  }
  for (PetscInt ii=0;ii<count;ii++) {
    cell_up2dw[ii][0] = tmp[ii][0];
    cell_up2dw[ii][1] = tmp[ii][1];
  }
  *nUp2Dw = count;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode DetermineUpwindFacesForSubcell_PlanarVerticalFaces(TDy tdy, PetscInt ivertex, PetscInt **cellsAbove, PetscInt **cellsBelow) {

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

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDySubcell *subcells = &mesh->subcells;
  TDyVertex *vertices = &mesh->vertices;

  PetscInt ncells    = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];

  // 1. Determine cells above and below the given vertex.
  //    - NOTE:
  //           (a) Cells may be present only at one level(=above/below)
  //           (b) If there are cells present at both levels, then
  //               - The number of cells above and below should be the same, AND
  //               - For each cell above the vertex, there should be a corresponding
  //                 cell below it.
  //           (c) A cell in the given level should share at least one
  //               face with another cell within the same level
  // 2. Rearrange above/below cells so they in anti-clockwise direction
  // 3. Compute traversal of cells and boundary faces
  // 4. Determine upwind/downind cells and boundary faces


  PetscInt **cell_ids_abv_blw;

  PetscErrorCode ierr;
  ierr = TDyAllocate_IntegerArray_2D(&cell_ids_abv_blw, 2, ncells+nfaces_bnd); CHKERRQ(ierr);

  PetscInt ncells_abv = 0;
  PetscInt ncells_blw = 0;

  // Copy precomputed cell IDs above the ivertex
  if (cellsAbove[ivertex][0] > 0) {
    ncells_abv = cellsAbove[ivertex][0];
    for (PetscInt ii=0; ii<cellsAbove[ivertex][0]; ii++) {
      cell_ids_abv_blw[0][ii] = cellsAbove[ivertex][ii+1];
    }
  }

  // Copy precomputed cell IDs below the ivertex
  if (cellsBelow[ivertex][0] > 0) {
    ncells_blw = cellsBelow[ivertex][0];
    for (PetscInt ii=0; ii<cellsBelow[ivertex][0]; ii++) {
      cell_ids_abv_blw[1][ii] = cellsBelow[ivertex][ii+1];
    }
  }

  if (ncells_abv > 1) {
    ierr = RearrangeCellsInAntiClockwiseDir(tdy,ivertex,cell_ids_abv_blw,ncells_abv,ncells_blw); CHKERRQ(ierr);
  } else if (ncells_blw > 1) {
    ierr = RearrangeCellsInAntiClockwiseDir(tdy,ivertex,cell_ids_abv_blw,ncells_abv,ncells_blw); CHKERRQ(ierr);
  }

  PetscInt **cell_traversal;
  ierr = TDyAllocate_IntegerArray_2D(&cell_traversal, 2, ncells+nfaces_bnd); CHKERRQ(ierr);
  PetscInt **cell_up2dw;
  ierr = TDyAllocate_IntegerArray_2D(&cell_up2dw, tdy->nfv, 2); CHKERRQ(ierr);

  PetscInt level, ncells_level;

  level = 0; ncells_level = ncells_abv;
  ierr = ConvertCellsAntiClockwiseDirInTraversalDir(tdy,ivertex,cell_ids_abv_blw[level],ncells_level,cell_traversal[level]); CHKERRQ(ierr);

  level = 1; ncells_level = ncells_blw;
  ierr = ConvertCellsAntiClockwiseDirInTraversalDir(tdy,ivertex,cell_ids_abv_blw[level],ncells_level,cell_traversal[level]); CHKERRQ(ierr);

  if (ncells_abv == 0 || ncells_blw == 0) {
    if (ncells_abv == 0) {
      level = 1;
      ncells_level = ncells_blw;
    } else {
      level = 0;
      ncells_level = ncells_abv;
    }
    ierr = AddUpDownBndFacesOfCellsInTraversalDirection(tdy,ivertex,cell_ids_abv_blw,ncells_level,level,cell_traversal); CHKERRQ(ierr);
  }

  PetscInt nUp2Dw;
  ierr = SetUpToDownConnections(tdy,ivertex,cell_traversal,cell_up2dw,&nUp2Dw); CHKERRQ(ierr);
  ierr = SetupUpwindFacesForSubcell(mesh,vertices,ivertex,cells,faces,subcells,cell_up2dw,nUp2Dw); CHKERRQ(ierr);

  ierr = TDyDeallocate_IntegerArray_2D(cell_traversal, 2); CHKERRQ(ierr);
  ierr = TDyDeallocate_IntegerArray_2D(cell_up2dw, tdy->nfv); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SaveCellIdsAtOppositeLevel(TDy tdy, PetscInt ivertex, PetscInt *icell_level, PetscInt ncells_level, PetscInt **icells_opposite_level) {

  PetscFunctionBegin;

  PetscInt ivertex_xy = VertexIdWithSameXYInACell(tdy, icell_level[0], ivertex);

  icells_opposite_level[ivertex_xy][0] = ncells_level;
  for (PetscInt ii=0; ii<ncells_level; ii++) {
    icells_opposite_level[ivertex_xy][ii+1] = icell_level[ii];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode FindCellsAboveAndBelowAVertex(TDy tdy, PetscInt ivertex, PetscInt **cellsAbove, PetscInt **cellsBelow) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  PetscInt ncells    = vertices->num_internal_cells[ivertex];
  PetscInt nfaces_bnd= vertices->num_boundary_faces[ivertex];
  PetscErrorCode ierr;

  PetscInt **cell_ids_abv_blw;
  ierr = TDyAllocate_IntegerArray_2D(&cell_ids_abv_blw, 2, ncells+nfaces_bnd); CHKERRQ(ierr);

  PetscInt ncells_abv = 0;
  PetscInt ncells_blw = 0;

  ierr = DetermineCellsAboveAndBelow(tdy,ivertex,cell_ids_abv_blw,&ncells_abv,&ncells_blw); CHKERRQ(ierr);

  // Save no. of cells
  cellsAbove[ivertex][0] = ncells_abv;
  cellsBelow[ivertex][0] = ncells_blw;

  // Save IDs of cells
  for (PetscInt icell = 0; icell<ncells_abv; icell++) {
    cellsAbove[ivertex][icell+1] = cell_ids_abv_blw[0][icell];
  }
  for (PetscInt icell = 0; icell<ncells_blw; icell++) {
    cellsBelow[ivertex][icell+1] = cell_ids_abv_blw[1][icell];
  }

  if (ncells_abv>0) {
    // Save the cells that are above the ivertex as the cells that are below the vertex that
    // is directly above the ivertex
    ierr = SaveCellIdsAtOppositeLevel(tdy, ivertex, cell_ids_abv_blw[0], ncells_abv, cellsBelow); CHKERRQ(ierr);
   }

   if (ncells_blw>0) {
    // Save the cells that are below the ivertex as the cells that are above the vertex that
    // is directly below the ivertex
     ierr = SaveCellIdsAtOppositeLevel(tdy, ivertex, cell_ids_abv_blw[1], ncells_blw, cellsAbove); CHKERRQ(ierr);
   }

  ierr = TDyDeallocate_IntegerArray_2D(cell_ids_abv_blw, 2); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode FindCellsAtUninitializedLevel (TDy tdy, PetscInt ivertex, PetscInt **icell_initialized, PetscInt **icell_uninitailized) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;

  PetscInt ncells = vertices->num_internal_cells[ivertex];

  // Are there more cells connected to ivertex than the no. of cells at the initialized level?
  if (ncells == icell_initialized[ivertex][0]) PetscFunctionReturn(0);

  // Set the no. of cells to zero
  icell_uninitailized[ivertex][0] = 0;

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];

  // Loop over all cells that are connected to the ivertex
  for (PetscInt ii=0; ii<ncells; ii++){

    PetscInt icell = vertices->internal_cell_ids[vOffsetIntCell+ii];
    PetscBool found = PETSC_FALSE;

    // Check if icell is one of the cells within the initialized cell list
    for (PetscInt jj=0; jj<icell_initialized[ivertex][0]; jj++) {
      if ( icell_initialized[ivertex][jj+1] == icell) {
        found = PETSC_TRUE;
        break;
      }
    }

    if (!found) {// Add the icell to the list of cells within the uninitalized cell list

     // Increase the count
     icell_uninitailized[ivertex][0]++;

     // Save the icell
     icell_uninitailized[ivertex][ icell_uninitailized[ivertex][0] ] =  vertices->internal_cell_ids[vOffsetIntCell+ii];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode FindCellsAboveAndBelowVertices(TDy tdy, PetscInt **cellsAbove, PetscInt **cellsBelow) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  PetscErrorCode ierr;

  for (PetscInt ivertex = 0; ivertex < mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    if (cellsAbove[ivertex][0] == -1 && cellsBelow[ivertex][0] == -1) {

      ierr = FindCellsAboveAndBelowAVertex(tdy, ivertex, cellsAbove, cellsBelow); CHKERRQ(ierr);

    } else if (cellsAbove[ivertex][0]  > -1 && cellsBelow[ivertex][0] == -1 ) {

      // Cells above the ivertex are initialized, while cells below the ivertex are uninitialized
      ierr = FindCellsAtUninitializedLevel(tdy, ivertex, cellsAbove, cellsBelow); CHKERRQ(ierr);

    } else if (cellsAbove[ivertex][0] == -1 && cellsBelow[ivertex][0]  > -1 ) {

      // Cells below the ivertex are initialized, while cells above the ivertex are uninitialized
      ierr = FindCellsAtUninitializedLevel(tdy, ivertex, cellsBelow, cellsAbove); CHKERRQ(ierr);
    }
  }

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

  DM dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDySubcell    *subcells = &mesh->subcells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);

  PetscInt dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (PetscInt icell=0; icell<c_end-c_start; icell++) {

    // save cell centroid
    PetscReal cell_cen[3], v_c[3];
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

    PetscInt num_subcells = cells->num_subcells[icell];

    PetscInt *vertex_ids, num_vertices;
    ierr = TDyMeshGetCellVertices(mesh, icell, &vertex_ids, &num_vertices); CHKERRQ(ierr);

    for (PetscInt isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      PetscInt ivertex = vertex_ids[isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

      PetscInt *subcell_face_ids, *subcell_vertex_ids, subcell_num_faces;
      PetscReal *subcell_face_areas;
      ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);
      ierr = TDyMeshGetSubcellVertices(mesh, subcell_id, &subcell_vertex_ids, &subcell_num_faces); CHKERRQ(ierr);
      ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &subcell_face_areas, &subcell_num_faces); CHKERRQ(ierr);

      // save coordindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertices, ivertex, dim, &v_c[0]); CHKERRQ(ierr);

      PetscInt num_shared_faces;

      // For a given cell, find all face ids that are share a vertex
      PetscInt f_idx[3];
      ierr = FindFaceIDsOfACellCommonToAVertex(cells->id[icell], faces, vertices, ivertex, f_idx, &num_shared_faces);

      // Update order of faces in f_idx so (face_ids[0], face_ids[1], face_ids[2])
      // form a plane such that normal to plane points toward the cell centroid
      ierr = UpdateFaceOrientationAroundAVertex(&cells->centroid[icell], faces, vertices, ivertex, dim, f_idx);
      for (PetscInt d=0;d<3;d++) {
        subcell_face_ids[d] = f_idx[d];
        subcell_vertex_ids[d] = ivertex;
      }

      PetscReal face_cen[3][3];
      PetscReal volume;

      for (PetscInt iface=0; iface<num_shared_faces; iface++) {
        PetscInt face_id = subcell_face_ids[iface];
        ierr = TDyFace_GetCentroid(faces, face_id, dim, &face_cen[iface][0]); CHKERRQ(ierr);
      }
      ierr = TDyComputeVolumeOfTetrahedron(cell_cen, face_cen[0], face_cen[1], face_cen[2], &volume); CHKERRQ(ierr);
      subcells->T[subcell_id] = volume*6.0;

      for (PetscInt iface=0; iface<num_shared_faces; iface++){

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

        PetscInt face_id = subcell_face_ids[iface];

        ierr = TDyFace_GetCentroid(faces, face_id, dim, &face_cen[iface][0]); CHKERRQ(ierr);

        PetscInt neighboring_vertex_ids[2];

        // Find 'n0' and 'n1'
        ierr = FindNeighboringVerticesOfAFace(faces,face_id,ivertex,neighboring_vertex_ids);

        PetscReal edge0_cen[3], edge1_cen[3];
        TDyCoordinate *subcell_face_centroids;
        ierr = TDyMeshGetSubcellFaceCentroids(mesh, subcell_id, &subcell_face_centroids, &subcell_num_faces); CHKERRQ(ierr);

        for (PetscInt d=0; d<dim; d++) {
          edge0_cen[d] = (v_c[d] + vertices->coordinate[neighboring_vertex_ids[0]].X[d])/2.0;
          edge1_cen[d] = (v_c[d] + vertices->coordinate[neighboring_vertex_ids[1]].X[d])/2.0;
          subcell_face_centroids[iface].X[d] = (v_c[d] + edge0_cen[d] + face_cen[iface][d] + edge1_cen[d])/4.0;
        }

        // The number of subfaces for each face is equal to number of vertices forming the face.
        // Thus, compute area of subface by dividing the area of face by number of vertices.
        PetscReal face_area;
        PetscInt num_vertices;
        ierr = TDyMeshGetFaceArea(mesh, face_id, &face_area); CHKERRQ(ierr);
        ierr = TDyMeshGetFaceNumVertices(mesh, face_id, &num_vertices); CHKERRQ(ierr);
        subcell_face_areas[iface] = face_area/num_vertices;

        // nu_vec on the "iface"-th is given as:
        //  = (x_{iface+1} - x_{cell_centroid}) x (x_{iface+2} - x_{cell_centroid})
        //  = (x_{f1_idx } - x_{cell_centroid}) x (x_{f2_idx } - x_{cell_centroid})
        PetscInt f1_idx = -1, f2_idx = -1, f3_idx = -1;

        // determin the f1_idx and f2_idx
        PetscReal f1[3], f2[3], f3[3];
        switch (iface) {
          case 0:
          f1_idx = 1;
          f2_idx = 2;
          f3_idx = 0;
          break;

          case 1:
          f1_idx = 2;
          f2_idx = 0;
          f3_idx = 1;
          break;

          case 2:
          f1_idx = 0;
          f2_idx = 1;
          f3_idx = 2;
        }

        // Save x_{f1_idx } and x_{f2_idx }
        PetscInt face_id_1 = subcell_face_ids[f1_idx];
        PetscInt face_id_2 = subcell_face_ids[f2_idx];
        PetscInt face_id_3 = subcell_face_ids[f3_idx];

        ierr = TDyFace_GetCentroid(faces, face_id_1, dim, &f1[0]); CHKERRQ(ierr);
        ierr = TDyFace_GetCentroid(faces, face_id_2, dim, &f2[0]); CHKERRQ(ierr);
        ierr = TDyFace_GetCentroid(faces, face_id_3, dim, &f3[0]); CHKERRQ(ierr);

        // Compute (x_{f1_idx } - x_{cell_centroid}) x (x_{f2_idx } - x_{cell_centroid})
        ierr = TDyNormalToTriangle(cell_cen, f1, f2, f_normal);

        // Save the data
        TDyVector *nu_vectors;
        PetscInt num_nu_vectors;
        ierr = TDyMeshGetSubcellNuVectors(mesh, subcell_id, &nu_vectors, &num_nu_vectors); CHKERRQ(ierr);

        for (PetscInt d=0; d<dim; d++) {
          nu_vectors[iface].V[d] = f_normal[d];
        }

        // Compute nu_star vector for TPF
        //
        // nu_star = T/dist * unit_normal_vec{x_neighbor, x_cell}
        // where
        //   dist = || x_cell - x_intercept||
        //   x_intercept is the intercept of the line joining x_cell and x_neighbor with the face
        PetscInt faceCellOffset = faces->cell_offset[face_id_3];
        PetscInt neighbor_cell_id;
        PetscReal neighbor_cell_cen[dim], dist;

        if (faces->cell_ids[faceCellOffset]==icell) {
          neighbor_cell_id = faces->cell_ids[faceCellOffset+1];
        } else {
          neighbor_cell_id = faces->cell_ids[faceCellOffset];
        }

        if (neighbor_cell_id >=0) {
          ierr = TDyCell_GetCentroid2(cells, neighbor_cell_id, dim, &neighbor_cell_cen[0]); CHKERRQ(ierr);
          ierr = TDyComputeLength(neighbor_cell_cen, cell_cen, dim, &dist); CHKERRQ(ierr);
          dist *= 0.50;
        } else {
          ierr = TDyFace_GetCentroid(faces, face_id_3, dim, &neighbor_cell_cen[0]); CHKERRQ(ierr);
          ierr = TDyComputeLength(neighbor_cell_cen, cell_cen, dim, &dist); CHKERRQ(ierr);
        }

        ierr = TDyUnitNormalVectorJoiningTwoVertices(neighbor_cell_cen, cell_cen, f_normal); CHKERRQ(ierr);

        PetscReal normal[3], dot_prod, value;
        for (PetscInt d=0; d<dim; d++) {
          normal[d] = faces->normal[face_id].V[d];
        }

        ierr = TDyDotProduct(normal,f_normal,&dot_prod); CHKERRQ(ierr);
        value = dot_prod;

        TDyVector *nu_star_vectors;
        PetscInt num_nu_star_vectors;
        ierr = TDyMeshGetSubcellNuStarVectors(mesh, subcell_id, &nu_star_vectors, &num_nu_star_vectors); CHKERRQ(ierr);

        for (PetscInt d=0; d<dim; d++) {
          nu_star_vectors[iface].V[d] = value*volume*6.0/dist;
        }
      }

    }
    PetscReal normal[3], centroid[3];
    ierr = DMPlexComputeCellGeometryFVM(dm, icell, &(cells->volume[icell]), &centroid[0],
                                        &normal[0]); CHKERRQ(ierr);
  }

  // Determine cell IDs that are above and below all vertices
  PetscInt **cellsAbove, **cellsBelow;
  PetscInt ncells_per_vertex = TDyMaxNumberOfCellsSharingAVertex(dm, tdy->closureSize, tdy->closure);

  ierr = TDyAllocate_IntegerArray_2D(&cellsAbove, mesh->num_vertices, ncells_per_vertex); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&cellsBelow, mesh->num_vertices, ncells_per_vertex); CHKERRQ(ierr);

  ierr = FindCellsAboveAndBelowVertices(tdy, cellsAbove, cellsBelow);

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    if (vertices->num_internal_cells[ivertex] > 1 && vertices->is_local[ivertex]) {
      ierr = DetermineUpwindFacesForSubcell_PlanarVerticalFaces(tdy, ivertex, cellsAbove, cellsBelow); CHKERRQ(ierr);
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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyVertex     *vertices = &mesh->vertices;
  TDyCell       *cells = &mesh->cells;
  TDyFace       *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    PetscReal v1[3], v2[3], v3[3], v4[3], normal[3];
    PetscReal f_cen[3], c_cen[3], f2c[3], dot_prod;

    PetscInt fOffsetVertex = faces->vertex_offset[iface];
    PetscInt fOffsetCell = faces->cell_offset[iface];

    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 0], dim, &v1[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 1], dim, &v2[0]); CHKERRQ(ierr);
    ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 2], dim, &v3[0]); CHKERRQ(ierr);

    if (faces->num_vertices[iface] == 4) {
      ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 3], dim, &v4[0]); CHKERRQ(ierr);

      // Check if v1-v2-v3-v4 are in the correct order such that
      // normal to (v1,v2,v3) and normal (v2,v3,v4) are in pointing
      // in the same direction
      PetscReal normal_123[3], normal_234[3];
      ierr = TDyUnitNormalToTriangle(v1, v2, v3, normal_123);  CHKERRQ(ierr);
      ierr = TDyUnitNormalToTriangle(v2, v3, v4, normal_234);  CHKERRQ(ierr);

      ierr = TDyDotProduct(normal_123,normal_234,&dot_prod); CHKERRQ(ierr);

      if (dot_prod < 0.0) {
        // Swap the order of vertices
        PetscInt tmp = faces->vertex_ids[fOffsetVertex + 2];
        faces->vertex_ids[fOffsetVertex + 2] = faces->vertex_ids[fOffsetVertex + 3];
        faces->vertex_ids[fOffsetVertex + 3] = tmp;
      }

      ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 2], dim, &v3[0]); CHKERRQ(ierr);
      ierr = TDyVertex_GetCoordinate(vertices, faces->vertex_ids[fOffsetVertex + 3], dim, &v4[0]); CHKERRQ(ierr);

    }

    ierr = TDyFace_GetCentroid(faces, iface, dim, &f_cen[0]); CHKERRQ(ierr);

    ierr = TDyCell_GetCentroid2(cells, faces->cell_ids[fOffsetCell+0], dim, &c_cen[0]); CHKERRQ(ierr);

    if (faces->num_vertices[iface] == 3) {
      ierr = TDyUnitNormalToTriangle(v1, v2, v3, normal);  CHKERRQ(ierr);
    } else {
      ierr = TDyNormalToQuadrilateral(v1, v2, v3, v4, normal); CHKERRQ(ierr);
    }
    for (PetscInt d=0; d<dim; d++){ faces->normal[iface].V[d] = normal[d];}

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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDySubcell    *subcells = &mesh->subcells;
  PetscErrorCode ierr;

  PetscInt       dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

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
  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {

    // save centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen_v[icell*dim]); CHKERRQ(ierr);

    // save volume
    cell_vol_v[icell] = cells->volume[icell];

    PetscInt *vertex_ids, num_vertices;
    PetscInt *neighbor_ids, num_neighbors;
    PetscInt *edge_ids, num_edges;
    ierr = TDyMeshGetCellVertices(mesh, icell, &vertex_ids, &num_vertices); CHKERRQ(ierr);
    ierr = TDyMeshGetCellNeighbors(mesh, icell, &neighbor_ids, &num_neighbors); CHKERRQ(ierr);
    ierr = TDyMeshGetCellEdges(mesh, icell, &edge_ids, &num_edges); CHKERRQ(ierr);

    for (PetscInt k=0; k<4; k++) {
      neigh_id_v [icell*4 + k] = neighbor_ids[k];
      vertex_id_v[icell*4 + k] = vertex_ids[k];
      edge_id_v  [icell*4 + k] = edge_ids[k];

      PetscInt subcell_id = icell*cells->num_subcells[icell]+k;

      scell_vol_v[icell*4 + k] = subcells->T[subcell_id];

      scell_gmatrix_v[icell*4*4 + k*4 + 0] = tdy->subc_Gmatrix[icell][k][0][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 1] = tdy->subc_Gmatrix[icell][k][0][1];
      scell_gmatrix_v[icell*4*4 + k*4 + 2] = tdy->subc_Gmatrix[icell][k][1][0];
      scell_gmatrix_v[icell*4*4 + k*4 + 3] = tdy->subc_Gmatrix[icell][k][1][1];

      ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, 0, dim, &scell_nu_v[count]); CHKERRQ(ierr);
      PetscInt sOffetNuVectors = subcells->nu_vector_offset[subcell_id];
      for (PetscInt d=0; d<dim; d++) {
        scell_cp_v[count] = subcells->variable_continuity_coordinates[sOffetNuVectors + 0].X[d];
        count++;
      }

      ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, 1, dim, &scell_nu_v[count]); CHKERRQ(ierr);
      for (PetscInt d=0; d<dim; d++) {
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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyEdge       *edges = &mesh->edges;
  PetscErrorCode ierr;

  PetscInt       dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  Vec edge_cen, edge_nor;
  PetscScalar *edge_cen_v, *edge_nor_v;

  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_cen);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_edges*dim, &edge_nor);
  CHKERRQ(ierr);

  ierr = VecGetArray(edge_cen, &edge_cen_v); CHKERRQ(ierr);
  ierr = VecGetArray(edge_nor, &edge_nor_v); CHKERRQ(ierr);

  for (PetscInt iedge=0; iedge<mesh->num_edges; iedge++) {
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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyVertex     *vertices = &mesh->vertices;
  PetscErrorCode ierr;

  PetscInt       dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

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

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    ierr = TDyVertex_GetCoordinate(vertices, ivertex, dim, &vert_coord_v[ivertex*dim]); CHKERRQ(ierr);
    PetscInt vOffsetIntCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetEdge    = vertices->edge_offset[ivertex];
    for (PetscInt i=0; i<4; i++) {
      vert_icell_ids_v[ivertex*4 + i]   = vertices->internal_cell_ids[vOffsetIntCell + i];
      vert_edge_ids_v[ivertex*4 + i]    = vertices->edge_ids[vOffsetSubcell+i];
      vert_subcell_ids_v[ivertex*4 + i] = vertices->subcell_ids[vOffsetEdge+i];
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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  PetscErrorCode ierr;

  PetscInt       dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  Vec tmat;
  ierr = VecCreateSeq(PETSC_COMM_SELF, mesh->num_vertices*5*5, &tmat);
  CHKERRQ(ierr);

  PetscScalar *tmat_v;
  ierr = VecGetArray(tmat, &tmat_v); CHKERRQ(ierr);

  PetscInt count = 0;
  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    for (PetscInt i=0; i<5; i++) {
      for (PetscInt j=0; j<5; j++) {
        tmat_v[count] = tdy->Trans[ivertex][i][j];
        count++;
      }
    }
  }

  ierr = VecRestoreArray(tmat, &tmat_v); CHKERRQ(ierr);

  ierr = VecDestroy(&tmat); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode Output2DMesh(TDy tdy) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  ierr = OutputCells2DMesh(tdy); CHKERRQ(ierr);
  ierr = OutputVertices2DMesh(tdy); CHKERRQ(ierr);
  ierr = OutputEdges2DMesh(tdy); CHKERRQ(ierr);
  ierr = OutputTransmissibilityMatrix2DMesh(tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyOutputMesh(TDy tdy) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt dim;
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
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  ierr = SaveMeshGeometricAttributes(tdy); CHKERRQ(ierr);
  ierr = SaveMeshConnectivityInfo(   tdy); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = UpdateCellOrientationAroundAVertex2DMesh(tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor2DMesh     (  tdy->dm, tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAEdge2DMesh(  tdy); CHKERRQ(ierr);
    break;

  case 3:
    ierr = ConvertCellsToCompressedFormat(tdy); CHKERRQ(ierr);
    ierr = ConvertVerticesToCompressedFormat(tdy); CHKERRQ(ierr);
    ierr = ConvertSubcellsToCompressedFormat(tdy); CHKERRQ(ierr);
    ierr = ConvertFacesToCompressedFormat(tdy); CHKERRQ(ierr);
    ierr = UpdateFaceOrderAroundAVertex3DMesh(tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAFace3DMesh(  tdy); CHKERRQ(ierr);
    break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in TDyBuildMesh");
    break;
  }

  ierr = IdentifyLocalCells(tdy); CHKERRQ(ierr);
  ierr = IdentifyLocalVertices(tdy); CHKERRQ(ierr);
  ierr = IdentifyLocalEdges(tdy); CHKERRQ(ierr);

  if (dim == 3) {
    ierr = IdentifyLocalFaces(tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor3DMesh(tdy); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
