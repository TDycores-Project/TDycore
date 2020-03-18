#include <petsc.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>
#include <private/tdymeshutilsimpl.h>

/* ---------------------------------------------------------------- */

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start,
                                PetscInt end) {
  return (closure >= start) && (closure < end);
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
  ierr = TDyAllocate_IntegerArray_1D(&cells->natural_id,num_cells); CHKERRQ(ierr);

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
  ierr = TDyAllocate_IntegerArray_1D(&vertices->subface_ids      ,num_vertices*num_faces ); CHKERRQ(ierr);
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
  mesh->num_subcells = cNum*num_subcells;
  ierr = AllocateMemoryForSubcells(cNum, num_subcells, subcell_type, &mesh->subcells); CHKERRQ(ierr);

  PetscFunctionReturn(0);

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

  PetscBool useNatural;
  ierr = DMGetUseNatural(dm, &useNatural); CHKERRQ(ierr);
  if (useNatural) {
    Vec            global, local, natural;
    PetscScalar   *p;
    PetscInt       size_natural, cumsize_natural, size_local, ii;
    PetscInt num_fields;

    ierr = DMGetNumFields(dm, &num_fields); CHKERRQ(ierr);

    // Create the natural vector
    ierr = DMCreateGlobalVector(dm, &natural);
    ierr = VecGetLocalSize(natural, &size_natural);
    ierr = MPI_Scan(&size_natural, &cumsize_natural, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

    // Add entries in the natural vector
    ierr = VecGetArray(natural, &p); CHKERRQ(ierr);
    for (ii = 0; ii < size_natural; ++ii) {
      if (ii % num_fields == 0)
        p[ii] = ii + cumsize_natural/num_fields - size_natural/num_fields;
      else
        p[ii] = 0;
    }
    ierr = VecRestoreArray(natural, &p); CHKERRQ(ierr);

    // Map natural IDs in global order
    ierr = DMCreateGlobalVector(dm, &global);CHKERRQ(ierr);
    ierr = DMPlexNaturalToGlobalBegin(dm, natural, global);CHKERRQ(ierr);
    ierr = DMPlexNaturalToGlobalEnd(dm, natural, global);CHKERRQ(ierr);

    // Map natural IDs in local order
    ierr = DMCreateLocalVector(dm, &local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local); CHKERRQ(ierr);

    // Save natural IDs
    ierr = VecGetLocalSize(local, &size_local);
    ierr = VecGetArray(local, &p); CHKERRQ(ierr);
    for (ii = 0; ii < size_local/num_fields; ++ii) cells->natural_id[ii] = p[ii*num_fields];
    ierr = VecRestoreArray(local, &p); CHKERRQ(ierr);

    // Cleanup
    ierr = VecDestroy(&natural); CHKERRQ(ierr);
    ierr = VecDestroy(&global); CHKERRQ(ierr);
    ierr = VecDestroy(&local); CHKERRQ(ierr);
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
        PetscInt cOffsetVert = cells->vertex_offset[icell];
        cells->vertex_ids[cOffsetVert + c2vCount] = ivertex ;

        PetscInt vOffsetCell = vertices->internal_cell_offset[ivertex];
        PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

        for (j=0; j<nverts_per_cell; j++) {
          if (vertices->internal_cell_ids[vOffsetCell + j] == -1) {
            vertices->num_internal_cells[ivertex]++;
            vertices->internal_cell_ids[vOffsetCell + j] = icell;
            vertices->subcell_ids[vOffsetSubcell + j]    = c2vCount;
            break;
          }
        }
        c2vCount++;
      } else if (IsClosureWithinBounds(tdy->closure[icell][i], eStart,
                                       eEnd)) { /* Is the closure an edge? */
        PetscInt iedge = tdy->closure[icell][i] - eStart;
        PetscInt cOffsetEdge = cells->edge_offset[icell];
        cells->edge_ids[cOffsetEdge + c2eCount] = iedge;
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
        PetscInt cOffsetFace = cells->face_offset[icell];
        PetscInt fOffsetCell = faces->cell_offset[iface];
        cells->face_ids[cOffsetFace + c2fCount] = iface;
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
      PetscInt vOffsetEdge = vertices->edge_offset[ivertex];
      vertices->edge_ids[vOffsetEdge + s] = iedge;
      if (!edges->is_internal[iedge]) vertices->num_boundary_cells[ivertex]++;
    }
  }

  PetscInt f;
  for (f=fStart; f<fEnd; f++){
    PetscInt iface = f-fStart;
    PetscInt fOffsetEdge = faces->edge_offset[iface];
    PetscInt fOffsetVertex = faces->vertex_offset[iface];

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
        PetscInt vOffsetFace = vertices->face_offset[ivertex];
        PetscInt ii;
        for (ii=0; ii<vertices->num_faces[ivertex]; ii++) {
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
    ierr = DMPlexGetSupportSize(dm, f, &supportSize); CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, f, &support); CHKERRQ(ierr);

    if (supportSize == 2) faces->is_internal[iface] = PETSC_TRUE;
    else                  faces->is_internal[iface] = PETSC_FALSE;

    for (s=0; s<supportSize; s++) {
      icell = support[s] - cStart;
        PetscBool found = PETSC_FALSE;
        PetscInt ii;
        PetscInt cOffsetFace = cells->face_offset[icell];
        for (ii=0; ii<cells->num_faces[icell]; ii++) {
          if (cells->face_ids[cOffsetFace+ii] == f-fStart) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          cells->face_ids[cOffsetFace + cells->num_faces[icell]] = f-fStart;
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

  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt vOffsetEdge    = vertices->edge_offset[ivertex];

  // compute angle to all cell centroids w.r.t. the shared vertix
  for (i=0; i<ncells; i++) {
    icell = vertices->internal_cell_ids[vOffsetCell + i];

    x = cells->centroid[icell].X[0] - vertices->coordinate[ivertex].X[0];
    y = cells->centroid[icell].X[1] - vertices->coordinate[ivertex].X[1];

    ids[count]              = icell;
    idx[count]              = count;
    subcell_ids[count]      = vertices->subcell_ids[vOffsetSubcell + i];
    is_cell[count]          = PETSC_TRUE;
    is_internal_edge[count] = PETSC_FALSE;

    ierr = ComputeTheta(x, y, &theta[count]);

    count++;
  }

  // compute angle to face centroids w.r.t. the shared vertix
  boundary_edge_present = PETSC_FALSE;

  for (i=0; i<nedges; i++) {
    iedge = vertices->edge_ids[vOffsetEdge + i];
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
    vertices->internal_cell_ids[vOffsetCell + i] = tmp_cell_ids[i];
    vertices->subcell_ids[vOffsetSubcell + i]       = tmp_subcell_ids[i];
  }

  // save information about sorted edge ids
  for (i=0; i<tmp_nedges; i++) {
    vertices->edge_ids[vOffsetEdge + i] = tmp_edge_ids[i];
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

    PetscInt vOffsetEdge = vertices->edge_offset[ivertex];

    if (vertices->num_internal_cells[ivertex] > 1) {
      ierr = UpdateCellOrientationAroundAVertex(tdy, ivertex); CHKERRQ(ierr);
    } else {

      edge_id_1 = vertices->edge_ids[vOffsetEdge + 0];
      edge_id_2 = vertices->edge_ids[vOffsetEdge + 1];

      x = edges->centroid[edge_id_1].X[0] - vertices->coordinate[ivertex].X[0];
      y = edges->centroid[edge_id_1].X[1] - vertices->coordinate[ivertex].X[1];
      ierr = ComputeTheta(x, y, &theta_1);

      x = edges->centroid[edge_id_2].X[0] - vertices->coordinate[ivertex].X[0];
      y = edges->centroid[edge_id_2].X[1] - vertices->coordinate[ivertex].X[1];
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
    PetscInt subface_ids_sorted[vertices->num_faces[ivertex]];
    PetscInt vOffsetFace = vertices->face_offset[ivertex];
    PetscInt count=0, iface;

    // First find all internal faces (i.e. face shared by two cells)
    for (iface=0;iface<vertices->num_faces[ivertex];iface++) {
      PetscInt face_id = vertices->face_ids[vOffsetFace + iface];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + iface];
      if (faces->num_cells[face_id]==2) {
        face_ids_sorted[count] = face_id;
        subface_ids_sorted[count] = subface_id;
        count++;
      }
    }
    
    // Now find all boundary faces (i.e. face shared by a single cell)
    for (iface=0;iface<vertices->num_faces[ivertex];iface++) {
      PetscInt face_id = vertices->face_ids[vOffsetFace+iface];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace+iface];
      if (faces->num_cells[face_id]==1) {
        face_ids_sorted[count] = face_id;
        subface_ids_sorted[count] = subface_id;
        count++;
      }
    }

    // Save the sorted faces
    for (iface=0;iface<vertices->num_faces[ivertex];iface++) {
      vertices->face_ids[vOffsetFace+iface] = face_ids_sorted[iface];
      vertices->subface_ids[vOffsetFace+iface] = subface_ids_sorted[iface];
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
    PetscInt eOffsetCell = edges->cell_offset[iedge];

    if (edges->is_internal[iedge]) {
      TDy_coordinate *cell_from_centroid = &cells->centroid[edges->cell_ids[eOffsetCell + 0]];
      TDy_coordinate *cell_to_centroid   = &cells->centroid[edges->cell_ids[eOffsetCell + 1]];

      dot_product = (cell_to_centroid->X[0] - cell_from_centroid->X[0]) * edges->normal[iedge].V[0] +
                    (cell_to_centroid->X[1] - cell_from_centroid->X[1]) * edges->normal[iedge].V[1];
      if (dot_product < 0.0) {
        PetscInt tmp = edges->cell_ids[eOffsetCell + 0];
        edges->cell_ids[eOffsetCell + 0] = edges->cell_ids[eOffsetCell + 1];
        edges->cell_ids[eOffsetCell + 1] = tmp;
      }
    } else {
      TDy_coordinate *cell_from_centroid = &cells->centroid[edges->cell_ids[eOffsetCell + 0]];

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

    // save cell centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cells->num_subcells[icell];

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      PetscInt cOffsetVertex = cells->vertex_offset[icell];

      PetscInt subcell_id = icell*num_subcells+isubcell;
      PetscInt sOffsetNuVectors = subcells->nu_vector_offset[subcell_id];

      // save coorindates of vertex that is part of the subcell
      ierr = TDyVertex_GetCoordinate(vertices, cells->vertex_ids[cOffsetVertex+isubcell], dim, &v_c[0]); CHKERRQ(ierr);

      // determine ids of up & down edges
      PetscInt cOffsetEdge = cells->edge_offset[icell];
      e_idx_up = cells->edge_ids[cOffsetEdge + isubcell];

      if (isubcell == 0) e_idx_dn = cells->edge_ids[cOffsetEdge + num_subcells-1];
      else               e_idx_dn = cells->edge_ids[cOffsetEdge + isubcell    -1];

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
PetscErrorCode ExtractCentroidForIthJthTraversalOrder(TDy_vertex *vertices, PetscInt ivertex, TDy_face *faces, TDy_cell *cells, PetscInt cell_traversal_ij, PetscReal cen[3]) {

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
PetscErrorCode SetupUpwindFacesForSubcell(TDy_vertex *vertices, PetscInt ivertex, TDy_cell *cells, TDy_face *faces, TDy_subcell *subcells, PetscInt **cell_up2dw) {

  PetscFunctionBegin;

  PetscInt icell, isubcell, iface, ncells, cell_id, nflux_in, ncells_bnd;
  PetscInt ii, boundary_cell_count;
  PetscInt vOffsetBoundaryFace, vOffsetFace;
  
  ncells = vertices->num_internal_cells[ivertex];
  ncells_bnd= vertices->num_boundary_cells[ivertex];
  vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];
  vOffsetFace = vertices->face_offset[ivertex];

  boundary_cell_count = 0;

  // Compute number of fluxes through internal faces
  nflux_in = 0;
  for (iface=0; iface<vertices->num_faces[ivertex]; iface++) {
    PetscInt faceID = vertices->face_ids[vOffsetFace+iface];
    if (faces->is_internal[faceID]) nflux_in++;
  }

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  vOffsetFace = vertices->face_offset[ivertex];

  for (icell=0; icell<ncells; icell++) {

    // Determine the cell and subcell id
    cell_id  = vertices->internal_cell_ids[vOffsetIntCell + icell];
    isubcell = vertices->subcell_ids[vOffsetSubcell + icell];

    // Get access to the cell and subcell
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    // Loop over all faces of the subcell
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Boundary face
      if (faces->cell_ids[fOffsetCell + 0] < 0 || faces->cell_ids[fOffsetCell + 1] < 0) {
        subcells->face_unknown_idx[sOffsetFace + iface] = boundary_cell_count+nflux_in;
        
        PetscInt face_id_local;
        for (ii=0; ii<ncells_bnd; ii++) {
          if (vertices->boundary_face_ids[vOffsetBoundaryFace + ii] == face_id) {
            face_id_local = ncells + ii;
            break;
          }
        }

        for (ii=0; ii<12; ii++) {
          if (cell_up2dw[ii][0] == face_id_local && cell_up2dw[ii][1] == icell){ subcells->is_face_up[sOffsetFace + iface] = PETSC_TRUE; break;}
          if (cell_up2dw[ii][1] == face_id_local && cell_up2dw[ii][0] == icell){ subcells->is_face_up[sOffsetFace + iface] = PETSC_FALSE; break;}
        }
        boundary_cell_count++;

      } else {

        // Find the index of cells given by faces->cell_ids[fOffsetCell + 0:1] within the cell id list given
        // vertex->internal_cell_ids
        PetscInt cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[vOffsetIntCell], ncells, faces->cell_ids[fOffsetCell + 0]);
        PetscInt cell_2 = TDyReturnIndexInList(&vertices->internal_cell_ids[vOffsetIntCell], ncells, faces->cell_ids[fOffsetCell + 1]);

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
  }

  PetscInt nup_bnd_flux=0, ndn_bnd_flux=0;

  PetscInt nflux_bc_up = 0;
  PetscInt nflux_bc_dn = 0;
  PetscInt i;
  PetscInt vOffsetCell = vertices->internal_cell_offset[ivertex];
  for (i=0; i<ncells; i++) {
    icell    = vertices->internal_cell_ids[vOffsetCell + i];
    isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    PetscInt iface;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt faceID = subcells->face_ids[sOffsetFace+iface];
      if (faces->is_internal[faceID]) continue;

      PetscBool upwind_entries;
      upwind_entries = (subcells->is_face_up[sOffsetFace + iface]==1);

      if (upwind_entries) nflux_bc_up++;
      else                nflux_bc_dn++;
    }

  }


  // Save the face index that corresponds to the flux in transmissibility matrix
  for (icell=0; icell<ncells; icell++) {

    // Determine the cell and subcell id
    cell_id  = vertices->internal_cell_ids[vOffsetIntCell+icell];
    isubcell = vertices->subcell_ids[vOffsetSubcell+icell];

    // Get access to the cell and subcell
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
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];

      PetscInt idx_flux = subcells->face_unknown_idx[sOffsetFace + iface];
      if (faces->is_internal[face_id]) {
        vertices->face_ids[vOffsetFace + idx_flux] = face_id;
        subcells->face_flux_idx[sOffsetFace + iface] = idx_flux;
      } else {
        if (subcells->is_face_up[sOffsetFace + iface]) {
          vertices->face_ids[vOffsetFace + nflux_in + nup_bnd_flux] = face_id;
          subcells->face_flux_idx[sOffsetFace + iface] = nflux_in+nup_bnd_flux;
          nup_bnd_flux++;
        } else {
          vertices->face_ids[vOffsetFace + nflux_in + nflux_bc_up + ndn_bnd_flux] = face_id;
          subcells->face_flux_idx[sOffsetFace + iface] = nflux_in+ndn_bnd_flux;
          ndn_bnd_flux++;
        }
      }

    }
  }

  // Since vertices->face_ids[] has been updated, now
  // update vertices->subface_ids[]
  for (iface=0; iface<vertices->num_faces[ivertex]; iface++) {
    PetscInt face_id = vertices->face_ids[vOffsetFace+iface];
    PetscInt fOffsetVertex = faces->vertex_offset[face_id];
    PetscBool found = PETSC_FALSE;
    for (ii=0; ii<faces->num_vertices[face_id]; ii++) {
      if (faces->vertex_ids[fOffsetVertex + ii] == ivertex) {
        vertices->subface_ids[vOffsetFace + iface] = ii;
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

  TDy_vertex *vertices;
  PetscBool sameXY = PETSC_FALSE;
  PetscReal dist = 0.0, eps = 1.e-14;
  PetscInt d;
  
  vertices = &tdy->mesh->vertices;

  for (d=0;d<2;d++)
    dist += PetscSqr(vertices->coordinate[ivertex_1].X[d] - vertices->coordinate[ivertex_2].X[d]);

  if (dist<eps) sameXY = PETSC_TRUE;

  PetscFunctionReturn(sameXY);
}


/* -------------------------------------------------------------------------- */
PetscBool IsCellAboveTheVertex(TDy tdy, PetscInt icell, PetscInt ivertex) {

  PetscFunctionBegin;
  
  PetscBool is_above;

  TDy_cell *cells;
  TDy_vertex *vertices;
  PetscInt cOffsetVert, iv;
  PetscBool found = PETSC_FALSE;
  
  cells = &tdy->mesh->cells;
  vertices = &tdy->mesh->vertices;

  cOffsetVert = cells->vertex_offset[icell];

  for (iv=0; iv<cells->num_vertices[icell]; iv++) {

    PetscInt ivertex_2 = cells->vertex_ids[cOffsetVert+iv];

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

  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  PetscInt icell, ncells_int;

  cells = &tdy->mesh->cells;
  faces = &tdy->mesh->faces;
  vertices = &tdy->mesh->vertices;

  ncells_int = vertices->num_internal_cells[ivertex];

  *ncells_abv = 0;
  *ncells_blw = 0;

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
  for (icell=0; icell<ncells_int; icell++) {
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
  PetscInt ii,iv,iface;
  PetscInt level, ncells_level;

  for (level=0; level<2; level++) {
    if (level==0) ncells_level = *ncells_abv;
    else          ncells_level = *ncells_blw;

    // Skip if there is only one cell
    if (ncells_level<2) continue;

    PetscInt ivertexAbvBlw[ncells_level];
    PetscInt IsBoundaryfaceIDsAbvBlw[ncells_level][2];

    PetscInt maxBndFaces = 0;
    for (ii=0; ii<ncells_level; ii++) {
      PetscInt cellID = cellsAbvBlw[level][ii];
      PetscInt cOffsetVert = cells->vertex_offset[cellID];
      PetscBool found = PETSC_FALSE;

      // Find the vertex that is above/below ivertex
      for (iv=0; iv<cells->num_vertices[cellID]; iv++) {
        ivertexAbvBlw[level] = cells->vertex_ids[cOffsetVert+iv];
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
      PetscInt faceCount = 0;
      for (iface=0; iface<cells->num_faces[cellID]; iface++) {
        PetscInt cOffsetFace = cells->face_offset[cellID];
        PetscInt faceID = cells->face_ids[cOffsetFace+iface];
        PetscInt fOffsetVert = faces->vertex_offset[faceID];

        // Does the faceID-th face contains vertices corresponding to ivertex and ivertexAbvBlw?
        PetscInt count=0;
        for (iv=0; iv<faces->num_vertices[faceID]; iv++) {
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
      if (IsBoundaryfaceIDsAbvBlw[ii][0]+IsBoundaryfaceIDsAbvBlw[ii][1] > maxBndFaces)
        maxBndFaces = IsBoundaryfaceIDsAbvBlw[ii][0]+IsBoundaryfaceIDsAbvBlw[ii][1];
    }

    // Rearrange the list only if there is a boundary face
    if (maxBndFaces>0) {
      PetscInt cellsAbvBlw_rearranged[ncells_level];
      PetscInt count_bnd=0,count_int=1;

      for (icell=0; icell<ncells_level; icell++) {

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
      for (icell=0; icell<ncells_level; icell++) cellsAbvBlw[level][icell] = cellsAbvBlw_rearranged[icell];
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

  TDy_cell *cells;
  PetscInt ii,jj;
  PetscBool are_neighbors = PETSC_FALSE;

  cells = &tdy->mesh->cells;

  PetscInt cOffsetFace1 = cells->face_offset[cell_id_1];
  PetscInt cOffsetFace2 = cells->face_offset[cell_id_2];

  for (ii=0; ii<cells->num_faces[cell_id_1]; ii++) {
    for (jj=0; jj<cells->num_faces[cell_id_2]; jj++) {
      if (cells->face_ids[cOffsetFace1+ii] == cells->face_ids[cOffsetFace2+jj] ) {
        are_neighbors = PETSC_TRUE;
        break;
      }
    }
  }

  PetscFunctionReturn(are_neighbors);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode RearrangeCellsInAntiClockwiseDir(TDy tdy, PetscInt ivertex, PetscInt **cellsAbvBlw,
                PetscInt ncells_abv, PetscInt ncells_blw) {

  PetscFunctionBegin;

  TDy_cell *cells;
  TDy_vertex *vertices;
  TDy_face *faces;
  PetscInt ii,jj,aa,ncells;
  PetscBool found;

  cells = &tdy->mesh->cells;
  vertices = &tdy->mesh->vertices;
  faces = &tdy->mesh->faces;

  if (ncells_abv>0) {
    ncells = ncells_abv;
    aa = 0;
  } else {
    ncells = ncells_blw;
    aa = 1;
  }

  PetscInt cell_order[ncells];
  PetscInt cell_used[ncells];

  for (ii=0; ii<ncells; ii++) cell_used[ii] = 0;

  // First put cells in an order that could be clockwise or anticlockwise
  cell_order[0] = cellsAbvBlw[aa][0];
  cell_used[0] = 1;
  
  for (ii=0; ii<ncells-1; ii++) {
    // For ii-th cell find a neighboring cell that hasn't been
    // previously identified it's neigbhor.

    found = PETSC_FALSE;

    for (jj=0; jj<ncells; jj++) {
    
      if (cell_used[jj] == 0) {
    
        if (AreCellsNeighbors(tdy, cell_order[ii], cellsAbvBlw[aa][jj])) {
          cell_order[ii+1] = cellsAbvBlw[aa][jj];
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

  // Find the face that is shared by cell_order[0] and cell_order[1]
  PetscInt vOffsetFace, iface, face_id;
  vOffsetFace = vertices->face_offset[ivertex];

  for (iface=0; iface<vertices->num_faces[ivertex]; iface++) {
    face_id = vertices->face_ids[vOffsetFace+iface];

    found = PETSC_FALSE;
    PetscInt fOffsetCell = faces->cell_offset[face_id];

    if (
        (faces->cell_ids[fOffsetCell  ] == cell_order[0] || faces->cell_ids[fOffsetCell  ]  == cell_order[1] ) &&
        (faces->cell_ids[fOffsetCell+1] == cell_order[0] || faces->cell_ids[fOffsetCell+1]  == cell_order[1] )
       ) {
      found = PETSC_TRUE;
      break;
    }
  }

  // Now rearrnge the cell order to be anticlockwise direction
  //   vec_a = (centroid_0 - ivertex)
  //   vec_b = (face_01    - ivertex)
  //
  // If dotprod(vec_a,vec_b) > 0, the cell_order is in anticlockwise
  // else flip the cell_order
  //
  PetscReal a[3], b[3], axb[3];
  PetscInt d, dim=2;
  PetscErrorCode ierr;

  if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a shared face");

  for (d=0; d<dim; d++) {
    a[d] = cells->centroid[cell_order[0]].X[d] - vertices->coordinate[ivertex].X[d];
    b[d] = faces->centroid[face_id      ].X[d] - vertices->coordinate[ivertex].X[d];
  }
  a[2] = 0.0;
  b[2] = 0.0;

  ierr = TDyCrossProduct(a,b,axb); CHKERRQ(ierr);

  if (axb[2]>0) {
    // Cells are in anticlockwise direction, so copy the ids
    for (ii=0; ii<ncells; ii++) cellsAbvBlw[aa][ii] = cell_order[ii];
  } else {
    // Cells are in clockwise direction, so flip the order
    for (ii=0; ii<ncells; ii++) cellsAbvBlw[aa][ii] = cell_order[ncells-ii-1];
  }

  if (ncells_abv>0 && ncells_blw>0) {
    // 1. Cells are present above and below the ivertex, and
    // 2. Initially cells above the vertex were sorted.
    // So, for each cell above the vertex, find the corresponding
    // cell below the vertex
    for (ii=0; ii<ncells_abv; ii++) {
      found = PETSC_FALSE;
      for (jj=0; jj<ncells_blw; jj++) {
        if (AreCellsNeighbors(tdy, cellsAbvBlw[0][ii], cellsAbvBlw[1][jj])) {
          cell_order[ii] = cellsAbvBlw[1][jj];
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a corresponding cell below the given cell");
    }

    // Update the order of cells below the vertex
    for (ii=0; ii<ncells_blw; ii++) cellsAbvBlw[1][ii] = cell_order[ii];
  }

  PetscFunctionReturn(0);

}


/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTraversalDirection(TDy tdy, PetscInt ivertex, PetscInt **cell_ids_abv_blw,
  PetscInt ncells_level, PetscInt level, PetscInt **cell_traversal){
  
  PetscFunctionBegin;

  TDy_face *faces;
  TDy_cell *cells;
  TDy_vertex *vertices;

  PetscInt ncells,ncells_bnd;

  cells = &tdy->mesh->cells;
  faces = &tdy->mesh->faces;
  vertices = &tdy->mesh->vertices;

  ncells    = vertices->num_internal_cells[ivertex];
  ncells_bnd= vertices->num_boundary_cells[ivertex];

  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  PetscInt ii,jj;
  PetscBool found;
  PetscInt bnd_faces_found[ncells_bnd];
  
  for (ii=0; ii<ncells_bnd; ii++) bnd_faces_found[ii] = 0;

  // Values of cell_traversal[0:1][:] should corresponds to cell/face IDs in
  // local numbering

  if (ncells_level>0) {

    // First find cell IDs in local numbering
    for (ii=0; ii<ncells_level; ii++) {
      found = PETSC_FALSE;
      for (jj=0; jj<ncells; jj++) {
        PetscInt cellID = vertices->internal_cell_ids[vOffsetIntCell + jj];
        if (cellID == cell_ids_abv_blw[level][ii]) {
          cell_traversal[level][ii] = jj;
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a the cell in the list");
    }


    if (ncells_level>1) {
      PetscInt kk,mm;
      PetscBool found;
      
      // For the first and the last cell, check if there is a boundary face
      // that corresponds to the anticlockwise traversal direction.
      for (kk=0;kk<2;kk++) {
        // Select the mm-th (first or last) cell
        if (kk == 0) mm = ncells_level-1;
        else         mm = 0;
        
        found = PETSC_FALSE;
        
        // Loop through all boundary face
        for (ii=0; ii<ncells_bnd; ii++) {
          
          // Skip the boundary face that has been previously identified
          if (bnd_faces_found[ii]) continue;
          
          PetscInt face_id = vertices->boundary_face_ids[vOffsetBoundaryFace + ii];
          PetscInt fOffsetCell = faces->cell_offset[face_id];
          
          // Check if the face belongs to mm-th cell
          if ((cell_ids_abv_blw[level][mm] == faces->cell_ids[fOffsetCell]  )||
              (cell_ids_abv_blw[level][mm] == faces->cell_ids[fOffsetCell+1])) {
            PetscInt fOffsetVertex = faces->vertex_offset[face_id];
            
            // Check if the face has a vertex that is exactly above or below the
            // ivertex
            for (jj=0; jj<faces->num_vertices[face_id]; jj++) {
              if ( (ivertex != faces->vertex_ids[fOffsetVertex + jj])
                  && VerticesHaveSameXYCoords(tdy, ivertex, faces->vertex_ids[fOffsetVertex + jj])) {
                found = PETSC_TRUE;
                bnd_faces_found[ii] = 1;
                cell_traversal[level][ncells_level+kk] = ncells+ ii;
                break;
              }
            }
            if (found) break;
          }
        }
      }
    } else {
      PetscInt mm = 0, count=0, face_id_1, ii_1, ii_2;
      for (ii=0; ii<ncells_bnd; ii++) {
        PetscInt face_id = vertices->boundary_face_ids[vOffsetBoundaryFace + ii];
        PetscInt fOffsetCell = faces->cell_offset[face_id];
        if ((cell_ids_abv_blw[level][mm] == faces->cell_ids[fOffsetCell]  )||
            (cell_ids_abv_blw[level][mm] == faces->cell_ids[fOffsetCell+1])) {
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
      
      
      PetscReal a[3], b[3], axb[3];
      PetscInt d, dim=2;
      PetscErrorCode ierr;
      PetscInt cellID = cell_ids_abv_blw[level][mm];

      for (ii=0; ii<ncells; ii++) {
        if (cellID == vertices->internal_cell_ids[vOffsetIntCell + ii]) {
          cell_traversal[level][0] = ii;
        }
      }

      for (d=0; d<dim; d++) {
        a[d] = cells->centroid[cellID   ].X[d] - vertices->coordinate[ivertex].X[d];
        b[d] = faces->centroid[face_id_1].X[d] - vertices->coordinate[ivertex].X[d];
      }
      a[2] = 0.0;
      b[2] = 0.0;
      
      ierr = TDyCrossProduct(a,b,axb); CHKERRQ(ierr);
      if (axb[2]>0) {
        cell_traversal[level][1] = ii_1;
        cell_traversal[level][2] = ii_2;
      } else {
        cell_traversal[level][1] = ii_2;
        cell_traversal[level][2] = ii_1;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeBoundaryFaceTraversalDirection(TDy tdy, PetscInt ivertex, PetscInt **cell_ids_abv_blw,
  PetscInt ncells_level, PetscInt level, PetscInt **cell_traversal){
  
  PetscFunctionBegin;

  TDy_face *faces;
  TDy_vertex *vertices;

  PetscInt ncells,ncells_bnd;

  faces = &tdy->mesh->faces;
  vertices = &tdy->mesh->vertices;

  ncells    = vertices->num_internal_cells[ivertex];
  ncells_bnd= vertices->num_boundary_cells[ivertex];

  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  PetscInt ii,jj,level_faces;
  PetscInt bnd_faces_found[ncells_bnd];
  
  if (level == 0) level_faces = 1;
  else            level_faces = 0;

  // Identify the boundary faces that have already been used
  for (ii=0; ii<ncells_bnd; ii++) bnd_faces_found[ii] = 0;
  for (ii=0; ii<ncells+ncells_bnd; ii++) {
    jj = cell_traversal[level][ii];
    if (jj >= ncells) bnd_faces_found[jj-ncells] = 1;
  }

  // For each cell at the given 'level', find a corresponding top/bottom face
  for (ii=0; ii<ncells_level; ii++) {
    for (jj=0; jj<ncells_bnd; jj++) {
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
PetscErrorCode DetermineUpwindFacesForSubcell_PlanarVerticalFaces(TDy tdy, PetscInt ivertex) {

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

  TDy_face *faces;
  TDy_cell *cells;
  TDy_subcell *subcells;
  TDy_vertex *vertices;

  PetscInt ncells,ncells_bnd;
  PetscInt **cell_traversal;
  PetscInt count;
  PetscInt ncells_abv,ncells_blw;
  PetscInt **cell_up2dw;
  PetscErrorCode ierr;

  cells = &tdy->mesh->cells;
  faces = &tdy->mesh->faces;
  subcells = &tdy->mesh->subcells;
  vertices = &tdy->mesh->vertices;

  ncells    = vertices->num_internal_cells[ivertex];
  ncells_bnd= vertices->num_boundary_cells[ivertex];

  PetscInt icell, cell_id, isubcell, iface, bnd_count=0;
  PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt vOffsetBoundaryFace = vertices->boundary_face_offset[ivertex];

  // For each vertex, determine boundary face IDs
  for (icell=0; icell<ncells; icell++) {
    // Determine the cell and subcell id
    cell_id  = vertices->internal_cell_ids[vOffsetIntCell + icell];
    isubcell = vertices->subcell_ids[vOffsetSubcell + icell];
    // Get access to the cell and subcell
    PetscInt subcell_id = cell_id*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    // Loop over all faces of the subcell
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Boundary face
      if (faces->cell_ids[fOffsetCell + 0] < 0 || faces->cell_ids[fOffsetCell + 1] < 0) {
        vertices->boundary_face_ids[vOffsetBoundaryFace + bnd_count] = face_id;
        bnd_count++;
      }
    }
  }

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

  ierr = TDyAllocate_IntegerArray_2D(&cell_ids_abv_blw, 2, ncells+ncells_bnd); CHKERRQ(ierr);

  ncells_abv = 0;
  ncells_blw = 0;

  ierr = DetermineCellsAboveAndBelow(tdy,ivertex,cell_ids_abv_blw,&ncells_abv,&ncells_blw); CHKERRQ(ierr);
  if (ncells_abv > 1) {
    ierr = RearrangeCellsInAntiClockwiseDir(tdy,ivertex,cell_ids_abv_blw,ncells_abv,ncells_blw); CHKERRQ(ierr);
  } else if (ncells_blw > 1) {
    ierr = RearrangeCellsInAntiClockwiseDir(tdy,ivertex,cell_ids_abv_blw,ncells_abv,ncells_blw); CHKERRQ(ierr);
  }

  ierr = TDyAllocate_IntegerArray_2D(&cell_traversal, 2, ncells+ncells_bnd); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&cell_up2dw, 12, 2); CHKERRQ(ierr);

  PetscInt level, ncells_level;

  level = 0; ncells_level = ncells_abv;
  ierr = ComputeTraversalDirection(tdy,ivertex,cell_ids_abv_blw,ncells_level,level,cell_traversal); CHKERRQ(ierr);

  level = 1; ncells_level = ncells_blw;
  ierr = ComputeTraversalDirection(tdy,ivertex,cell_ids_abv_blw,ncells_level,level,cell_traversal); CHKERRQ(ierr);

  if (ncells_abv == 0 || ncells_blw == 0) {
    if (ncells_abv == 0) {
      level = 1;
      ncells_level = ncells_blw;
    } else {
      level = 0;
      ncells_level = ncells_abv;
    }
    ierr = ComputeBoundaryFaceTraversalDirection(tdy,ivertex,cell_ids_abv_blw,ncells_level,level,cell_traversal); CHKERRQ(ierr);
  }

  PetscInt ii,jj,aa,bb,l2r_dir;
  count=0;

  for (level=0; level<2; level++) {
    if (level==0) l2r_dir = 1;
    else          l2r_dir = 0;
    for (ii = 0; ii<ncells+ncells_bnd; ii++) {
      aa = cell_traversal[level][ii];
      if (aa == -1) break;

      if (ii == ncells+ncells_bnd-1) bb = cell_traversal[level][0];
      else                           bb = cell_traversal[level][ii+1];

      if (bb == -1) bb = cell_traversal[level][0];
      if (aa>=ncells && bb>=ncells) continue;

      PetscInt increase_count = PETSC_FALSE;

      if (aa < ncells && bb < ncells) {
        increase_count = PETSC_TRUE;
      } else if (aa < ncells && bb>= ncells) {
        increase_count = PETSC_TRUE;
      } else if (aa >= ncells && bb<ncells) {
        increase_count = PETSC_TRUE;
      }

      if (increase_count) {
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
  }

  for (ii=0; ii<ncells+ncells_bnd; ii++) {
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
  
  // Rearrange cell_up2dw
  PetscInt tmp[count][2],found[count];
  for (ii=0;ii<count;ii++) found[ii] = 0;
  jj = 0;
  for (ii=0;ii<count;ii++) {
    if (cell_up2dw[ii][0]<ncells && cell_up2dw[ii][1]<ncells) {
      tmp[jj][0] = cell_up2dw[ii][0];
      tmp[jj][1] = cell_up2dw[ii][1];
      found[ii] = 1;
      jj++;
    }
  }
  for (ii=0;ii<count;ii++) {
    if (found[ii]==0) {
      tmp[jj][0] = cell_up2dw[ii][0];
      tmp[jj][1] = cell_up2dw[ii][1];
      jj++;
    }
  }
  for (ii=0;ii<count;ii++) {
    cell_up2dw[ii][0] = tmp[ii][0];
    cell_up2dw[ii][1] = tmp[ii][1];
  }

  ierr = SetupUpwindFacesForSubcell(vertices,ivertex,cells,faces,subcells,cell_up2dw); CHKERRQ(ierr);

  ierr = TDyDeallocate_IntegerArray_2D(cell_traversal, 2); CHKERRQ(ierr);
  ierr = TDyDeallocate_IntegerArray_2D(cell_up2dw, ncells+ncells_bnd); CHKERRQ(ierr);

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

    // save cell centroid
    ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

    num_subcells = cells->num_subcells[icell];

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      // set pointer to vertex and subcell
      PetscInt cOffsetVertex = cells->vertex_offset[icell];
      ivertex = cells->vertex_ids[cOffsetVertex+isubcell];
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
      ierr = DetermineUpwindFacesForSubcell_PlanarVerticalFaces(tdy, ivertex); CHKERRQ(ierr);
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

    PetscReal v1[3], v2[3], v3[3], v4[3], normal[3];
    PetscReal f_cen[3], c_cen[3], f2c[3], dot_prod;
    PetscInt d;

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

    PetscInt cOffsetNeighbor = cells->neighbor_offset[icell];
    PetscInt cOffsetVertex = cells->vertex_offset[icell];
    PetscInt cOffsetEdge = cells->edge_offset[icell];

    for (k=0; k<4; k++) {
      neigh_id_v [icell*4 + k] = cells->neighbor_ids[cOffsetNeighbor+k];
      vertex_id_v[icell*4 + k] = cells->vertex_ids[cOffsetVertex+k];
      edge_id_v  [icell*4 + k] = cells->edge_ids[cOffsetEdge+k];

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
    PetscInt vOffsetIntCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetEdge    = vertices->edge_offset[ivertex];
    for (i=0; i<4; i++) {
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

  ierr = SaveMeshGeometricAttributes(tdy); CHKERRQ(ierr);
  ierr = SaveMeshConnectivityInfo(   tdy); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = UpdateCellOrientationAroundAVertex2DMesh(tdy); CHKERRQ(ierr);
    ierr = SetupSubcellsFor2DMesh     (  tdy->dm, tdy); CHKERRQ(ierr);
    ierr = UpdateCellOrientationAroundAEdge2DMesh(  tdy); CHKERRQ(ierr);
    break;

  case 3:
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
