#include <petsc.h>
#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>

/* -------------------------------------------------------------------------- */
static PetscErrorCode AllocateCells(
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
static PetscErrorCode AllocateSubcells(
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
static PetscErrorCode AllocateVertices(
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
static PetscErrorCode AllocateEdges(
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
static PetscErrorCode AllocateFaces(
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

/// Constructs a mesh from a PETSc DM.
/// @param [in] dm A PETSc DM from which the mesh is created
/// @param [out] mesh the newly constructed mesh instance
PetscErrorCode TDyMeshCreate(DM dm, TDyMesh **mesh) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  TDY_START_FUNCTION_TIMER()

  PetscInt dim;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (dim != 3) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"MPFA-O only supports 3D meshes");
  }

  *mesh = malloc(sizeof(TDyMesh));

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
  PetscInt f_start, f_end;
  ierr = DMPlexGetDepthStratum( dm, 2, &f_start, &f_end); CHKERRQ(ierr);
  num_faces = f_end - f_start;

  TDyMesh *m = *mesh;
  m->num_cells    = num_cells;
  m->num_faces    = num_faces;
  m->num_edges    = num_edges;
  m->num_vertices = num_vertices;

  m->maxClosureSize = 27*4*4;
  ierr = TDyAllocate_IntegerArray_1D(&m->closureSize, num_cells+num_faces+num_edges+num_vertices); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&m->closure, num_cells+num_faces+num_edges+num_vertices, 2*m->maxClosureSize); CHKERRQ(ierr);
  ierr = TDySaveClosures(dm, m->closureSize, m->closure, &m->maxClosureSize); CHKERRQ(ierr);

  // compute number of vertices per grid cell
  PetscInt nverts_per_cell = TDyGetNumberOfCellVerticesWithClosures(dm, m->closureSize, m->closure);
  TDyCellType cell_type = GetCellType(nverts_per_cell);

  ierr = AllocateCells(num_cells, cell_type, &m->cells); CHKERRQ(ierr);
  ierr = AllocateEdges(num_edges, cell_type, &m->edges); CHKERRQ(ierr);
  ierr = AllocateFaces(num_faces, cell_type, &m->faces); CHKERRQ(ierr);

  PetscInt ncells_per_vertex = TDyMaxNumberOfCellsSharingAVertex(dm, m->closureSize, m->closure);
  PetscInt nfaces_per_vertex = TDyMaxNumberOfFacesSharingAVertex(dm, m->closureSize, m->closure);
  PetscInt nedges_per_vertex = TDyMaxNumberOfEdgesSharingAVertex(dm, m->closureSize, m->closure);
  ierr = AllocateVertices(num_vertices, ncells_per_vertex, nfaces_per_vertex,
                          nedges_per_vertex, cell_type, &m->vertices); CHKERRQ(ierr);

  TDySubcellType subcell_type = GetSubcellTypeForCellType(cell_type);
  PetscInt num_subcells  = GetNumSubcellsForSubcellType(subcell_type);
  m->num_subcells = num_cells*num_subcells;
  ierr = AllocateSubcells(num_cells, num_subcells, subcell_type,
                          &m->subcells); CHKERRQ(ierr);

  ierr = TDyRegionCreate(&m->region_connected); CHKERRQ(ierr);

  ierr = IdentifyLocalCells(dm, mesh); CHKERRQ(ierr);
  ierr = IdentifyLocalVertices(mesh); CHKERRQ(ierr);
  ierr = IdentifyLocalEdges(mesh); CHKERRQ(ierr);
  ierr = IdentifyLocalFaces(mesh); CHKERRQ(ierr);

  ierr = SaveNaturalIDs(mesh, dm); CHKERRQ(ierr);

  ierr = ConvertCellsToCompressedFormat(mesh); CHKERRQ(ierr);
  ierr = ConvertVerticesToCompressedFormat(mesh); CHKERRQ(ierr);
  ierr = ConvertSubcellsToCompressedFormat(mesh); CHKERRQ(ierr);
  ierr = ConvertFacesToCompressedFormat(mesh); CHKERRQ(ierr);
  ierr = UpdateFaceOrderAroundAVertex(mesh); CHKERRQ(ierr);
  ierr = UpdateCellOrientationAroundAFace(mesh); CHKERRQ(ierr);
  ierr = SetupSubcells(mesh); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/// Destroy a mesh, freeing any resources it uses.
/// @param [inout] mesh A mesh instance to be destroyed
PetscErrorCode TDyMeshDestroy(TDyMesh *mesh) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/// Returns the (maximum) numbers of cells and faces per vertex in the mesh.
/// @param [in] mesh A mesh instance
/// @param [out] num_cells The maximum number of cells connected to any vertex.
/// @param [out] num_faces The maximum number of faces connected to any vertex.
PetscErrorCode TDyMeshGetMaxVertexConnectivity(TDyMesh *mesh,
                                               PetscInt *num_cells,
                                               PetscInt *num_faces) {
  PetscFunctionBegin;
  *num_cells = TDyMaxNumberOfCellsSharingAVertex(dm, m->closureSize, m->closure);
  *num_faces = TDyMaxNumberOfFacesSharingAVertex(dm, m->closureSize, m->closure);
  PetscFunctionReturn(0);
}

/// Given a mesh and a cell index, retrieve an array of cell edge indices, and
/// their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] edges Stores a pointer to an array of edges for the
///                    given cell
/// @param [out] num_edges Stores the number of edges for the given cell
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
PetscErrorCode TDyMeshGetCellVertices(TDyMesh *mesh,
                                      PetscInt cell,
                                      PetscInt **vertices,
                                      PetscInt *num_vertices) {
  PetscInt offset = mesh->cells.vertex_offset[cell];
  *vertices = &mesh->cells.vertex_ids[offset];
  *num_vertices = mesh->cells.vertex_offset[cell+1] - offset;
  return 0;
}

/// Given a mesh and a cell index, returns number of cell vertices
/// their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] num_vertices Stores the number of vertices for the given cell
PetscErrorCode TDyMeshGetCellNumVertices(TDyMesh *mesh,
                                         PetscInt cell,
                                         PetscInt *num_vertices) {
  PetscInt offset = mesh->cells.vertex_offset[cell];
  *num_vertices = mesh->cells.vertex_offset[cell+1] - offset;
  return 0;
}

/// Given a mesh and a cell index, retrieve an array of indices of faces bounding the cell,
/// and their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] faces Stores a pointer to an array of faces bounding the given
///                    cell
/// @param [out] num_faces Stores the number of faces bounding the given cell
PetscErrorCode TDyMeshGetCellFaces(TDyMesh *mesh,
                                   PetscInt cell,
                                   PetscInt **faces,
                                   PetscInt *num_faces) {
  PetscInt offset = mesh->cells.face_offset[cell];
  *faces = &mesh->cells.face_ids[offset];
  *num_faces = mesh->cells.face_offset[cell+1] - offset;
  return 0;
}

/// Given a mesh and a cell index, returns number of cell faces
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] num_faces Stores the number of faces bounding the given cell
PetscErrorCode TDyMeshGetCellNumFaces(TDyMesh *mesh,
                                      PetscInt cell,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->cells.face_offset[cell];
  *num_faces = mesh->cells.face_offset[cell+1] - offset;
  return 0;
}

/// Given a mesh and a cell index, retrieve an array of indices of neighboring
/// cells, and their number.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] neighbors Stores a pointer to an array of cells neighboring the
///                        given cell
/// @param [out] num_neighbors Stores the number of cells neighboring the given
///                            cell
PetscErrorCode TDyMeshGetCellNeighbors(TDyMesh *mesh,
                                       PetscInt cell,
                                       PetscInt **neighbors,
                                       PetscInt *num_neighbors) {
  PetscInt offset = mesh->cells.neighbor_offset[cell];
  *neighbors = &mesh->cells.neighbor_ids[offset];
  *num_neighbors = mesh->cells.neighbor_offset[cell+1] - offset;
  return 0;
}

/// Given a mesh and a cell index, return number of cell neighbors
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] num_neighbors Stores the number of cells neighboring the given
///                            cell
PetscErrorCode TDyMeshGetCellNumNeighbors(TDyMesh *mesh,
                                          PetscInt cell,
                                          PetscInt *num_neighbors) {
  PetscInt offset = mesh->cells.neighbor_offset[cell];
  *num_neighbors = mesh->cells.neighbor_offset[cell+1] - offset;
  return 0;
}

/// Retrieve the centroid for the given cell in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] centroid Stores the centroid of the given cell
PetscErrorCode TDyMeshGetCellCentroid(TDyMesh *mesh,
                                      PetscInt cell,
                                      TDyCoordinate *centroid) {
  *centroid = mesh->cells.centroid[cell];
  return 0;
}

/// Retrieve the volume for the given cell in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] cell The index of a cell within the mesh
/// @param [out] centroid Stores the volume of the given cell
PetscErrorCode TDyMeshGetCellVolume(TDyMesh *mesh,
                                    PetscInt cell,
                                    PetscReal *volume) {
  *volume = mesh->cells.volume[cell];
  return 0;
}

/// Given a mesh and a vertex index, retrieve an array of associated internal
/// cell indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] int_cells Stores a pointer to an array of internal cells
///                        attached to the given vertex
/// @param [out] num_int_cells Stores the number of internal cells attached to
///                            the given vertex
PetscErrorCode TDyMeshGetVertexInternalCells(TDyMesh *mesh,
                                             PetscInt vertex,
                                             PetscInt **int_cells,
                                             PetscInt *num_int_cells) {
  PetscInt offset = mesh->vertices.internal_cell_offset[vertex];
  *int_cells = &mesh->vertices.internal_cell_ids[offset];
  *num_int_cells = mesh->vertices.internal_cell_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, return number of associated internal
/// cells
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_int_cells Stores the number of internal cells attached to
///                            the given vertex
PetscErrorCode TDyMeshGetVertexNumInternalCells(TDyMesh *mesh,
                                                PetscInt vertex,
                                                PetscInt *num_int_cells) {
  PetscInt offset = mesh->vertices.internal_cell_offset[vertex];
  *num_int_cells = mesh->vertices.internal_cell_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, return number of associated subcells
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_subcells Stores the number of subcells attached to the
///                           given vertex
PetscErrorCode TDyMeshGetVertexNumSubcells(TDyMesh *mesh,
                                           PetscInt vertex,
                                           PetscInt *num_subcells) {
  PetscInt offset = mesh->vertices.subcell_offset[vertex];
  *num_subcells = mesh->vertices.subcell_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, retrieve an array of associated face
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] faces Stores a pointer to an array of faces attached to
///                    the given vertex
/// @param [out] num_faces Stores the number of faces attached to the given
///                        vertex
PetscErrorCode TDyMeshGetVertexFaces(TDyMesh *mesh,
                                     PetscInt vertex,
                                     PetscInt **faces,
                                     PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.face_offset[vertex];
  *faces = &mesh->vertices.face_ids[offset];
  *num_faces = mesh->vertices.face_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, return number of faces
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_faces Stores the number of faces attached to the given
///                        vertex
PetscErrorCode TDyMeshGetVertexNumFaces(TDyMesh *mesh,
                                        PetscInt vertex,
                                        PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.face_offset[vertex];
  *num_faces = mesh->vertices.face_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, retrieve an array of associated subface
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] subfaces Stores a pointer to an array of faces attached to
///                    the given vertex
/// @param [out] num_subfaces Stores the number of faces attached to the given
///                        vertex
PetscErrorCode TDyMeshGetVertexSubfaces(TDyMesh *mesh,
                                     PetscInt vertex,
                                     PetscInt **subfaces,
                                     PetscInt *num_subfaces) {
  PetscInt offset = mesh->vertices.face_offset[vertex];
  *subfaces = &mesh->vertices.subface_ids[offset];
  *num_subfaces = mesh->vertices.face_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, retrieve an array of associated boundary
////face indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] faces Stores a pointer to an array of boundary faces attached
///                    to the given vertex
/// @param [out] num_faces Stores the number of boundary faces attached to the
///                        given vertex
PetscErrorCode TDyMeshGetVertexBoundaryFaces(TDyMesh *mesh,
                                             PetscInt vertex,
                                             PetscInt **faces,
                                             PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.boundary_face_offset[vertex];
  *faces = &mesh->vertices.boundary_face_ids[offset];
  *num_faces = mesh->vertices.boundary_face_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, return number of associated boundary
////faces
/// @param [in] mesh A mesh instance
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] num_faces Stores the number of boundary faces attached to the
///                        given vertex
PetscErrorCode TDyMeshGetVertexNumBoundaryFaces(TDyMesh *mesh,
                                                PetscInt vertex,
                                                PetscInt *num_faces) {
  PetscInt offset = mesh->vertices.boundary_face_offset[vertex];
  *num_faces = mesh->vertices.boundary_face_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a face index, retrieve an array of associated cell indices
/// and their number.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] cells Stores a pointer to an array of cells attached to the
///                    given face
/// @param [out] num_cells Stores the number of cells attached to the given
///                        face
PetscErrorCode TDyMeshGetFaceCells(TDyMesh *mesh,
                                   PetscInt face,
                                   PetscInt **cells,
                                   PetscInt *num_cells) {
  PetscInt offset = mesh->faces.cell_offset[face];
  *cells = &mesh->faces.cell_ids[offset];
  *num_cells = mesh->faces.cell_offset[face+1] - offset;
  return 0;
}

/// Given a mesh and a face index, return number of associate cells
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] num_cells Stores the number of cells attached to the given
///                        face
PetscErrorCode TDyMeshGetFaceNumCells(TDyMesh *mesh,
                                      PetscInt face,
                                      PetscInt *num_cells) {
  PetscInt offset = mesh->faces.cell_offset[face];
  *num_cells = mesh->faces.cell_offset[face+1] - offset;
  return 0;
}

/// Given a mesh and a face index, retrieve an array of associated vertex
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] vertices Stores a pointer to an array of vertices attached to
///                       the given face
/// @param [out] num_vertices Stores the number of subcells attached to the given
///                           face
PetscErrorCode TDyMeshGetFaceVertices(TDyMesh *mesh,
                                      PetscInt face,
                                      PetscInt **vertices,
                                      PetscInt *num_vertices) {
  PetscInt offset = mesh->faces.vertex_offset[face];
  *vertices = &mesh->faces.vertex_ids[offset];
  *num_vertices = mesh->faces.vertex_offset[face+1] - offset;
  return 0;
}

/// Given a mesh and a face index, return number of associated vertex
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a vertex within the mesh
/// @param [out] num_vertices Stores the number of subcells attached to the given
///                           face
PetscErrorCode TDyMeshGetFaceNumVertices(TDyMesh *mesh,
                                         PetscInt face,
                                         PetscInt *num_vertices) {
  PetscInt offset = mesh->faces.vertex_offset[face];
  *num_vertices = mesh->faces.vertex_offset[face+1] - offset;
  return 0;
}

/// Retrieve the centroid for the given face in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a face within the mesh
/// @param [out] centroid Stores the centroid of the given face
PetscErrorCode TDyMeshGetFaceCentroid(TDyMesh *mesh,
                                      PetscInt face,
                                      TDyCoordinate *centroid) {
  *centroid = mesh->faces.centroid[face];
  return 0;
}

/// Retrieve the normal vector for the given face in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a face within the mesh
/// @param [out] normal Stores the normal vector for the given face
PetscErrorCode TDyMeshGetFaceNormal(TDyMesh *mesh,
                                    PetscInt face,
                                    TDyVector *normal) {
  *normal = mesh->faces.normal[face];
  return 0;
}

/// Retrieve the area of the given face in the given mesh.
/// @param [in] mesh A mesh instance
/// @param [in] face The index of a face within the mesh
/// @param [out] area Stores the area of the given face
PetscErrorCode TDyMeshGetFaceArea(TDyMesh *mesh,
                                  PetscInt face,
                                  PetscReal *area) {
  *area = mesh->faces.area[face];
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated face
/// indices and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] faces Stores a pointer to an array of faces attached to
///                    the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellFaces(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **faces,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *faces = &mesh->subcells.face_ids[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, return number of associated faces
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellNumFaces(TDyMesh *mesh,
                                         PetscInt subcell,
                                         PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated is_face_up
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of is_face_up array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellIsFaceUp(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **is_face_up,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *is_face_up = &mesh->subcells.is_face_up[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated face_unknown_idx
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face_unkown_idx array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellFaceUnknownIdxs(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **face_unknown_idx,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_unknown_idx = &mesh->subcells.face_unknown_idx[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated face_flux_idx
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face_flux_idx array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellFaceFluxIdxs(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **face_flux_idx,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_flux_idx = &mesh->subcells.face_flux_idx[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated face areas
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face areas array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellFaceAreas(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscReal **face_area,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_area = &mesh->subcells.face_area[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated vertices
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of vertices array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellVertices(TDyMesh *mesh,
                                      PetscInt subcell,
                                      PetscInt **vertices,
                                      PetscInt *num_faces) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *vertices = &mesh->subcells.vertex_ids[offset];
  *num_faces = mesh->subcells.face_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated nu_vectors
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of nu_vectors array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellNuVectors(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyVector **nu_vectors,
                                      PetscInt *num_nu_vectors) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *nu_vectors = &mesh->subcells.nu_vector[offset];
  *num_nu_vectors = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated nu_star_vectors
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of nu_star_vectors array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellNuStarVectors(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyVector **nu_star_vectors,
                                      PetscInt *num_nu_star_vectors) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *nu_star_vectors = &mesh->subcells.nu_star_vector[offset];
  *num_nu_star_vectors = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated variable_continuity_coordinates
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of variable_continuity_coordinates array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellVariableContinutiyCoordinates(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyCoordinate **variable_continuity_coordinates,
                                      PetscInt *num_variable_continuity_coordinates) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *variable_continuity_coordinates = &mesh->subcells.variable_continuity_coordinates[offset];
  *num_variable_continuity_coordinates = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated face_centroid
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of face_centroid array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellFaceCentroids(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyCoordinate **face_centroids,
                                      PetscInt *num_face_centroids) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *face_centroids = &mesh->subcells.variable_continuity_coordinates[offset];
  *num_face_centroids = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  return 0;
}

/// Given a mesh and a subcell index, retrieve an array of associated vertices_coordinates
/// array and their number.
/// @param [in] mesh A mesh instance
/// @param [in] subcell The index of a subcell within the mesh
/// @param [out] is_face_up Stores a pointer to an array of vertices_coordinates array attached to
///                         the given subcell
/// @param [out] num_faces Stores the number of faces attached to the
///                        given subcell
PetscErrorCode TDyMeshGetSubcellVerticesCoordinates(TDyMesh *mesh,
                                      PetscInt subcell,
                                      TDyCoordinate **vertices_coordinates,
                                      PetscInt *num_vertices_coordinates) {
  PetscInt offset = mesh->subcells.face_offset[subcell];
  *vertices_coordinates = &mesh->subcells.vertices_coordinates[offset];
  *num_vertices_coordinates = mesh->subcells.nu_vector_offset[subcell+1] - offset;
  return 0;
}

/// Reads a geometric attribute file. Currently only PETSc binary file format
/// is supported
///
/// @param [inout] tdy A TDy struct
/// @returns 0  on success or a non-zero error code on failure
PetscErrorCode TDyMeshReadGeometry(TDyMesh *mesh, const char* filename) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;

  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // 1. Read the binary geometric attribute file
  //
  // The i-th entry of the output strided Vec is
  //
  // volume_i, centroid_<x|y|z>_i, numNeighbor, nat_id_ij, face_area_of_ij, face_centroid_<x|y|z>_ij
  //
  
  PetscInt numLocalCells = TDyMeshGetNumberOfLocalCells(mesh);
  PetscInt numNeighbor = cells->num_neighbors[0]; // It is assumed that all cells have same number of neighbors
  PetscInt numCellAttr = 5; // volume_i, centroid_<x|y|z>_i, numNeighbor
  PetscInt numNeigbhorAttr = 5; // nat_id_ij, face_centroid_<x|y|z>_ij face_area_of_ij
  PetscInt stride = numCellAttr + numNeighbor * numNeigbhorAttr;

  Vec natural_vec;
  ierr = VecCreate(PETSC_COMM_WORLD,&natural_vec); CHKERRQ(ierr);
  ierr = VecSetSizes(natural_vec,numLocalCells*stride,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(natural_vec,stride); CHKERRQ(ierr);
  ierr = VecSetFromOptions(natural_vec); CHKERRQ(ierr);

  Vec local_vec;
  ierr = VecCreate(PETSC_COMM_WORLD,&local_vec); CHKERRQ(ierr);
  ierr = VecSetSizes(local_vec,mesh->num_cells*stride,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(local_vec,stride); CHKERRQ(ierr);
  ierr = VecSetFromOptions(local_vec); CHKERRQ(ierr);

  // Read the data
  ierr = TDyReadBinaryPetscVec(natural_vec, filename); CHKERRQ(ierr);

  //
  // 2. Perform natural-to-local scatter of the geometric attributes
  //

  // Create index for natural vector
  PetscInt local_offset = 0;
  ierr = MPI_Exscan(&mesh->num_cells,&local_offset,1,MPI_INTEGER,MPI_SUM,PETSC_COMM_WORLD);

  PetscInt int_array_from[mesh->num_cells];
  PetscInt int_array_to[mesh->num_cells];
  for (PetscInt i=0; i<mesh->num_cells; i++){
    int_array_to[i] = i + local_offset;
    int_array_from[i] = cells->natural_id[i];
  }

  IS is_from;
  ierr = ISCreateBlock(PETSC_COMM_WORLD,stride,mesh->num_cells,int_array_from,PETSC_COPY_VALUES,&is_from); CHKERRQ(ierr);

  // Create index for local vector
  IS is_to;
  ierr = ISCreateBlock(PETSC_COMM_WORLD,stride,mesh->num_cells,int_array_to,PETSC_COPY_VALUES,&is_to); CHKERRQ(ierr);

  // Create natural-to-local scatter
  VecScatter n2l_scatter;
  VecScatterCreate(natural_vec,is_from,local_vec,is_to,&n2l_scatter); CHKERRQ(ierr);

  ierr = VecScatterBegin(n2l_scatter,natural_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(n2l_scatter,natural_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  TDySavePetscVecAsBinary(local_vec,"cells_local.bin");

  //
  // 3. Save geometric attributes locally
  //
  PetscScalar *values;
  ierr = VecGetArray(local_vec,&values); CHKERRQ(ierr);

  for (PetscInt icell=0; icell<mesh->num_cells; icell++){

    // The i-th entry of the output strided Vec is
    //
    //  col 1        col 2-4            col-5       col 6         col 7            col 8-10
    // volume_i, centroid_<x|y|z>_i, numNeighbor, nat_id_ij, face_area_of_ij, face_centroid_<x|y|z>_ij, ...

    PetscInt col = 0;
    PetscInt offset = icell*stride;

    // col 1: cell volume
    cells->volume[icell] = values[ offset + col];      
    col++;

    // col 2-4: cell centroid
    for (PetscInt d=0; d<3; d++) {
      cells->centroid[icell].X[d] = values[offset + col];
      col++;
    }

    PetscInt *face_ids, num_faces;
    TDyMeshGetCellFaces(mesh, icell, &face_ids, &num_faces);

    // col 5: number of neighbors
    if ( num_faces != values[offset + col]) {
      printf("ReadMeshGeometricAttributes: Number of faces do not match = num_faces%d values[%d] = %f; rank = %d\n",num_faces, offset + col, values[offset + col],rank);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ReadMeshGeometricAttributes: Number of faces do not match");
    }
    col++;

    // Initialize if information about face/neigbhor is set
    PetscInt bnd_face_set[num_faces];
    for (PetscInt iface=0; iface<num_faces; iface++) {
      bnd_face_set[iface] = 0;
    }

    // col 6-10; 11-15; ...
    for (PetscInt iface=0; iface<num_faces; iface++) {

      PetscBool face_found = PETSC_FALSE;
      PetscInt face_id;

      // col 6
      PetscInt neighbor_id_in_data = values[offset + col];
      col++;

      if (neighbor_id_in_data < 0) {

        // This face/neighbor is on boundary.
        // Find the first boundary face for which geometric attributes
        // hasn't already been set

        for (PetscInt i=0; i<num_faces; i++) {
          face_id = face_ids[i];

          PetscInt *cell_ids, tmp;
          ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &tmp); CHKERRQ(ierr);
          if (bnd_face_set[i] == 0 && (cell_ids[0]<0 || cell_ids[1]<0)) {
            bnd_face_set[i] = 1;
            face_found = PETSC_TRUE;
            break;
          }
        }
      } else {

        // This face/neigbhor is internal. 
        // Loop over all the faces of the cell to find the face index of
        // the current cell whose neighbor's natural id is neighbor_id_in_data

        for (PetscInt i=0; i<num_faces; i++) {
          face_id = face_ids[i];
          PetscInt *cell_ids, tmp;
          ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &tmp); CHKERRQ(ierr);

          PetscInt nat_id_up = cells->natural_id[cell_ids[0]];
          PetscInt nat_id_dn = cells->natural_id[cell_ids[1]];

          if ( nat_id_up == neighbor_id_in_data || nat_id_dn == neighbor_id_in_data){
            face_found = PETSC_TRUE;
            break;
          }
        }
      }

      if (!face_found && cells->is_local[icell]) {
        // Only stop if the cell is internal and a face is not found
        printf("Not found icell == %d; iface == %d; neighbor_id_in_data = %d; rank = %d\n",icell,iface,neighbor_id_in_data,rank);
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ReadMeshGeometricAttributes: Face not found");
      } else {

        // col 7: Save face area
        faces->area[face_id] = values[offset + col];
        col++;

        // col 8-10: Save face centroid
        for (PetscInt d=0; d<3; d++) {
          faces->centroid[face_id].X[d] = values[offset + col];
          col++;
        }
      }
    }    
  }
  ierr = VecRestoreArray(local_vec,&values); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Outputs geometric attribues of mesh to a file. Currently only PETSc binary
/// file format is supported.
///
/// @param [inout] tdy A TDy struct
/// @returns 0  on success or a non-zero error code on failure
PetscErrorCode TDyMeshWriteGeometry(TDy tdy, const char* filename) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;

  // The i-th entry of the output strided Vec is
  //
    //  col 1        col 2-4            col-5       col 6         col 7            col 8-10
  // volume_i, centroid_<x|y|z>_i, numNeighbor, nat_id_ij, face_area_of_ij, face_centroid_<x|y|z>_ij
  //

  PetscInt numLocalCells = TDyMeshGetNumberOfLocalCells(mesh);

  PetscInt numNeighbor = cells->num_neighbors[0]; // It is assumed that all cells have same number of neighbors
  PetscInt numCellAttr = 5; // volume_i, centroid_<x|y|z>_i, numNeighbor
  PetscInt numNeigbhorAttr = 5; // nat_id_ij, face_centroid_<x|y|z>_ij face_area_of_ij
  PetscInt stride = numCellAttr + numNeighbor * numNeigbhorAttr;

  // Creates vectors for storing geometric attributes in global and natural
  // index
  Vec global_vec, natural_vec;
  ierr = VecCreate(PETSC_COMM_WORLD,&global_vec); CHKERRQ(ierr);
  ierr = VecSetSizes(global_vec,numLocalCells*stride,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(global_vec,stride); CHKERRQ(ierr);
  ierr = VecSetFromOptions(global_vec); CHKERRQ(ierr);

  ierr = VecDuplicate(global_vec, &natural_vec); CHKERRQ(ierr);

  PetscScalar *values;
  ierr = VecGetArray(global_vec,&values); CHKERRQ(ierr);

  // Fill in the geometric attributes for only local cells
  PetscInt count=0;
  for (PetscInt icell=0; icell<mesh->num_cells; icell++){
    if (mesh->cells.is_local[icell]) {
      PetscInt offset = 0;

      // cell volume
      values[count*stride + offset] = cells->volume[icell];      
      offset++;

      // cell centroid
      for (PetscInt d=0; d<3; d++) {
        values[count*stride + offset] = cells->centroid[icell].X[d];
        offset++;
      }

      PetscInt *face_ids, num_faces;
      TDyMeshGetCellFaces(mesh, icell, &face_ids, &num_faces);

      // number of neighbors
      values[count*stride + offset] = num_faces;
      offset++;

      for (PetscInt n=0; n<num_faces; n++){
        PetscInt face_id = face_ids[n];

        PetscInt *face_cell_ids, tmp;
        ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &tmp); CHKERRQ(ierr);

        PetscInt neighbor_id;
        if (face_cell_ids[0] == icell) {
          neighbor_id = face_cell_ids[1];
        } else {
          neighbor_id = face_cell_ids[0];
        }

        PetscInt neighbor_nat_id = neighbor_id;
        
        if (neighbor_id >= 0) {
          neighbor_nat_id = cells->natural_id[neighbor_id];
        }

        // natural ID of neigbhor
        values[count*stride + offset] = neighbor_nat_id;
        offset++;

        // face area
        values[count*stride + offset] = faces->area[face_id];
        offset++;

        // face centroid
        for (PetscInt d=0; d<3; d++) {
          values[count*stride + offset] = faces->centroid[face_id].X[d];
          offset++;
        }

      } // loop over neighbors

    count++;
    } // loop over local cells
  }
  ierr = VecRestoreArray(global_vec,&values); CHKERRQ(ierr);

  // Create (from/to) Index Sets for vector scatter
  PetscInt global_offset = 0;
  ierr = MPI_Exscan(&numLocalCells,&global_offset,1,MPI_INTEGER,MPI_SUM,PETSC_COMM_WORLD);

  PetscInt int_array[numLocalCells];
  for (PetscInt i=0; i<numLocalCells; i++){
    int_array[i] = i + global_offset;
  }

  IS is_from;
  ierr = ISCreateBlock(PETSC_COMM_WORLD,stride,numLocalCells,int_array,PETSC_COPY_VALUES,&is_from); CHKERRQ(ierr);

  IS is_to;
  count = 0;
  for (PetscInt icell=0; icell<mesh->num_cells; icell++){
    if (mesh->cells.is_local[icell]) {
      int_array[count++] = cells->natural_id[icell];
    }
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD,stride,numLocalCells,int_array,PETSC_COPY_VALUES,&is_to); CHKERRQ(ierr);

  // Create vector scatter
  VecScatter g2n_scatter;
  VecScatterCreate(global_vec,is_from,natural_vec,is_to,&g2n_scatter); CHKERRQ(ierr);

  // Scatter data
  ierr = VecScatterBegin(g2n_scatter,global_vec,natural_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(g2n_scatter,global_vec,natural_vec,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);

  // Output data
  ierr = TDySavePetscVecAsBinary(natural_vec, filename); CHKERRQ(ierr);

  // Cleanup
  ierr = ISDestroy(&is_from); CHKERRQ(ierr);
  ierr = ISDestroy(&is_to); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g2n_scatter); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

