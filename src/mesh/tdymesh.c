#include <petsc.h>
#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>

/// Construct a mesh from a PETSc DM.
/// @param [in] dm A PETSc DM object from which a mesh is created
/// @param [out] mesh Stores the newly constructed mesh
PetscErrorCode TDyMeshCreateFromDM(DM dm, TDyMesh *mesh) {
  return 0;
}

/// Destroy a mesh, freeing any resources it uses.
/// @param [inout] mesh A mesh object to be destroyed
PetscErrorCode TDyMeshDestroy(TDyMesh *mesh) {
  return 0;
}

/// Given a mesh and a cell index, retrieve an array of cell edge indices, and
/// their number.
/// @param [in] mesh A mesh object
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
  return 0;
}

/// Given a mesh and a cell index, retrieve an array of cell vertex indices, and
/// their number.
/// @param [in] mesh A mesh object
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

/// Given a mesh and a cell index, retrieve an array of indices of faces bounding the cell,
/// and their number.
/// @param [in] mesh A mesh object
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

/// Given a mesh and a cell index, retrieve an array of indices of neighboring
/// cells, and their number.
/// @param [in] mesh A mesh object
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

/// Retrieve the centroid for the given cell in the given mesh.
/// @param [in] mesh A mesh object
/// @param [in] cell The index of a cell within the mesh
/// @param [out] centroid Stores the centroid of the given cell
PetscErrorCode TDyMeshGetCellCentroid(TDyMesh *mesh,
                                      PetscInt cell,
                                      TDyCoordinate *centroid) {
  *centroid = mesh->cells.centroid[cell];
  return 0;
}

/// Retrieve the volume for the given cell in the given mesh.
/// @param [in] mesh A mesh object
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
/// @param [in] mesh A mesh object
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

/// Given a mesh and a vertex index, retrieve an array of associated subcell
/// indices and their number.
/// @param [in] mesh A mesh object
/// @param [in] vertex The index of a vertex within the mesh
/// @param [out] subcells Stores a pointer to an array of subcells attached to
///                       the given vertex
/// @param [out] num_subcells Stores the number of subcells attached to the
///                           given vertex
PetscErrorCode TDyMeshGetVertexSubcells(TDyMesh *mesh,
                                        PetscInt vertex,
                                        PetscInt **subcells,
                                        PetscInt *num_subcells) {
  PetscInt offset = mesh->vertices.subcell_offset[vertex];
  *subcells = &mesh->vertices.subcell_ids[offset];
  *num_subcells = mesh->vertices.subcell_offset[vertex+1] - offset;
  return 0;
}

/// Given a mesh and a vertex index, retrieve an array of associated face
/// indices and their number.
/// @param [in] mesh A mesh object
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

/// Given a mesh and a vertex index, retrieve an array of associated subface
/// indices and their number.
/// @param [in] mesh A mesh object
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
/// @param [in] mesh A mesh object
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

/// Given a mesh and a face index, retrieve an array of associated cell indices
/// and their number.
/// @param [in] mesh A mesh object
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

/// Given a mesh and a face index, retrieve an array of associated vertex
/// indices and their number.
/// @param [in] mesh A mesh object
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

/// Retrieve the centroid for the given face in the given mesh.
/// @param [in] mesh A mesh object
/// @param [in] face The index of a face within the mesh
/// @param [out] centroid Stores the centroid of the given face
PetscErrorCode TDyMeshGetFaceCentroid(TDyMesh *mesh,
                                      PetscInt face,
                                      TDyCoordinate *centroid) {
  *centroid = mesh->faces.centroid[face];
  return 0;
}

/// Retrieve the normal vector for the given face in the given mesh.
/// @param [in] mesh A mesh object
/// @param [in] face The index of a face within the mesh
/// @param [out] normal Stores the normal vector for the given face
PetscErrorCode TDyMeshGetFaceNormal(TDyMesh *mesh,
                                    PetscInt face,
                                    TDyVector *normal) {
  *normal = mesh->faces.normal[face];
  return 0;
}

/// Retrieve the area of the given face in the given mesh.
/// @param [in] mesh A mesh object
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
/// @param [in] mesh A mesh object
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

