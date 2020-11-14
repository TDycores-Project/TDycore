#include <petsc.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>

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
        case 6:
          cell_type = CELL_WEDGE_TYPE;
          break;
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
    case CELL_WEDGE_TYPE:
      value = 6;
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
    case CELL_WEDGE_TYPE:
      value = 16;
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
    case CELL_WEDGE_TYPE:
      value = 4;
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
    case CELL_WEDGE_TYPE:
      value = 2;
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
    case CELL_WEDGE_TYPE:
      value = 4;
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
    case CELL_WEDGE_TYPE:
      value = 4;
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
    case CELL_WEDGE_TYPE:
      value = 4;
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
    case CELL_WEDGE_TYPE:
      value = 9;
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
    case CELL_WEDGE_TYPE:
      value = 5;
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
    case CELL_WEDGE_TYPE:
      value = 5;
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
    case CELL_WEDGE_TYPE:
      value = 24;
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
    case CELL_WEDGE_TYPE:
      value = SUBCELL_HEX_TYPE;
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
PetscInt TDyMeshGetNumberOfLocalCells(TDy_mesh *mesh) {

  PetscInt nLocalCells = 0;
  PetscInt icell;

  PetscFunctionBegin;

  for (icell = 0; icell<mesh->num_cells; icell++) {
    if (mesh->cells.is_local[icell]) nLocalCells++;
  }

  PetscFunctionReturn(nLocalCells);
}

/* -------------------------------------------------------------------------- */
PetscInt TDyMeshGetNumberOfLocalFacess(TDy_mesh *mesh) {

  PetscInt nLocalFaces = 0;
  PetscInt iface;

  PetscFunctionBegin;

  for (iface = 0; iface<mesh->num_faces; iface++) {
    if (mesh->faces.is_local[iface]) nLocalFaces++;
  }

  PetscFunctionReturn(nLocalFaces);
}

/* -------------------------------------------------------------------------- */
PetscInt TDyMeshGetNumberOfNonLocalFacess(TDy_mesh *mesh) {

  PetscInt nNonLocalFaces = 0;
  PetscInt iface;

  PetscFunctionBegin;

  for (iface = 0; iface<mesh->num_faces; iface++) {
    if (!mesh->faces.is_local[iface]) nNonLocalFaces++;
  }

  PetscFunctionReturn(nNonLocalFaces);
}

/* -------------------------------------------------------------------------- */
PetscInt TDyMeshGetNumberOfNonInternalFacess(TDy_mesh *mesh) {

  PetscInt nNonInternalFaces = 0;
  PetscInt iface;

  PetscFunctionBegin;

  for (iface = 0; iface<mesh->num_faces; iface++) {
    if (!mesh->faces.is_internal[iface]) nNonInternalFaces++;
  }

  PetscFunctionReturn(nNonInternalFaces);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode AreFacesNeighbors(TDy_face *faces, PetscInt face_id_1, PetscInt face_id_2) {

  PetscBool are_neighbors;
  PetscInt iedge_1, iedge_2;

  are_neighbors = PETSC_FALSE;

  PetscInt fOffetEdge_1 = faces->edge_offset[face_id_1];
  PetscInt fOffetEdge_2 = faces->edge_offset[face_id_2];

  for (iedge_1=0; iedge_1<faces->num_edges[face_id_1]; iedge_1++) {
    for (iedge_2=0; iedge_2<faces->num_edges[face_id_2]; iedge_2++) {
      if (faces->edge_ids[fOffetEdge_1 + iedge_1] == faces->edge_ids[fOffetEdge_2 + iedge_2]) {
        are_neighbors = PETSC_TRUE;
        break;
      }
    }
  }

  PetscFunctionReturn(are_neighbors);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetIthNuVector(TDy_subcell *subcells, PetscInt isubcell, PetscInt i, PetscInt dim, PetscReal *nu_vec) {
  PetscFunctionBegin;
  PetscInt d;
  PetscInt sOffsetNu = subcells->nu_vector_offset[isubcell];

  if (i>=subcells->num_nu_vectors[isubcell]) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th nu_vec exceeds max nu_vecs");
  }
  
  for (d=0; d<dim; d++) nu_vec[d] = subcells->nu_vector[sOffsetNu + i].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetIthNuStarVector(TDy_subcell *subcells, PetscInt isubcell, PetscInt i, PetscInt dim, PetscReal *nu_vec) {
  PetscFunctionBegin;
  PetscInt d;
  PetscInt sOffsetNu = subcells->nu_vector_offset[isubcell];

  if (i>=subcells->num_nu_vectors[isubcell]) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th nu_vec exceeds max nu_vecs");
  }
  
  for (d=0; d<dim; d++) nu_vec[d] = subcells->nu_star_vector[sOffsetNu + i].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetIthFaceCentroid(TDy_subcell *subcells, PetscInt isubcell, PetscInt i, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  PetscInt sOffsetNu = subcells->nu_vector_offset[isubcell];

  if (i>=subcells->num_faces[isubcell]) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Subcell: Requested i-th face centroid exceeds max num_faces");
  }

  for (d=0; d<dim; d++) centroid[d] = subcells->face_centroid[sOffsetNu + i].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDy_subcell* subcells, PetscInt isubcell, PetscInt face_id, PetscInt *face_idx) {

  PetscFunctionBegin;
  PetscInt i;
  *face_idx = -1;

  PetscInt sOffsetFace = subcells->face_offset[isubcell];
  for (i=0; i<subcells->num_faces[isubcell];i++) {
    if (subcells->face_ids[sOffsetFace+i] == face_id) {
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
PetscErrorCode TDyEdge_GetCentroid(TDy_edge *edges, PetscInt iedge, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = edges->centroid[iedge].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyEdge_GetNormal(TDy_edge *edges, PetscInt iedge, PetscInt dim, PetscReal *normal) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) normal[d] = edges->normal[iedge].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFace_GetCentroid(TDy_face *faces, PetscInt iface, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = faces->centroid[iface].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFace_GetNormal(TDy_face *faces, PetscInt iface, PetscInt dim, PetscReal *normal) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) normal[d] = faces->normal[iface].V[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyVertex_GetCoordinate(TDy_vertex *vertices, PetscInt ivertex, PetscInt dim, PetscReal *coor) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) coor[d] = vertices->coordinate[ivertex].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
PetscErrorCode TDyCell_GetCentroid(TDy_cell *cell, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = cell->centroid.X[d];
  PetscFunctionReturn(0);
}
*/

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyCell_GetCentroid2(TDy_cell *cells, PetscInt icell, PetscInt dim, PetscReal *centroid) {
  PetscFunctionBegin;
  PetscInt d;
  for (d=0; d<dim; d++) centroid[d] = cells->centroid[icell].X[d];
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode FindNeighboringVerticesOfAFace(TDy_face *faces, PetscInt iface, PetscInt vertex_id,
                                              PetscInt neighboring_vertex_ids[2]) {

  PetscFunctionBegin;

  PetscInt vertex_ids[faces->num_vertices[iface]];
  PetscInt count = 0, ivertex, start = -1;
  PetscInt fOffsetVertex = faces->vertex_offset[iface];

  // Find the first occurance of "vertex_id" within list of vertices forming the face
  for (ivertex=0; ivertex<faces->num_vertices[iface]; ivertex++) {
    if (faces->vertex_ids[fOffsetVertex + ivertex] == vertex_id) {
      start = ivertex;
      break;
    }
  }

  if (start == -1) {
    printf("Did not find vertex id = %d within face id = %d\n",vertex_id,iface);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Stopping in FindNeighboringVerticesOfAFace");
  }

  // Save vertex indices starting from the first occurance of "vertex_id" to the last vertex
  // from the list
  for (ivertex=start; ivertex<faces->num_vertices[iface]; ivertex++){
    vertex_ids[count] = faces->vertex_ids[fOffsetVertex + ivertex];
    count++;
  }

  // Now, save vertex indices starting from 0 to the first occurance of "vertex_id"
  // from the list
  for (ivertex=0; ivertex<start; ivertex++){
    vertex_ids[count] = faces->vertex_ids[fOffsetVertex + ivertex];
    count++;
  }

  // The vertex_ids[1] and vertex_ids[end] are neighbors of "vertex_id"
  neighboring_vertex_ids[0] = vertex_ids[1];
  neighboring_vertex_ids[1] = vertex_ids[faces->num_vertices[iface]-1];

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode FindFaceIDsOfACellCommonToAVertex(PetscInt cell_id, TDy_face *faces,
                                                 TDy_vertex *vertices, PetscInt ivertex,
                                                 PetscInt f_idx[3],
                                                 PetscInt *num_shared_faces) {
  
  PetscFunctionBegin;
  
  PetscInt iface;
  //PetscInt cell_id = cell->id;
  
  *num_shared_faces = 0;
  
  // Find the faces of "cell_id" that are shared by the vertex
  for (iface=0; iface<vertices->num_faces[ivertex]; iface++){
    
    PetscInt vOffsetFace = vertices->face_offset[ivertex];
    PetscInt face_id = vertices->face_ids[vOffsetFace + iface];
    PetscInt fOffsetCell = faces->cell_offset[face_id];
    
    if (faces->cell_ids[fOffsetCell + 0] == cell_id || faces->cell_ids[fOffsetCell + 1] == cell_id){
      
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
    printf("Was expecting to find 3 faces of the cell to be shared by the vertex, but instead found %d common faces\n",*num_shared_faces);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported vertex type for 3D mesh");
  }
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalCells(TDy tdy) {

  PetscErrorCode ierr;
  DM             dm;
  Vec            junkVec;
  PetscInt       junkInt;
  PetscInt       gref;
  PetscInt       cStart, cEnd, c;
  TDy_cell       *cells;

  PetscFunctionBegin;

  dm = tdy->dm;
  cells = &tdy->mesh->cells;

  PetscMPIInt rank;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);

  // Once needs to atleast haved called a DMCreateXYZ() before using DMPlexGetPointGlobal()
  ierr = DMCreateGlobalVector(dm, &junkVec); CHKERRQ(ierr);
  ierr = VecDestroy(&junkVec); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      cells->is_local[c] = PETSC_TRUE;
      cells->global_id[c] = gref;
    } else {
      cells->is_local[c] = PETSC_FALSE;
      cells->global_id[c] = -gref-1;
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalVertices(TDy tdy) {

  PetscInt       ivertex, icell, c;
  TDy_mesh       *mesh = tdy->mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  PetscInt       vStart, vEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    for (c=0; c<vertices->num_internal_cells[ivertex]; c++) {
      PetscInt vOffsetIntCell = vertices->internal_cell_offset[ivertex];
      icell = vertices->internal_cell_ids[vOffsetIntCell + c];
      if (icell >= 0 && cells->is_local[icell]) vertices->is_local[ivertex] = PETSC_TRUE;
    }

  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalEdges(TDy tdy) {

  PetscInt iedge, icell_1, icell_2;
  TDy_mesh *mesh = tdy->mesh;
  TDy_cell *cells;
  TDy_edge *edges;
  PetscInt       eStart, eEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  cells = &mesh->cells;
  edges = &mesh->edges;

  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd); CHKERRQ(ierr);

  for (iedge=0; iedge<mesh->num_edges; iedge++) {

    PetscInt eOffsetCell = edges->cell_offset[iedge];

    if (edges->cell_ids[eOffsetCell] == -1 || edges->cell_ids[eOffsetCell + 1] == -1) edges->is_internal[iedge] = PETSC_FALSE;
    if (!edges->is_internal[iedge]) { // Is it a boundary edge?

      // Determine the cell ID for the boundary edge
      if (edges->cell_ids[eOffsetCell + 0] != -1) icell_1 = edges->cell_ids[eOffsetCell + 0];
      else                                icell_1 = edges->cell_ids[eOffsetCell + 1];

      // Is the cell locally owned?
      if (icell_1 >= 0 && cells->is_local[icell_1]) edges->is_local[iedge] = PETSC_TRUE;

    } else { // An internal edge

      // Save the two cell ID
      icell_1 = edges->cell_ids[eOffsetCell + 0];
      icell_2 = edges->cell_ids[eOffsetCell + 1];

      if (cells->is_local[icell_1] && cells->is_local[icell_2]) { // Are both cells locally owned?

        edges->is_local[iedge] = PETSC_TRUE;

      } else if (cells->is_local[icell_1] && !cells->is_local[icell_2]) { // Is icell_1 locally owned?

        // Is the global ID of icell_1 lower than global ID of icell_2?
        if (cells->global_id[icell_1] < cells->global_id[icell_2]) edges->is_local[iedge] = PETSC_TRUE;

      } else if (!cells->is_local[icell_1] && cells->is_local[icell_2]) { // Is icell_2 locally owned

        // Is the global ID of icell_2 lower than global ID of icell_1?
        if (cells->global_id[icell_2] < cells->global_id[icell_1]) edges->is_local[iedge] = PETSC_TRUE;

      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalFaces(TDy tdy) {

  PetscInt iface, icell_1, icell_2;
  TDy_mesh *mesh = tdy->mesh;
  TDy_cell *cells;
  TDy_face *faces;
  PetscInt       fStart, fEnd;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  cells = &mesh->cells;
  faces = &mesh->faces;

  mesh->num_boundary_faces = 0;

  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    PetscInt fOffsetCell = faces->cell_offset[iface];

    if (!faces->is_internal[iface]) { // Is it a boundary face?

      mesh->num_boundary_faces++;

      // Determine the cell ID for the boundary edge
      if (faces->cell_ids[fOffsetCell + 0] >= 0) {
        icell_1 = faces->cell_ids[fOffsetCell + 0];
        faces->cell_ids[fOffsetCell + 1] = -mesh->num_boundary_faces;
      } else {
        icell_1 = faces->cell_ids[fOffsetCell + 1];
        faces->cell_ids[fOffsetCell + 0] = -mesh->num_boundary_faces;
      }

      // Is the cell locally owned?
      if (icell_1 >= 0 && cells->is_local[icell_1]) faces->is_local[iface] = PETSC_TRUE;

    } else { // An internal face

      // Save the two cell ID
      icell_1 = faces->cell_ids[fOffsetCell + 0];
      icell_2 = faces->cell_ids[fOffsetCell + 1];

      // Is either cell locally owned?
      if (cells->is_local[icell_1] || cells->is_local[icell_2]) faces->is_local[iface] = PETSC_TRUE;

    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDy_cell *cells, PetscInt cell_id, TDy_vertex *vertices, PetscInt ivertex, TDy_subcell *subcells, PetscInt *subcell_id) {

  PetscFunctionBegin;

  PetscInt i, isubcell = -1;

  for (i=0; i<vertices->num_internal_cells[ivertex];i++){
    PetscInt vOffsetCell = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    if (vertices->internal_cell_ids[vOffsetCell + i] == cells->id[cell_id]) {
      isubcell = vertices->subcell_ids[vOffsetSubcell + i];
      break;
    }
  }

  if (isubcell == -1) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Did not find a subcell of a given cell that includes the given vertex");
  }
  
  *subcell_id = cells->id[cell_id]*cells->num_subcells[cell_id]+isubcell;
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyGetNumCellsLocal(TDy tdy, PetscInt *num_cells) {

  PetscFunctionBegin;

  *num_cells = tdy->mesh->num_cells;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyGetCellNaturalIDsLocal(TDy tdy, PetscInt *ni, PetscInt nat_ids[]) {

  TDy_mesh *mesh = tdy->mesh;
  TDy_cell *cells;
  PetscInt icell;

  PetscFunctionBegin;
  *ni = 0;

  cells = &mesh->cells;

  for (icell=0; icell<mesh->num_cells; icell++) {
    nat_ids[*ni] = cells->natural_id[icell];
    *ni += 1;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyGetCellIsLocal(TDy tdy, PetscInt *ni, PetscInt is_local[]) {

  TDy_mesh *mesh = tdy->mesh;
  TDy_cell *cells;
  PetscInt icell;

  PetscFunctionBegin;
  *ni = 0;

  cells = &mesh->cells;

  for (icell=0; icell<mesh->num_cells; icell++) {
    if (cells->is_local[icell]) is_local[*ni] = 1;
    else                        is_local[*ni] = 0;
    *ni += 1;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyPrintSubcellInfo(TDy tdy, PetscInt icell, PetscInt isubcell) {

  PetscFunctionBegin;

  TDy_mesh *mesh = tdy->mesh;
  TDy_cell *cells = &mesh->cells;
  TDy_subcell *subcells = &mesh->subcells;

  PetscInt subcell_id = icell*cells->num_subcells[icell] + isubcell;
  PetscInt sOffsetFace = subcells->face_offset[subcell_id];

  printf("Subcell_id = %02d is %d-th subcell of cell_id = %d; ",subcell_id, isubcell, icell);
  printf(" No. faces = %d; ",subcells->num_faces[subcell_id]);

  PetscInt iface;
  printf(" Face Ids: ");
  for (iface = 0; iface<subcells->num_faces[subcell_id]; iface++) {
    printf("  %02d ",subcells->face_ids[sOffsetFace + iface]);
  }
  printf("\n");

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyPrintFaceInfo(TDy tdy, PetscInt iface) {

  PetscFunctionBegin;

  TDy_mesh *mesh = tdy->mesh;
  TDy_face *faces = &mesh->faces;

  printf("Face_id = %d; ",iface);

  PetscInt dim, d;
  PetscErrorCode ierr;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  printf(" Centroid: ");
  for (d = 0; d<dim; d++) {
    printf(" %+e ",faces->centroid[iface].X[d]);
  }
  printf("\n");

  PetscFunctionReturn(0);
}

