#include <petsc.h>
#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>
#include <private/tdyregionimpl.h>
#include <private/tdydiscretizationimpl.h>
#include <private/tdymeshimpl.h>

/* -------------------------------------------------------------------------- */
static PetscErrorCode TDyMeshMapIndices(TDyDiscretizationType *discretization, TDyMesh** mesh) {

  PetscErrorCode ierr;

  TDyMesh *mesh_ptr = *mesh;

  TDyUGrid *ugrid;
  ierr = TDyDiscretizationGetTDyUGrid(discretization,&ugrid);

  TDyUGDM *ugdm;
  ierr = TDyDiscretizationGetTDyUGDM(discretization,&ugdm);

  PetscInt ngmax = ugrid->num_cells_global;
  PetscInt nlmax = ugrid->num_cells_local;

  ierr = TDyAllocate_IntegerArray_1D(&mesh_ptr->nG2L,ngmax); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&mesh_ptr->nL2G,nlmax); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&mesh_ptr->nG2A,ngmax); CHKERRQ(ierr);

  const PetscInt *int_ptr;
  ierr = ISGetIndices(ugdm->IS_GhostedCells_in_PetscOrder, &int_ptr); CHKERRQ(ierr);

  for (PetscInt icell=0; icell<nlmax; icell++) {
    mesh_ptr->nG2L[icell] = icell;
    mesh_ptr->nL2G[icell] = icell;
    mesh_ptr->nG2A[icell] = int_ptr[icell];
  }

  for (PetscInt icell=nlmax; icell<ngmax; icell++) {
    mesh_ptr->nG2L[icell] = -1;
    mesh_ptr->nG2A[icell] = int_ptr[icell];
  }

  ierr = ISRestoreIndices(ugdm->IS_GhostedCells_in_PetscOrder, &int_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode ComputeVolume(TDyCoordinate *coords, PetscInt ncoords, PetscReal *volume) {

  PetscErrorCode ierr;
  PetscReal tmp_vol;

  PetscInt hex_tet_num = 5;
  PetscInt hex_tet_ids[5][4] = {
    {0, 1, 2, 5},
    {0, 5, 7, 4},
    {0, 5, 2, 7},
    {7, 5, 2, 6},
    {0, 2, 3, 7}
  };

  PetscInt wedge_tet_num = 3;
  PetscInt wedge_tet_ids[3][4] = {
    {0, 1, 2, 3},
    {2, 3, 4, 5},
    {0, 1, 3, 4}
  };

  PetscInt prism_tet_num = 2;
  PetscInt prism_tet_ids[3][4] = {
    {0, 1, 2, 4},
    {2, 3, 0, 4}
  };

  PetscInt a,b,c,d;

  *volume = 0.0;
  switch (ncoords){
  case 8:
    for (PetscInt i=0; i<hex_tet_num; i++) {
      a=hex_tet_ids[i][0];
      b=hex_tet_ids[i][1];
      c=hex_tet_ids[i][2];
      d=hex_tet_ids[i][3];

      ierr = VolumeofTetrahedron(coords[a].X, coords[b].X, coords[c].X, coords[d].X, &tmp_vol); CHKERRQ(ierr);

      *volume += tmp_vol;
    }
    break;

  case 6:
    for (PetscInt i=0; i<wedge_tet_num; i++) {
      a=wedge_tet_ids[i][0];
      b=wedge_tet_ids[i][1];
      c=wedge_tet_ids[i][2];
      d=wedge_tet_ids[i][3];

      ierr = VolumeofTetrahedron(coords[a].X, coords[b].X, coords[c].X, coords[d].X, &tmp_vol); CHKERRQ(ierr);

      *volume += tmp_vol;
    }
    break;

  case 5:
    for (PetscInt i=0; i<prism_tet_num; i++) {
      a=prism_tet_ids[i][0];
      b=prism_tet_ids[i][1];
      c=prism_tet_ids[i][2];
      d=prism_tet_ids[i][3];

      ierr = VolumeofTetrahedron(coords[a].X, coords[b].X, coords[c].X, coords[d].X, &tmp_vol); CHKERRQ(ierr);

      *volume += tmp_vol;
    }
    break;

  case 4:
    a=0;
    b=1;
    c=2;
    d=3;

    ierr = VolumeofTetrahedron(coords[a].X, coords[b].X, coords[c].X, coords[d].X, &tmp_vol); CHKERRQ(ierr);

    *volume += tmp_vol;
    break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"ComputeVolume only supports hex, wedge, prism, and tetrahedron cell types.");
    break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode TDySetupCellsFromDiscretization(TDyDiscretizationType *discretization, TDyMesh **mesh) {

  PetscErrorCode ierr;

  // compute number of vertices per grid cell
  TDyUGrid *ugrid;
  ierr = TDyDiscretizationGetTDyUGrid(discretization, &ugrid);

  PetscInt nverts_per_cell = ugrid->max_verts_per_cell;
  PetscInt num_cells = ugrid->num_cells_global;
  PetscInt ngmax = ugrid->num_cells_global;

  TDyCellType cell_type = GetCellType(nverts_per_cell);

  TDyMesh *mesh_ptr = *mesh;
  ierr = AllocateCells(num_cells, cell_type, &mesh_ptr->cells); CHKERRQ(ierr);

  TDyCell *cells = &mesh_ptr->cells;
  PetscReal **vertices = ugrid->vertices;
  PetscInt **cell_vertices = ugrid->cell_vertices;

  TDyCoordinate vertex_coords[nverts_per_cell];

  PetscInt dim=3;
  PetscReal xyz[dim];

  for (PetscInt icell=0; icell<ngmax; icell++) {

    PetscInt nvmax = ugrid->cell_num_vertices[icell];

    // initialize x,y,z cell centroid to zero
    for (PetscInt idim=0; idim<3; idim++) {
      xyz[idim] = 0.0;
    }

    // - compute centroid as mean of vertex coordinates
    // - save vertex coordinates for computation of cell volume
    for (PetscInt ivertex=0; ivertex<nvmax; ivertex++) {
      PetscInt vertex_id = cell_vertices[icell][ivertex];

      for (PetscInt idim=0; idim<dim; idim++) {
        vertex_coords[ivertex].X[idim] = vertices[vertex_id][idim];
        xyz[idim] += vertices[vertex_id][idim];
      }
    }

    for (PetscInt idim=0; idim<dim; idim++) {
      cells->centroid[icell].X[idim] = xyz[idim]/nvmax;
    }

    // compute cell volume
    ierr = ComputeVolume(vertex_coords, nvmax, &cells->volume[icell]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode GetHexFaceVertices(PetscInt iface, PetscInt *vertex_ids){

PetscFunctionBegin;

switch (iface)
{
case 0:
  vertex_ids[0] = 0;
  vertex_ids[1] = 1;
  vertex_ids[2] = 5;
  vertex_ids[3] = 4;
  break;

case 1:
  vertex_ids[0] = 1;
  vertex_ids[1] = 2;
  vertex_ids[2] = 6;
  vertex_ids[3] = 5;
  break;

case 2:
  vertex_ids[0] = 2;
  vertex_ids[1] = 3;
  vertex_ids[2] = 7;
  vertex_ids[3] = 6;
  break;

case 3:
  vertex_ids[0] = 3;
  vertex_ids[1] = 0;
  vertex_ids[2] = 4;
  vertex_ids[3] = 7;
  break;

case 4:
  vertex_ids[0] = 0;
  vertex_ids[1] = 3;
  vertex_ids[2] = 2;
  vertex_ids[3] = 1;
  break;

case 5:
  vertex_ids[0] = 4;
  vertex_ids[1] = 5;
  vertex_ids[2] = 6;
  vertex_ids[3] = 7;
  break;

default:
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Hexahedron can only have 6 faces");
  break;
}

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode GetWedgeFaceVertices(PetscInt iface, PetscInt *vertex_ids){

PetscFunctionBegin;

switch (iface)
{
case 0:
  vertex_ids[0] = 0;
  vertex_ids[1] = 1;
  vertex_ids[2] = 4;
  vertex_ids[3] = 3;
  break;

case 1:
  vertex_ids[0] = 1;
  vertex_ids[1] = 2;
  vertex_ids[2] = 5;
  vertex_ids[3] = 4;
  break;

case 2:
  vertex_ids[0] = 2;
  vertex_ids[1] = 0;
  vertex_ids[2] = 3;
  vertex_ids[3] = 5;
  break;

case 3:
  vertex_ids[0] = 0;
  vertex_ids[1] = 2;
  vertex_ids[2] = 1;
  vertex_ids[3] = -1;
  break;

case 4:
  vertex_ids[0] = 3;
  vertex_ids[1] = 4;
  vertex_ids[2] = 5;
  vertex_ids[3] = -1;
  break;

default:
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Wedge can only have 5 faces");
  break;
}

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode GetPyramidFaceVertices(PetscInt iface, PetscInt *vertex_ids){

PetscFunctionBegin;

switch (iface)
{
case 0:
  vertex_ids[0] = 0;
  vertex_ids[1] = 1;
  vertex_ids[2] = 4;
  vertex_ids[3] = -1;
  break;

case 1:
  vertex_ids[0] = 1;
  vertex_ids[1] = 2;
  vertex_ids[2] = 4;
  vertex_ids[3] = -1;
  break;

case 2:
  vertex_ids[0] = 2;
  vertex_ids[1] = 3;
  vertex_ids[2] = 4;
  vertex_ids[3] = -1;
  break;

case 3:
  vertex_ids[0] = 3;
  vertex_ids[1] = 0;
  vertex_ids[2] = 4;
  vertex_ids[3] = -1;
  break;

case 4:
  vertex_ids[0] = 0;
  vertex_ids[1] = 3;
  vertex_ids[2] = 2;
  vertex_ids[3] = 1;
  break;

default:
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Pyramid can only have 5 faces");
  break;
}

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode GetTetFaceVertices(PetscInt iface, PetscInt *vertex_ids){

PetscFunctionBegin;

switch (iface)
{
case 0:
  vertex_ids[0] = 0;
  vertex_ids[1] = 1;
  vertex_ids[2] = 3;
  vertex_ids[3] = -1;
  break;

case 1:
  vertex_ids[0] = 1;
  vertex_ids[1] = 2;
  vertex_ids[2] = 3;
  vertex_ids[3] = -1;
  break;

case 2:
  vertex_ids[0] = 0;
  vertex_ids[1] = 3;
  vertex_ids[2] = 2;
  vertex_ids[3] = -1;
  break;

case 3:
  vertex_ids[0] = 0;
  vertex_ids[1] = 2;
  vertex_ids[2] = 1;
  vertex_ids[3] = -1;
  break;

default:
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Tetrahedron can only have 4 faces");
  break;
}

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode GetFaceVertices(TDyCellType cell_type, PetscInt iface, PetscInt *vertex_ids){

  PetscErrorCode ierr;

  switch (cell_type)
  {
  case CELL_HEX_TYPE:
  ierr = GetHexFaceVertices(iface, vertex_ids); CHKERRQ(ierr);
  break;

  case CELL_WEDGE_TYPE:
  ierr = GetWedgeFaceVertices(iface, vertex_ids); CHKERRQ(ierr);
  break;

  case CELL_PYRAMID_TYPE:
  ierr = GetPyramidFaceVertices(iface, vertex_ids); CHKERRQ(ierr);
  break;

  case CELL_TET_TYPE:
  ierr = GetTetFaceVertices(iface, vertex_ids); CHKERRQ(ierr);
  break;

  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknown cell type. Supported cell types include hexahedron, wedge, prism, and tetrahedron");
    break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscBool AllVerticesPresentInDual(TDyUGrid *ugrid, PetscInt cell_id, PetscInt cell_id2, PetscInt iface, PetscInt **cell_to_face, PetscInt **face_to_vertex) {

  PetscInt num_vertices = ugrid->cell_num_vertices[cell_id];
  TDyCellType cell_type = GetCellType(num_vertices);

  PetscInt num_face_vertices = GetNumOfVerticesOfIthFacesForCellType(cell_type, iface);
  PetscInt face_id = cell_to_face[iface][cell_id];

  // Check if all vertices of the face_id (of cell_id) are present in cell_id2
  PetscBool vertex_found = PETSC_FALSE;
  for (PetscInt ivertex=0; ivertex<num_face_vertices; ivertex++) {
    PetscInt vertex_id = face_to_vertex[ivertex][face_id];

    PetscInt num_vertices2 = ugrid->cell_num_vertices[cell_id2];
    vertex_found = PETSC_FALSE;
    for (PetscInt ivertex2=0; ivertex2<num_vertices2; ivertex2++) {
      PetscInt vertex_id2 = ugrid->cell_vertices[cell_id2][ivertex2];

      // check if the ivertex of cell_id is found in cell_id2
      if (vertex_id == vertex_id2) {
        vertex_found = PETSC_TRUE;
        continue;
      }
    }

    if (!vertex_found) {
      break;
    }
  }

  PetscFunctionReturn(vertex_found);
}

/* -------------------------------------------------------------------------- */
static PetscInt GetCorrespondingFaceInDualCell(TDyUGrid *ugrid, PetscInt cell_id, PetscInt cell_id2, PetscInt iface, PetscInt **cell_to_face, PetscInt **face_to_vertex) {


  PetscInt corresponding_face_id = -1;

  PetscInt num_cell_vertices = ugrid->cell_num_vertices[cell_id];
  TDyCellType cell_type = GetCellType(num_cell_vertices);
  PetscInt nvertices = GetNumOfVerticesOfIthFacesForCellType(cell_type, iface);
  PetscInt face_id = cell_to_face[iface][cell_id];

  PetscInt num_cell_vertices2 = ugrid->cell_num_vertices[cell_id2];
  TDyCellType cell_type2 = GetCellType(num_cell_vertices2);
  PetscInt nfaces2 = GetNumFacesForCellType(cell_type2);

  for (PetscInt iface2=0; iface2<nfaces2; iface2++) {
    PetscInt face_id2 = cell_to_face[iface2][cell_id2];
    PetscInt nvertices2 = GetNumOfVerticesOfIthFacesForCellType(cell_type2, iface2);

    // Check if the number of vertices forming the face are same
    if (nvertices == nvertices2) {

      // Now check if all the vertices of 'iface' are present in 'face2'
      PetscInt num_match = 0;

      for (PetscInt ivertex=0; ivertex<nvertices; ivertex++) {
          PetscInt vertex_id = face_to_vertex[ivertex][face_id];
          PetscBool vertex_found = PETSC_FALSE;

        for (PetscInt ivertex2=0; ivertex2<nvertices2; ivertex2++) {
          PetscInt vertex_id2 = face_to_vertex[ivertex2][face_id2];
          if (vertex_id == vertex_id2) {
            vertex_found = PETSC_TRUE;
            num_match++;
            break;
          }
        }
        if (!vertex_found) {
          break;
        }
      }

      if (num_match == nvertices) {
        corresponding_face_id = iface2;
        break;
      }
    }
  }

  PetscFunctionReturn(corresponding_face_id);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode SetupMaps_C2F_F2C_F2V(TDyUGrid *ugrid, PetscInt **cell_to_face, PetscInt **face_to_cell, PetscInt **face_to_vertex) {

  PetscErrorCode ierr;
  PetscInt ngmax = ugrid->num_cells_global;

  PetscInt *vertex_ids;
  ierr = TDyAllocate_IntegerArray_1D(&vertex_ids, 4); CHKERRQ(ierr);

  PetscInt face_count=0;
  for (PetscInt icell=0; icell<ngmax; icell++) {
    PetscInt num_vertices = ugrid->cell_num_vertices[icell];
    TDyCellType cell_type = GetCellType(num_vertices);
    PetscInt nfaces = GetNumFacesForCellType(cell_type);

    for (PetscInt iface=0; iface<nfaces; iface++) {
      cell_to_face[iface][icell] = face_count;
      face_to_cell[0][face_count] = icell;

      PetscInt nvertices = GetNumOfVerticesOfIthFacesForCellType(cell_type, iface);
      ierr = GetFaceVertices(cell_type, iface, vertex_ids); CHKERRQ(ierr);

      for (PetscInt ivertex=0; ivertex<nvertices; ivertex++) {

        PetscInt vertex_id_relative = vertex_ids[ivertex];
        PetscInt vertex_id = ugrid->cell_vertices[icell][vertex_id_relative];

        face_to_vertex[ivertex][face_count] = vertex_id;

        if (vertex_id > -1) {
          ugrid->face_to_vertex_natural[ivertex][face_count] = ugrid->vertex_ids_natural[vertex_id];
        }
      }

      face_count++;
    }
  }
  free(vertex_ids);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode RemoveDuplicateFaces(TDyUGrid *ugrid, PetscInt **face_to_vertex, PetscInt **face_to_cell, PetscInt **cell_to_face) {

  PetscInt nlmax = ugrid->num_cells_local;

  for (PetscInt icell=0; icell<nlmax; icell++) {

    // 1. Pick cell_id
    PetscInt cell_id = icell;
    PetscInt num_vertices = ugrid->cell_num_vertices[icell];
    TDyCellType cell_type = GetCellType(num_vertices);
    PetscInt nfaces = GetNumFacesForCellType(cell_type);
    PetscBool common_face_found = PETSC_FALSE;

    for (PetscInt idual=0; idual<ugrid->cell_num_neighbors_ghosted[icell]; idual++) {
      common_face_found = PETSC_FALSE;
      // 2. Pick a neighbor of cell_id
      PetscInt cell_id2 = PetscAbs(ugrid->cell_neighbors_ghosted[cell_id][idual]);
      if (cell_id2 < 0) cell_id2 = -cell_id2;

      if (cell_id2 <= cell_id) {
        // 3. skip this neigbhor because the duplicate face was removed earlier
        common_face_found = PETSC_TRUE;
        continue;
      }

      common_face_found = PETSC_FALSE;
      for (PetscInt iface=0; iface<nfaces; iface++) {
        PetscInt face_id = cell_to_face[iface][cell_id];

        // 4. Check if 'cell_id' and 'cell_id2' share a face
        common_face_found = PETSC_FALSE;
        if (AllVerticesPresentInDual(ugrid, cell_id, cell_id2, iface, cell_to_face, face_to_vertex)) { // face_to_vertex, ugrid%cell_vertices
  
          // 5. For face_id of cell_id, find the corresponding face_id2 of cell_id2
          PetscInt iface2 = GetCorrespondingFaceInDualCell(ugrid, cell_id, cell_id2, iface, cell_to_face, face_to_vertex); // cell_to_face, face_to_vertex
          if (iface2 > -1) {
            common_face_found = PETSC_TRUE;
            PetscInt face_id2 = cell_to_face[iface2][cell_id2];

            if (face_id2 > face_id) {
              cell_to_face[iface2][cell_id2] = face_id;
              face_to_cell[0][face_id2] = -face_to_cell[0][face_id2];
              face_to_cell[1][face_id ] = cell_id2;
            } else {
              cell_to_face[iface][cell_id] = face_id2;
              face_to_cell[0][face_id ] = -face_to_cell[0][face_id ];
              face_to_cell[1][face_id2] = cell_id;
            }
          }
        }

        if (common_face_found) break;
      }
      if (!common_face_found) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"2. Did not find a common face between two neighbors");
      }
    } // idual-loop
  }
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode UpdateMapsC2F_F2C_F2V(TDyUGrid *ugrid, PetscInt **cell_to_face, PetscInt **face_to_cell, PetscInt **face_to_vertex) {

  PetscErrorCode ierr;
  PetscInt max_vert_per_face = ugrid->max_vert_per_face;
  PetscInt max_face_per_cell = ugrid->max_face_per_cell;
  PetscInt ngmax = ugrid->num_cells_global;

  // Determine number of unique faces
  PetscInt face_count = 0;
  for (PetscInt iface=0; iface<max_face_per_cell*ngmax; iface++){
    if (face_to_cell[0][iface] >= 0) {
      face_count++;
    }
  }

  // Save face-to-vertex mapping
  ierr = TDyAllocate_IntegerArray_2D(&ugrid->face_to_vertex,max_vert_per_face,face_count); CHKERRQ(ierr);
  face_count = 0;
  for (PetscInt iface=0; iface<max_face_per_cell*ngmax; iface++){
    if (face_to_cell[0][iface] >= 0) {
      for (PetscInt ivertex=0; ivertex<max_vert_per_face; ivertex++) {
        ugrid->face_to_vertex[ivertex][face_count] = face_to_vertex[ivertex][face_count];
      }
      face_count++;
    }
  }
  ierr = TDyDeallocate_IntegerArray_2D(face_to_vertex, max_vert_per_face);

  // Since duplicate faces have been removed, update the face-to-cell mapping
  PetscInt **tmp_int_2d;
  PetscInt *tmp_int;
  ierr = TDyAllocate_IntegerArray_1D(&tmp_int, max_face_per_cell*ngmax); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&tmp_int_2d, 2, face_count); CHKERRQ(ierr);

  face_count = 0;
  for (PetscInt iface=0; iface<max_face_per_cell*ngmax; iface++) {
    if (face_to_cell[0][iface] > -1) {

      for (PetscInt i=0; i<2; i++) {
        tmp_int_2d[i][face_count] = face_to_cell[i][iface];
      }
      tmp_int[iface] = face_count;

      face_count++;
    }
  }
  ierr = TDyDeallocate_IntegerArray_2D(face_to_cell, 2); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&face_to_cell, 2, face_count); CHKERRQ(ierr);
  for (PetscInt iface=0; iface<face_count; iface++) {
    for (PetscInt i=0; i<2; i++) {
      face_to_cell[i][iface] = tmp_int_2d[i][iface];
    }
  }
  ierr = TDyDeallocate_IntegerArray_2D(tmp_int_2d, 2);

  // Since duplicate faces have been removed, update the cell-to-face mapping
  for (PetscInt iface=0; iface<face_count; iface++) {
    PetscInt face_id = iface;
    for (PetscInt i=0; i<2; i++) {

      PetscInt cell_id = face_to_cell[i][face_id];

      if (cell_id < 0) continue;

      PetscBool found = PETSC_FALSE;
      PetscInt num_vertices = ugrid->cell_num_vertices[cell_id];
      TDyCellType cell_type = GetCellType(num_vertices);
      PetscInt nfaces = GetNumFacesForCellType(cell_type);

      for (PetscInt iface2=0; iface2<nfaces; iface2++) {
        PetscInt face_id2 = cell_to_face[iface2][cell_id];

        if (face_id2 < 0) continue;

        if (face_id == tmp_int[face_id2]) {
          found = PETSC_TRUE;
          cell_to_face[iface2][cell_id] = face_id;
          break;
        }
      }
      if (!found) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Remapping of face ids failed");
      }
    }
  }
  free(tmp_int);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode SetupMap_V2C(TDyUGrid *ugrid, PetscInt *num_vertex_to_cell, PetscInt **vertex_to_cell) {

  PetscInt num_vertices_local = ugrid->num_verts_local;
  PetscInt max_cells_sharing_a_vertex = ugrid->max_cells_sharing_a_vertex;
  PetscInt ngmax = ugrid->num_cells_global;

  for (PetscInt ivertex=0; ivertex<num_vertices_local; ivertex++){
    num_vertex_to_cell[ivertex] = 0;
  }

  for (PetscInt icell=0; icell<ngmax; icell++) {

    PetscInt nvertex = ugrid->cell_num_vertices[icell];

    for (PetscInt ivertex=0; ivertex<nvertex; ivertex++) {
      PetscInt vertex_id = ugrid->cell_vertices[icell][ivertex];

      if (vertex_id < 0) continue;

      PetscInt idx=num_vertex_to_cell[ivertex];
      vertex_to_cell[idx][vertex_id] = icell;

      num_vertex_to_cell[ivertex]++;
      if (num_vertex_to_cell[ivertex] > max_cells_sharing_a_vertex) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Vertex is shared by more than max_cells_sharing_a_vertex");
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode CreateInternalFaces(PetscInt **face_to_cell, PetscInt **cell_to_face, TDyUGrid *ugrid, TDyMesh **mesh) {

  PetscErrorCode ierr;

  PetscInt nlmax = ugrid->num_cells_local;
  PetscInt nconn = 0;

  // Compute number of internal faces
  for (PetscInt icell=0; icell<nlmax; icell++) {
    PetscInt ndual = ugrid->cell_num_neighbors_ghosted[icell];
    for (PetscInt idual=0; idual<ndual; idual++) {
      PetscInt dual_id = ugrid->cell_neighbors_ghosted[icell][idual];
      if (dual_id < 0 || icell < dual_id){
        nconn++;
      }
    }
  }

  TDyMesh *mesh_ptr = *mesh;
  TDyCellType cell_type = CELL_HEX_TYPE;
  ierr = AllocateFaces (nconn, cell_type, &mesh_ptr->faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&ugrid->connection_to_face, nconn); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&ugrid->face_area, nconn); CHKERRQ(ierr);


  PetscInt iconn=0, offset = 0;
  for (PetscInt icell=0; icell<nlmax; icell++) {

    PetscInt ndual = ugrid->cell_num_neighbors_ghosted[icell];

    for (PetscInt idual=0; idual<ndual; idual++) {

      PetscInt dual_id = ugrid->cell_neighbors_ghosted[icell][idual];
      PetscInt face_id = -1;

      if (icell < PetscAbs(dual_id)){
        PetscBool found = PETSC_FALSE;
        PetscInt iface = -1, iface_tmp = -1;
        PetscInt iface2 = -1, iface2_tmp = -1;
        PetscInt cell_id2 = -1;

        PetscInt num_cell_vertices = ugrid->cell_num_vertices[icell];
        TDyCellType icell_type = GetCellType(num_cell_vertices);
        PetscInt nfaces = GetNumFacesForCellType(icell_type);

        // Find the iface that is shared by icell and cell_id2
        for (iface=0; iface<nfaces; iface++) {
          face_id = cell_to_face[iface][icell];
          for (PetscInt iside=0; iside<2; iside++) {
            cell_id2 = face_to_cell[iside][face_id];
            if (cell_id2 == abs(dual_id)) {
              iface_tmp = iface;
              found = PETSC_TRUE;
              break;
            }
          }
          if (found) break;
        }

        if (found) {
          ugrid->connection_to_face[iconn] = face_id;
        } else {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Face not found");
        }
        iface = iface_tmp;

        // Check that there exist a correspond face in the cell_id2 for
        // the iface-th of icell-th
        found = PETSC_FALSE;
        for (iface2=0; iface2<ugrid->cell_num_vertices[cell_id2]; iface2++) {
          if (cell_to_face[iface][icell] == cell_to_face[iface2][cell_id2]) {
            iface2_tmp = iface2;
            found = PETSC_TRUE;
            break;
          }
        }
        iface2 = iface2_tmp;

        // Now check the corresponding faces are of the same type
        if (found) {
          PetscInt num_cell_vertices = ugrid->cell_num_vertices[icell];
          TDyCellType icell_type = GetCellType(num_cell_vertices);
          TDyFaceType face_type = GetFaceTypeForCellType(icell_type, iface);

          PetscInt num_cell_vertices2 = ugrid->cell_num_vertices[cell_id2];
          TDyCellType icell_type2 = GetCellType(num_cell_vertices2);
          TDyFaceType face_type2 = GetFaceTypeForCellType(icell_type2,iface2);

          if (face_type != face_type2) {
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Face types do not match");
          }

        } else {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Global face not found");
        }

        mesh_ptr->faces.cell_offset[iface] = offset;
        mesh_ptr->faces.cell_ids[offset++] = icell;
        mesh_ptr->faces.cell_ids[offset++] = abs(dual_id);
        mesh_ptr->faces.id[iface] = cell_to_face[iface][icell];

        iconn++;
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode TDySetupFacesFromDiscretization(TDyDiscretizationType *discretization, TDyMesh **mesh) {

  PetscErrorCode ierr;

  // face_to_vertex
  // face_to_cell
  // cell_to_face
  // vertex_to_cell
  TDyUGrid *ugrid;
  ierr = TDyDiscretizationGetTDyUGrid(discretization, &ugrid);

  PetscInt num_vertices_local = ugrid->num_verts_local;
  PetscInt ngmax = ugrid->num_cells_global;

  PetscInt max_cells_sharing_a_vertex = ugrid->max_cells_sharing_a_vertex;
  PetscInt max_vert_per_face = ugrid->max_vert_per_face;
  PetscInt max_face_per_cell = ugrid->max_face_per_cell;

  PetscInt **face_to_vertex;
  PetscInt **face_to_cell;
  PetscInt **cell_to_face;
  ierr = TDyAllocate_IntegerArray_2D(&face_to_vertex, max_vert_per_face, max_face_per_cell*ngmax); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&face_to_cell, 2, max_face_per_cell*ngmax); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&cell_to_face, max_face_per_cell, ngmax); CHKERRQ(ierr);

  ierr = TDyAllocate_IntegerArray_2D(&ugrid->face_to_vertex_natural,max_vert_per_face, max_face_per_cell*ngmax); CHKERRQ(ierr);

  // Set up few mappings
  ierr = SetupMaps_C2F_F2C_F2V(ugrid,cell_to_face,face_to_cell,face_to_vertex);

  // Remove duplicate faces that are shared between two cells
  ierr = RemoveDuplicateFaces(ugrid, face_to_vertex, face_to_cell, cell_to_face); CHKERRQ(ierr);

  // Update the maps after removing duplicate faces
  ierr = UpdateMapsC2F_F2C_F2V(ugrid, cell_to_face, face_to_cell, face_to_vertex);

  // Set up vertex-to-cell mapping
  PetscInt **vertex_to_cell;
  PetscInt *num_vertex_to_cell;
  ierr = TDyAllocate_IntegerArray_2D(&vertex_to_cell, max_cells_sharing_a_vertex, num_vertices_local); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&num_vertex_to_cell, num_vertices_local); CHKERRQ(ierr);
  ierr = SetupMap_V2C(ugrid, num_vertex_to_cell, vertex_to_cell); CHKERRQ(ierr);

  // Create internal faces
  ierr = CreateInternalFaces(face_to_cell, cell_to_face, ugrid, mesh);


  PetscFunctionReturn(0);
}

/// Constructs a mesh from TDycore-managed (i) TDyDM, and (ii) TDyUGrid
/// @param [in] discretization A TDyDiscretizationType from which the mesh is created
/// @param [out] mesh the newly constructed mesh instance
PetscErrorCode TDyMeshCreateFromDiscretization(TDyDiscretizationType *discretization, TDyMesh** mesh) {

  PetscErrorCode ierr;

  *mesh = malloc(sizeof(TDyMesh));

  ierr = TDyMeshMapIndices(discretization, mesh); CHKERRQ(ierr);
  ierr = TDySetupCellsFromDiscretization(discretization, mesh); CHKERRQ(ierr);
  ierr = TDySetupFacesFromDiscretization(discretization, mesh); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
