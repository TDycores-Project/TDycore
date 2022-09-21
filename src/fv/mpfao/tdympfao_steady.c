#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdycharacteristiccurvesimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices(TDy tdy, Mat K,Vec F) {
  TDyMPFAO      *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  PetscInt       ivertex, cell_id_up, cell_id_dn;
  PetscInt       irow, icol, row, col, vertex_id;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;
    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];

    PetscInt *face_ids, num_faces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);


    if (vertices->num_boundary_faces[ivertex] == 0) {
      PetscInt nflux_in = vertices->num_faces[ivertex];

      for (irow=0; irow<nflux_in; irow++) {

        PetscInt face_id = face_ids[irow];
        PetscInt *cell_ids, num_cells;
        ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

        cell_id_up = cell_ids[0];
        cell_id_dn = cell_ids[1];

        for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
          col   = vertices->internal_cell_ids[vOffsetCell + icol];
          col   = cells->global_id[vertices->internal_cell_ids[vOffsetCell + icol]];
          if (col<0) col = -col - 1;

          value = -mpfao->Trans[vertex_id][irow][icol];

          row = cells->global_id[cell_id_up];
          if (cells->is_local[cell_id_up]) {ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);}

          row = cells->global_id[cell_id_dn];
          if (cells->is_local[cell_id_dn]) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
        }
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(TDy tdy,
    Mat K,Vec F) {

  TDyMPFAO      *mpfao = tdy->context;
  Conditions    *conditions = tdy->conditions;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  TDySubcell    *subcells = &mesh->subcells;
  PetscInt       ivertex, icell, isubcell, cell_id_up, cell_id_dn;
  PetscInt       irow, icol, row, col, vertex_id;
  PetscInt       ncells, nfaces_bnd;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscInt npcen, npitf_bc, nflux_in;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // Get boundary conditions for each face set.
  PetscInt num_face_sets = 1; // FIXME
  BoundaryConditions bcs[num_face_sets];
  for (PetscInt face_set = 0; face_set < num_face_sets; ++face_set) {
    ConditionsGetBCs(tdy->conditions, face_set, &bcs[face_set]);
  }

    /*

    flux                   =              T                 *     P
            _           _     _                           _    _        _
           |  _       _  |   |  _           _     _     _  |  |  _    _  |
           | |         | |   | |             |   |       | |  | |      | |
           | | flux_in | |   | |    T_00     |   | T_01  | |  | | Pcen | |
           | |         | |   | |             |   |       | |  | |      | |
    flux = | |_       _| | = | |_           _|   |_     _| |  | |      | |
           |             |   |                             |  | |_    _| |
           |  _       _  |   |  _           _     _     _  |  |  _    _  |
           | | flux_bc | |   | |    T_10     |   | T_11  | |  | | Pbc  | |
           | |_       _| |   | |_           _|   |_     _| |  | |_    _| |
           |_           _|   |_                           _|  |_        _|

    */

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    nfaces_bnd= vertices->num_boundary_faces[ivertex];

    if (nfaces_bnd == 0) continue;
    if (ncells < 2)  continue;
    if (!vertices->is_local[ivertex]) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

    PetscInt *face_ids, num_faces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);

    npcen    = vertices->num_internal_cells[ivertex];
    npitf_bc = vertices->num_boundary_faces[ivertex];

    nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    // Vertex is on the boundary

    PetscScalar pBoundary[mpfao->nfv];
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertices->internal_cell_ids[vOffsetCell + irow];
      isubcell = vertices->subcell_ids[vOffsetSubcell + irow];

      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

      PetscInt *face_ids, num_faces;
      ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &face_ids, &num_faces); CHKERRQ(ierr);

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        PetscInt face_id = face_ids[iface];
        PetscInt *cell_ids, num_cells;
        ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

        cell_id_up = cell_ids[0];
        cell_id_dn = cell_ids[1];

        if (!faces->is_internal[face_id]) {
          PetscInt f;
          f = faces->id[face_id] + fStart;
          PetscInt face_set = 0; // FIXME
          // FIXME: can we assume that this face has a pressure BC?
          ierr = EnforceFlowBC(&bcs[face_set].flow_bc, 0.0, 1,
              &(mpfao->X[f*dim]), &pBoundary[numBoundary]);CHKERRQ(ierr);
          numBoundary++;
        }
      }
    }

    for (irow=0; irow<nflux_in; irow++){

      PetscInt face_id = face_ids[irow];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];

      if (cells->is_local[cell_id_up]) {
        row   = cells->global_id[cell_id_up];

        // +T_00
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[vOffsetCell + icol]];
          value = -mpfao->Trans[vertex_id][irow][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = -mpfao->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }

      if (cells->is_local[cell_id_dn]) {

        row   = cells->global_id[cell_id_dn];

        // -T_00
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[vOffsetCell + icol]];
          value = -mpfao->Trans[vertex_id][irow][icol];
          ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = -mpfao->Trans[vertex_id][irow][icol + npcen] *
          pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }

    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<npitf_bc; irow++) {

      //row = cell_ids_from_to[irow][0];

      PetscInt face_id = face_ids[irow + nflux_in];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {

        row   = cells->global_id[cell_id_up];

        // +T_10
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[vOffsetCell + icol]];
          value = -mpfao->Trans[vertex_id][irow+nflux_in][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = -mpfao->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }

      }

      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) {
        row   = cells->global_id[cell_id_dn];

        // -T_10
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[vOffsetCell + icol]];
          value = -mpfao->Trans[vertex_id][irow+nflux_in][icol];
          {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
        }

        //  +T_11 * Pbc
        for (icol=0; icol<vertices->num_boundary_faces[ivertex]; icol++) {
          value = -mpfao->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(TDy tdy,
    Mat K,Vec F) {

  TDyMPFAO      *mpfao = tdy->context;
  Conditions    *conditions = tdy->conditions;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  TDySubcell    *subcells = &mesh->subcells;
  PetscInt       ivertex, icell;
  PetscInt       icol, row, col, iface, isubcell;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      sign;
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  // Get boundary conditions for each face set.
  PetscInt num_face_sets = 1; // FIXME
  BoundaryConditions bcs[num_face_sets];
  for (PetscInt face_set = 0; face_set < num_face_sets; ++face_set) {
    ConditionsGetBCs(tdy->conditions, face_set, &bcs[face_set]);
  }

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (vertices->num_boundary_faces[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

    // Vertex is on the boundary

    PetscScalar pBoundary[mpfao->nfv];
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    icell    = vertices->internal_cell_ids[vOffsetCell + 0];
    isubcell = vertices->subcell_ids[vOffsetSubcell + 0];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

    PetscInt *face_ids, num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &face_ids, &num_faces); CHKERRQ(ierr);

    numBoundary = 0;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = face_ids[iface];

      PetscInt f;
      f = faces->id[face_id] + fStart;
      PetscInt face_set = 0; // FIXME
      // FIXME: Can we assume a pressure boundary condition??
      ierr = EnforceFlowBC(&bcs[face_set].flow_bc, 0.0, 1,
        &(mpfao->X[f*dim]), &pBoundary[numBoundary]);CHKERRQ(ierr);
      numBoundary++;
    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = face_ids[iface];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      row = cell_ids[0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;

      value = 0.0;
      for (j=0; j<dim; j++) {
        value += sign*(-mpfao->Trans[vertices->id[ivertex]][iface][j]);
      }

      row   = cells->global_id[icell];
      col   = cells->global_id[icell];
      if (cells->is_local[icell]) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}

    }

    // For fluxes through boundary edges, only add contribution to the vector
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = face_ids[iface];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      row = cell_ids[0];
      PetscInt cell_id_up = cell_ids[0];
      PetscInt cell_id_dn = cell_ids[1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {
        row   = cells->global_id[cell_id_up];
        for (icol=0; icol<vertices->num_boundary_faces[ivertex]; icol++) {
          value = -mpfao->Trans[vertices->id[ivertex]][iface][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }

      }

      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) {
        row   = cells->global_id[cell_id_dn];
        for (icol=0; icol<vertices->num_boundary_faces[ivertex]; icol++) {
          value = -mpfao->Trans[vertices->id[ivertex]][iface][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

