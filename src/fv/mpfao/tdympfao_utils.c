#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdydiscretizationimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGtimesZ(PetscReal *gravity, PetscReal *X, PetscInt dim, PetscReal *gz) {

  PetscInt d;

  PetscFunctionBegin;

  *gz = 0.0;
  for (d=0;d<dim;d++) *gz += fabs(gravity[d])*X[d];

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOUpdateBoundaryState(TDy tdy) {

  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Loop over boundary faces and assemble a list of "boundary cells" attached
  // to them, plus indices for storing quantities on the boundary.
  // num_boundary_cells <= mesh->num_boundary_faces, so we can pre-size
  // our list of cells.
  PetscInt num_boundary_cells = 0;
  PetscInt boundary_cells[mesh->num_boundary_faces];
  PetscInt p_bnd_indices[mesh->num_boundary_faces];
  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue; // skip non-boundary faces

    PetscInt *cell_ids, num_face_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_face_cells); CHKERRQ(ierr);

    if (cell_ids[0] >= 0) {
      boundary_cells[num_boundary_cells] = cell_ids[0];
      p_bnd_indices[num_boundary_cells] = -cell_ids[1] - 1;
    } else {
      boundary_cells[num_boundary_cells] = cell_ids[1];
      p_bnd_indices[num_boundary_cells] = -cell_ids[0] - 1;
    }
    ++num_boundary_cells;
  }

  // Store the capillary pressure and residual saturation on the boundary.
  PetscReal Pc[num_boundary_cells], Sr[num_boundary_cells];
  for (PetscInt c = 0; c < num_boundary_cells; ++c) {
    PetscInt c_index = boundary_cells[c];
    Sr[c] = mpfao->Sr[c_index];
    PetscInt b_index = p_bnd_indices[c];
    Pc[c] = mpfao->Pref - mpfao->P_bnd[b_index];
  }

  // Compute the saturation and its derivatives on the boundary.
  CharacteristicCurves *cc = tdy->cc;
  PetscReal S[num_boundary_cells], dS_dP[num_boundary_cells],
            d2S_dP2[num_boundary_cells];
  ierr = SaturationComputeOnPoints(cc->saturation, num_boundary_cells,
                                   boundary_cells, Sr, Pc, S, dS_dP, d2S_dP2);
  CHKERRQ(ierr);

  // Compute the effective saturation and its derivative w.r.t. S on the
  // boundary.
  PetscReal Se[num_boundary_cells], dSe_dS[num_boundary_cells];
  for (PetscInt c = 0; c < num_boundary_cells; ++c) {
    Se[c] = (S[c] - Sr[c])/(1.0 - Sr[c]);
    dSe_dS[c] = 1.0/(1.0 - Sr[c]);
  }

  // Compute the relative permeability and its derivative on the boundary.
  PetscReal Kr[num_boundary_cells], dKr_dSe[num_boundary_cells];
  ierr = RelativePermeabilityComputeOnPoints(cc->rel_perm, num_boundary_cells,
                                             boundary_cells, Se, Kr, dKr_dSe);
  CHKERRQ(ierr);

  // Copy the boundary quantities into place.
  for (PetscInt c = 0; c < num_boundary_cells; ++c) {
    PetscInt p_bnd_idx = p_bnd_indices[c];
    mpfao->S_bnd[p_bnd_idx] = S[c];
    mpfao->dS_dP_bnd[p_bnd_idx] = dS_dP[c];
    mpfao->d2S_dP2_bnd[p_bnd_idx] = d2S_dP2[c];
    mpfao->Kr_bnd[p_bnd_idx] = Kr[c];
    mpfao->dKr_dS_bnd[p_bnd_idx] = dKr_dSe[c] * dSe_dS[c];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm = (&tdy->tdydm)->dm;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  TDySubcell    *subcells = &mesh->subcells;
  PetscInt       ivertex, cell_id_up;
  PetscInt       irow, icol, vertex_id;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscScalar    *u;
  Vec            localU;
  PetscReal      vel_normal;
  PetscReal      X[3], vel[3];
  PetscReal      gz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  Conditions *conditions = tdy->conditions;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_faces[ivertex] == 0) {

      PetscInt    nflux_in = vertices->num_faces[ivertex];
      PetscScalar Pcomputed[vertices->num_internal_cells[ivertex]];
      PetscScalar Vcomputed[nflux_in];

      PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];

      PetscInt *face_ids, num_faces;
      ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);

      // Save local pressure stencil and initialize veloctiy
      PetscInt icell, cell_id;
      for (icell=0; icell<vertices->num_internal_cells[ivertex]; icell++) {
        cell_id = vertices->internal_cell_ids[vOffsetCell + icell];
        ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[cell_id].X,dim,&gz);
        Pcomputed[icell] = u[cell_id] + mpfao->rho[cell_id]*gz;
      }

      for (irow=0; irow<nflux_in; irow++) {
        Vcomputed[irow] = 0.0;
      }

      // F = T*P
      for (irow=0; irow<nflux_in; irow++) {

        PetscInt face_id = face_ids[irow];
        PetscInt *cell_ids, num_cells;
        ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

        if (!faces->is_local[face_id]) continue;

        cell_id_up = cell_ids[0];

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);

        PetscInt num_faces;
        PetscReal *face_areas;
        ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

        for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {

          Vcomputed[irow] += -mpfao->Trans[vertex_id][irow][icol]*Pcomputed[icol]/faces->area[face_id];

        }
        mpfao->vel[face_id] += Vcomputed[irow];
        mpfao->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (ConditionsHasBoundaryVelocity(conditions)) {
          ierr = ConditionsComputeBoundaryVelocity(conditions, 1, X,vel);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)* face_areas[iface];

          *vel_error += PetscPowReal( (Vcomputed[irow] - vel_normal), 2.0);
          (*count)++;
        }
      }
    }
  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm = (&tdy->tdydm)->dm;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  TDySubcell    *subcells = &mesh->subcells;
  PetscInt       ivertex, icell, cell_id_up, cell_id_dn;
  PetscInt       irow, icol, vertex_id;
  PetscInt       ncells, nfaces_bnd;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  Vec            localU;
  PetscScalar    *u;
  PetscInt npcen, npitf_bc, nflux_in;
  PetscReal       vel_normal;
  PetscReal       X[3], vel[3];
  PetscReal       gz=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;


  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  Conditions* conditions = tdy->conditions;

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
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertices->internal_cell_ids[vOffsetCell + irow];
      PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + irow];

      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

      PetscInt *face_ids, num_faces;
      ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &face_ids, &num_faces); CHKERRQ(ierr);

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        PetscInt face_id = face_ids[iface];
        PetscInt *cell_ids, num_cells;
        ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_cells); CHKERRQ(ierr);

        cell_id_up = cell_ids[0];
        cell_id_dn = cell_ids[1];

        if (faces->is_internal[face_id] == 0) {
          PetscInt f;
          f = faces->id[face_id] + fStart;
          if (ConditionsHasBoundaryPressure(conditions)) {
            ierr = ConditionsComputeBoundaryPressure(conditions, 1,
              &(mpfao->X[f*dim]), &pBoundary[numBoundary]); CHKERRQ(ierr);
          } else {
            ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[icell].X,dim,&gz); CHKERRQ(ierr);
            pBoundary[numBoundary] = u[icell] + mpfao->rho[icell]*gz;
          }
          numBoundary++;
        }
      }
    }

    for (irow=0; irow<nflux_in; irow++){

      PetscInt face_id = face_ids[irow];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      if (!faces->is_local[face_id]) continue;

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];
      icell = vertices->internal_cell_ids[vOffsetCell + irow];

      if (cells->is_local[cell_id_up]) {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);

        PetscInt num_faces;
        PetscReal *face_areas;
        ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

        // +T_00 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + mpfao->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;

          value += -mpfao->Trans[vertex_id][irow][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -mpfao->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        mpfao->vel[face_id] += value;
        mpfao->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (ConditionsHasBoundaryVelocity(conditions)) {
          ierr = ConditionsComputeBoundaryVelocity(conditions, 1, X,vel);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)* face_areas[iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      } else {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_dn, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);

        PetscInt num_faces;
        PetscReal *face_areas;
        ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

        // -T_00 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + mpfao->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;

          value += -mpfao->Trans[vertex_id][irow][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -mpfao->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        mpfao->vel[face_id] += value;
        mpfao->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (ConditionsHasBoundaryVelocity(conditions)) {
          ierr = ConditionsComputeBoundaryVelocity(conditions, 1, X,vel);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*face_areas[iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }
      }
    }

    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<npitf_bc; irow++) {

      PetscInt face_id = face_ids[irow + nflux_in];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      if (!faces->is_local[face_id]) continue;

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);

        PetscInt num_faces;
        PetscReal *face_areas;
        ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

        // +T_10 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + mpfao->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;
          value += -mpfao->Trans[vertex_id][irow+nflux_in][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -mpfao->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        mpfao->vel[face_id] += value;
        mpfao->vel_count[face_id]++;
        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (ConditionsHasBoundaryVelocity(conditions)) {
          ierr = ConditionsComputeBoundaryVelocity(conditions, 1, X,vel);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*face_areas[iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      } else {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells,cell_id_dn, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);

        PetscInt num_faces;
        PetscReal *face_areas;
        ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

        // -T_10 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + mpfao->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;
          value += -mpfao->Trans[vertex_id][irow+nflux_in][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  +T_11 * Pbc
        for (icol=0; icol<vertices->num_boundary_faces[ivertex]; icol++) {
          value += -mpfao->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        mpfao->vel[face_id] += value;
        mpfao->vel_count[face_id]++;
        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (ConditionsHasBoundaryVelocity(conditions)) {
          ierr = ConditionsComputeBoundaryVelocity(conditions, 1, X,vel);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*face_areas[ iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      }
    }
  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm = (&tdy->tdydm)->dm;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  TDySubcell    *subcells = &mesh->subcells;
  Conditions    *conditions = tdy->conditions;
  PetscInt       ivertex, icell;
  PetscInt       row, iface, isubcell;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscReal      sign;
  PetscInt       j;
  PetscScalar    *u;
  Vec            localU;
  PetscReal      vel_normal;
  PetscReal      X[3], vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (vertices->num_boundary_faces[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

    // Vertex is on the boundary

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    icell    = vertices->internal_cell_ids[vOffsetCell + 0];
    isubcell = vertices->subcell_ids[vOffsetSubcell + 0];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

    PetscScalar pBoundary[subcells->num_faces[subcell_id]];

    ierr = ExtractSubGmatrix(mpfao, icell, isubcell, dim, Gmatrix);

    PetscInt *face_ids, num_faces;
    PetscReal *face_areas;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

    for (iface=0; iface<num_faces; iface++) {

      PetscInt face_id = face_ids[iface];

      PetscInt f;
      f = face_id + fStart;
      if (ConditionsHasBoundaryPressure(conditions)) {
        ierr = ConditionsComputeBoundaryPressure(conditions, 1,
          &(mpfao->X[f*dim]), &pBoundary[iface]); CHKERRQ(ierr);
      } else {
        pBoundary[iface] = u[icell];
      }

    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = face_ids[iface];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_cells); CHKERRQ(ierr);

      if (!faces->is_local[face_id]) continue;

      row = cell_ids[0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;

      value = 0.0;
      for (j=0; j<dim; j++) {
        value -= sign*Gmatrix[iface][j]*(pBoundary[j] - u[icell])/faces->area[face_id];
      }

      //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
      // Should it be '-value' or 'value'?
      value = sign*value;
      mpfao->vel[face_id] += value;
      mpfao->vel_count[face_id]++;
      ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
      if (ConditionsHasBoundaryVelocity(conditions)) {
        ierr = ConditionsComputeBoundaryVelocity(conditions, 1, X,vel);CHKERRQ(ierr);
        vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim) * face_areas[iface];

        *vel_error += PetscPowReal( (value - vel_normal), 2.0);
        (*count)++;
      }
    }

  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAORecoverVelocity(TDy tdy, Vec U) {

  PetscFunctionBegin;
  PetscErrorCode ierr;
  TDyMPFAO *mpfao = tdy->context;
  PetscReal vel_error = 0.0;
  PetscInt count = 0, iface;

  for (iface=0;iface<mpfao->mesh->num_faces;iface++) mpfao->vel[iface] = 0.0;

  ierr = TDyMPFAORecoverVelocity_InternalVertices(tdy, U, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices(tdy, U, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices(tdy, U, &vel_error, &count); CHKERRQ(ierr);

  PetscReal vel_error_sum;
  PetscInt  count_sum;

  ierr = MPI_Allreduce(&vel_error,&vel_error_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&count,&count_sum,1,MPI_INT,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);

  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank==0) printf("%15.14f ",PetscPowReal(vel_error_sum/count_sum,0.5));

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDy tdy, Vec Ul) {

  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscInt dim;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal *p_vec_ptr, *u_p;
  PetscInt c, cStart, cEnd;
  Conditions *conditions = tdy->conditions;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum((&tdy->tdydm)->dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscInt ncells = mesh->num_cells;
  PetscReal p[ncells];
  if (mpfao->Temp_subc_Gmatrix) { // TH
    for (c=0;c<ncells;c++) {
      p[c] = u_p[c*2];
    }
  }
  else {
    for (c=0;c<ncells;c++)
      p[c] = u_p[c];
  }


  ierr = DMGetDimension((&tdy->tdydm)->dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

    PetscInt *cell_ids, num_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_cells); CHKERRQ(ierr);

    if (cell_ids[0] >= 0) {
      cell_id = cell_ids[0];
      p_bnd_idx = -cell_ids[1] - 1;
    } else {
      cell_id = cell_ids[1];
      p_bnd_idx = -cell_ids[0] - 1;
    }

    if (ConditionsHasBoundaryPressure(conditions)) {
      ierr = ConditionsComputeBoundaryPressure(conditions, 1,
        faces->centroid[iface].X, &(mpfao->P_bnd[p_bnd_idx])); CHKERRQ(ierr);
    } else {
      mpfao->P_bnd[p_bnd_idx] = p[cell_id];
    }

    p_vec_ptr[p_bnd_idx + ncells] = mpfao->P_bnd[p_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDy tdy, Vec Ul) {

  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscInt dim;
  PetscInt t_bnd_idx, cell_id, iface;
  PetscReal *t_vec_ptr, *u_p;
  PetscInt c, cStart, cEnd;
  Conditions *conditions = tdy->conditions;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum((&tdy->tdydm)->dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->Temp_P_vec,&t_vec_ptr); CHKERRQ(ierr);

  PetscInt ncells = mesh->num_cells;
  PetscReal t[ncells];
  for (c=0;c<ncells;c++) {
    t[c] = u_p[c*2+1];
  }

  ierr = DMGetDimension((&tdy->tdydm)->dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

    PetscInt *cell_ids, num_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_cells); CHKERRQ(ierr);

    if (cell_ids[0] >= 0) {
      cell_id = cell_ids[0];
      t_bnd_idx = -cell_ids[1] - 1;
    } else {
      cell_id = cell_ids[1];
      t_bnd_idx = -cell_ids[0] - 1;
    }

    if (ConditionsHasBoundaryTemperature(conditions)) {
      ierr = ConditionsComputeBoundaryTemperature(conditions, 1,
        faces->centroid[iface].X, &(mpfao->T_bnd[t_bnd_idx])); CHKERRQ(ierr);
    } else {
      mpfao->T_bnd[t_bnd_idx] = t[cell_id];
    }

    t_vec_ptr[t_bnd_idx + ncells] = mpfao->T_bnd[t_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->Temp_P_vec,&t_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ExtractSubGmatrix(TDyMPFAO *mpfao, PetscInt cell_id,
                                 PetscInt sub_cell_id, PetscInt dim,
                                 PetscReal **Gmatrix) {

  PetscInt i, j;

  PetscFunctionBegin;

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      Gmatrix[i][j] = mpfao->subc_Gmatrix[cell_id][sub_cell_id][i][j];
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ExtractTempSubGmatrix(TDyMPFAO *mpfao, PetscInt cell_id,
                                     PetscInt sub_cell_id, PetscInt dim,
                                     PetscReal **Gmatrix) {

  PetscInt i, j;

  PetscFunctionBegin;

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      Gmatrix[i][j] = mpfao->Temp_subc_Gmatrix[cell_id][sub_cell_id][i][j];
    }
  }

  PetscFunctionReturn(0);
}
