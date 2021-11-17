#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdydiscretization.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGtimesZ(PetscReal *gravity, PetscReal *X, PetscInt dim, PetscReal *gz) {

  PetscInt d;

  PetscFunctionBegin;

  *gz = 0.0;
  for (d=0;d<dim;d++) *gz += fabs(gravity[d])*X[d];

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyUpdateBoundaryState(TDy tdy) {

  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscReal Se,dSe_dS,dKr_dSe,n,m,alpha,Kr;
  PetscInt dim;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal Sr,S,dS_dP,d2S_dP2,P;

  PetscFunctionBegin;


  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  CharacteristicCurve *cc = tdy->cc;
  CharacteristicCurve *cc_bnd = tdy->cc_bnd;

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

    switch (cc->SatFuncType[cell_id]) {
    case SAT_FUNC_GARDNER :
      n = cc->gardner_n[cell_id];
      m = cc->gardner_m[cell_id];
      alpha = cc->vg_alpha[cell_id];
      Sr = cc->sr[cell_id];
      P = tdy->Pref - tdy->P_BND[p_bnd_idx];

      PressureSaturation_Gardner(n,m,alpha,Sr,P,&S,&dS_dP,&d2S_dP2);
      break;
    case SAT_FUNC_VAN_GENUCHTEN :
      n = cc->gardner_n[cell_id];
      m = cc->vg_m[cell_id];
      alpha = cc->vg_alpha[cell_id];
      Sr = cc->sr[cell_id];
      P = tdy->Pref - tdy->P_BND[p_bnd_idx];

      PressureSaturation_VanGenuchten(m,alpha,Sr,P,&S,&dS_dP,&d2S_dP2);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown saturation function");
      break;
    }

    Se = (S - Sr)/(1.0 - Sr);
    dSe_dS = 1.0/(1.0 - Sr);

    switch (cc->RelPermFuncType[cell_id]) {
    case REL_PERM_FUNC_IRMAY :
      m = cc->irmay_m[cell_id];
      RelativePermeability_Irmay(m,Se,&Kr,NULL);
      break;
    case REL_PERM_FUNC_MUALEM :
      m = cc->mualem_m[cell_id];
      RelativePermeability_Mualem(m,cc->mualem_poly_low[cell_id],cc->mualem_poly_coeffs[cell_id],Se,&Kr,&dKr_dSe);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown relative permeability function");
      break;
    }

    cc_bnd->S[p_bnd_idx] = S;
    cc_bnd->dS_dP[p_bnd_idx] = dS_dP;
    cc_bnd->d2S_dP2[p_bnd_idx] = d2S_dP2;
    cc_bnd->Kr[p_bnd_idx] = Kr;
    cc_bnd->dKr_dS[p_bnd_idx] = dKr_dSe * dSe_dS;

    //for(j=0; j<dim2; j++) matprop->K[i*dim2+j] = matprop->K0[i*dim2+j] * Kr;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
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
  ierr = TDyGlobalToLocal(tdy,U,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

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
        ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz);
        Pcomputed[icell] = u[cell_id] + tdy->rho[cell_id]*gz;
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

          Vcomputed[irow] += -tdy->Trans[vertex_id][irow][icol]*Pcomputed[icol]/faces->area[face_id];

        }
        tdy->vel[face_id] += Vcomputed[irow];
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->compute_boundary_velocity) {
          ierr = (*tdy->ops->compute_boundary_velocity)(tdy,X,vel,tdy->boundary_velocity_ctx);CHKERRQ(ierr);
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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
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
  ierr = TDyGlobalToLocal(tdy,U,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

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

    PetscScalar pBoundary[tdy->nfv];
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
          if (tdy->ops->compute_boundary_pressure) {
            ierr = (*tdy->ops->compute_boundary_pressure)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->boundary_pressure_ctx);CHKERRQ(ierr);
          } else {
            ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[icell].X,dim,&gz); CHKERRQ(ierr);
            pBoundary[numBoundary] = u[icell] + tdy->rho[icell]*gz;
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
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;

          value += -tdy->Trans[vertex_id][irow][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->compute_boundary_velocity) {
          ierr = (*tdy->ops->compute_boundary_velocity)(tdy,X,vel,tdy->boundary_velocity_ctx);CHKERRQ(ierr);
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
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;

          value += -tdy->Trans[vertex_id][irow][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->compute_boundary_velocity) {
          ierr = (*tdy->ops->compute_boundary_velocity)(tdy,X,vel,tdy->boundary_velocity_ctx);CHKERRQ(ierr);
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
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;
        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->compute_boundary_velocity) {
          ierr = (*tdy->ops->compute_boundary_velocity)(tdy,X,vel,tdy->boundary_velocity_ctx);CHKERRQ(ierr);
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
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[vOffsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[vOffsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[vOffsetCell + icol]]*gz;
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol]*Pcomputed/faces->area[face_id];
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  +T_11 * Pbc
        for (icol=0; icol<vertices->num_boundary_faces[ivertex]; icol++) {
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol]/faces->area[face_id];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;
        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->compute_boundary_velocity) {
          ierr = (*tdy->ops->compute_boundary_velocity)(tdy,X,vel,tdy->boundary_velocity_ctx);CHKERRQ(ierr);
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

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  TDyFace       *faces = &mesh->faces;
  TDySubcell    *subcells = &mesh->subcells;
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
  ierr = TDyGlobalToLocal(tdy,U,localU); CHKERRQ(ierr);
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

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);

    PetscInt *face_ids, num_faces;
    PetscReal *face_areas;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &face_areas, &num_faces); CHKERRQ(ierr);

    for (iface=0; iface<num_faces; iface++) {

      PetscInt face_id = face_ids[iface];

      PetscInt f;
      f = face_id + fStart;
      if (tdy->ops->compute_boundary_pressure) {
        ierr = (*tdy->ops->compute_boundary_pressure)(tdy, &(tdy->X[f*dim]), &pBoundary[iface], tdy->boundary_pressure_ctx);CHKERRQ(ierr);
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
      tdy->vel[face_id] += value;
      tdy->vel_count[face_id]++;
      ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
      if (tdy->ops->compute_boundary_velocity) {
        ierr = (*tdy->ops->compute_boundary_velocity)(tdy,X,vel,tdy->boundary_velocity_ctx);CHKERRQ(ierr);
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
  PetscReal vel_error = 0.0;
  PetscInt count = 0, iface;

  for (iface=0;iface<tdy->mesh->num_faces;iface++) tdy->vel[iface] = 0.0;

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

  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscInt dim, ncells;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal *p, *p_vec_ptr, *u_p;
  PetscInt c, cStart, cEnd;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&p);CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  if (tdy->options.mode == TH) {
    for (c=0;c<cEnd-cStart;c++) {
      p[c] = u_p[c*2];
    }
  }
  else if (tdy->options.mode == SALINITY) {
    for (c=0;c<cEnd-cStart;c++) {
      p[c] = u_p[c*2];
    }
  }
  else {
    for (c=0;c<cEnd-cStart;c++) p[c] = u_p[c];
  }

  ncells = mesh->num_cells;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

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

    if (tdy->ops->compute_boundary_pressure) {
      ierr = (*tdy->ops->compute_boundary_pressure)(tdy, (faces->centroid[iface].X), &(tdy->P_BND[p_bnd_idx]), tdy->boundary_pressure_ctx);CHKERRQ(ierr);
    } else {
      tdy->P_BND[p_bnd_idx] = p[cell_id];
    }

    p_vec_ptr[p_bnd_idx + ncells] = tdy->P_BND[p_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDy tdy, Vec Ul) {

  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscInt dim, ncells;
  PetscInt t_bnd_idx, cell_id, iface;
  PetscReal *t, *t_vec_ptr, *u_p;
  PetscInt c, cStart, cEnd;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&t);CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->Temp_P_vec,&t_vec_ptr); CHKERRQ(ierr);

  for (c=0;c<cEnd-cStart;c++) {
    t[c] = u_p[c*2+1];
  }

  ncells = mesh->num_cells;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

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

    if (tdy->ops->compute_boundary_temperature) {
      ierr = (*tdy->ops->compute_boundary_temperature)(tdy, (faces->centroid[iface].X), &(tdy->T_BND[t_bnd_idx]), tdy->boundary_temperature_ctx);CHKERRQ(ierr);
    } else {
      tdy->T_BND[t_bnd_idx] = t[cell_id];
    }

    t_vec_ptr[t_bnd_idx + ncells] = tdy->T_BND[t_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->Temp_P_vec,&t_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_SetBoundaryConcentration(TDy tdy, Vec Ul) {

  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscInt dim, ncells;
  PetscInt psi_bnd_idx, cell_id, iface;
  PetscReal *psi, *psi_vec_ptr, *u_p;
  PetscInt c, cStart, cEnd;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&psi);CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->Psi_vec,&psi_vec_ptr); CHKERRQ(ierr);

  for (c=0;c<cEnd-cStart;c++) {
    psi[c] = u_p[c*2+1];
  }

  ncells = mesh->num_cells;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

    PetscInt *cell_ids, num_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_cells); CHKERRQ(ierr);

    if (cell_ids[0] >= 0) {
      cell_id = cell_ids[0];
      psi_bnd_idx = -cell_ids[1] - 1;
    } else {
      cell_id = cell_ids[1];
      psi_bnd_idx = -cell_ids[0] - 1;
    }

    //   if (tdy->ops->compute_boundary_temperature) {
    //   ierr = (*tdy->ops->compute_boundary_temperature)(tdy, (faces->centroid[iface].X), &(tdy->T_BND[t_bnd_idx]), tdy->boundary_temperature_ctx);CHKERRQ(ierr);
    //   } else {
      tdy->Psi_BND[psi_bnd_idx] = psi[cell_id];
      // }

    psi_vec_ptr[psi_bnd_idx + ncells] = tdy->Psi_BND[psi_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->Psi_vec,&psi_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
