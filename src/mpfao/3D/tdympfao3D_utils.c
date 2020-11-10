#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGtimesZ(PetscReal *gravity, PetscReal *X, PetscInt dim, PetscReal *gz) {

  PetscInt d;

  PetscFunctionBegin;
  
  *gz = 0.0;
  if (dim == 3) {
    for (d=0;d<dim;d++) *gz += fabs(gravity[d])*X[d];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyUpdateBoundaryState(TDy tdy) {

  TDy_mesh *mesh;
  TDy_face *faces;
  PetscErrorCode ierr;
  PetscReal Se,dSe_dS,dKr_dSe,n=0.5,m=0.8,alpha=1.e-4,Kr; /* FIX: generalize */
  PetscInt dim;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal Sr,S,dS_dP,d2S_dP2,P;

  PetscFunctionBegin;

  mesh = tdy->mesh;
  faces = &mesh->faces;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  CharacteristicCurve cc = tdy->cc;
  CharacteristicCurve cc_bnd = tdy->cc_bnd;
  MaterialProp matprop = tdy->matprop;

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

    PetscInt fOffsetCell = faces->cell_offset[iface];

    if (faces->cell_ids[fOffsetCell + 0] >= 0) {
      cell_id = faces->cell_ids[fOffsetCell + 0];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
    } else {
      cell_id = faces->cell_ids[fOffsetCell + 1];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;
    }

    switch (cc->SatFuncType[cell_id]) {
    case SAT_FUNC_GARDNER :
      Sr = cc->sr[cell_id];
      P = tdy->Pref - tdy->P_BND[p_bnd_idx];

      PressureSaturation_Gardner(n,m,alpha,Sr,P,&S,&dS_dP,&d2S_dP2);
      break;
    case SAT_FUNC_VAN_GENUCHTEN :
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
      RelativePermeability_Irmay(m,Se,&Kr,NULL);
      break;
    case REL_PERM_FUNC_MUALEM :
      RelativePermeability_Mualem(m,Se,&Kr,&dKr_dSe);
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

PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices_3DMesh(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
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

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_faces[ivertex] == 0) {

      PetscInt    nflux_in = vertices->num_faces[ivertex];
      PetscScalar Pcomputed[vertices->num_internal_cells[ivertex]];
      PetscScalar Vcomputed[nflux_in];

      PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
      PetscInt vOffsetFace    = vertices->face_offset[ivertex];

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

        PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
        PetscInt fOffsetCell = faces->cell_offset[face_id];

        if (!faces->is_local[face_id]) continue;

        cell_id_up = faces->cell_ids[fOffsetCell + 0];

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->face_offset[subcell_id];

        for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {

          Vcomputed[irow] += -tdy->Trans[vertex_id][irow][icol]*Pcomputed[icol]/faces->area[face_id];
          
        }
        tdy->vel[face_id] += Vcomputed[irow];
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

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
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
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

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
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
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

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
      PetscInt sOffsetFace = subcells->face_offset[subcell_id];

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
        PetscInt fOffsetCell = faces->cell_offset[face_id];

        cell_id_up = faces->cell_ids[fOffsetCell + 0];
        cell_id_dn = faces->cell_ids[fOffsetCell + 1];

        if (faces->is_internal[face_id] == 0) {
          PetscInt f;
          f = faces->id[face_id] + fStart;
          if (tdy->ops->computedirichletvalue) {
            ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
          } else {
            ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[icell].X,dim,&gz); CHKERRQ(ierr);
            pBoundary[numBoundary] = u[icell] + tdy->rho[icell]*gz;
          }
          numBoundary++;
        }
      }
    }

    for (irow=0; irow<nflux_in; irow++){

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      if (!faces->is_local[face_id]) continue;

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      icell = vertices->internal_cell_ids[vOffsetCell + irow];

      if (cells->is_local[cell_id_up]) {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->face_offset[subcell_id];

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
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      } else {
      
        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_dn, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->face_offset[subcell_id];

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
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }
      }
    }
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<npitf_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow + nflux_in];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      if (!faces->is_local[face_id]) continue;

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->face_offset[subcell_id];

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
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      } else {
      
        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells,cell_id_dn, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->face_offset[subcell_id];

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
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

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
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
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

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
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
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    PetscScalar pBoundary[subcells->num_faces[subcell_id]];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {
      
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];

      PetscInt f;
      f = face_id + fStart;
      if (tdy->ops->computedirichletvalue) {
        ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[iface], tdy->dirichletvaluectx);CHKERRQ(ierr);
      } else {
        pBoundary[iface] = u[icell];
      }

    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      if (!faces->is_local[face_id]) continue;

      row = faces->cell_ids[fOffsetCell + 0];
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
      if (tdy->ops->computedirichletflux) {
        ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
        vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

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

PetscErrorCode TDyMPFAORecoverVelocity_3DMesh(TDy tdy, Vec U) {

  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscReal vel_error = 0.0;
  PetscInt count = 0, iface;

  for (iface=0;iface<tdy->mesh->num_faces;iface++) tdy->vel[iface] = 0.0;

  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, U, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, U, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, U, &vel_error, &count); CHKERRQ(ierr);

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

  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  PetscErrorCode ierr;
  PetscInt dim, ncells;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal *p, gz, *p_vec_ptr, *u_p;
  PetscInt c, cStart, cEnd;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&p);CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  if (tdy->mode == TH) {
    for (c=0;c<cEnd-cStart;c++) {
      p[c] = u_p[c*2];
    }
  }
  else {
    for (c=0;c<cEnd-cStart;c++) p[c] = u_p[c];
  }

  mesh = tdy->mesh;
  cells = &mesh->cells;
  faces = &mesh->faces;
  ncells = mesh->num_cells;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

      PetscInt fOffsetCell = faces->cell_offset[iface];

    if (faces->cell_ids[fOffsetCell + 0] >= 0) {
      cell_id = faces->cell_ids[fOffsetCell + 0];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
    } else {
      cell_id = faces->cell_ids[fOffsetCell + 1];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;
    }

    if (tdy->ops->computedirichletvalue) {
      ierr = (*tdy->ops->computedirichletvalue)(tdy, (faces->centroid[iface].X), &(tdy->P_BND[p_bnd_idx]), tdy->dirichletvaluectx);CHKERRQ(ierr);
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      tdy->P_BND[p_bnd_idx] += tdy->rho[cell_id]*gz;
    } else {
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      tdy->P_BND[p_bnd_idx] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    p_vec_ptr[p_bnd_idx + ncells] = tdy->P_BND[p_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDy tdy, Vec Ul) {

  TDy_mesh *mesh;
  TDy_face *faces;
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

  mesh = tdy->mesh;
  faces = &mesh->faces;
  ncells = mesh->num_cells;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

      PetscInt fOffsetCell = faces->cell_offset[iface];

    if (faces->cell_ids[fOffsetCell + 0] >= 0) {
      cell_id = faces->cell_ids[fOffsetCell + 0];
      t_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
    } else {
      cell_id = faces->cell_ids[fOffsetCell + 1];
      t_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;
    }

    if (tdy->ops->computetemperaturedirichletvalue) {
      ierr = (*tdy->ops->computetemperaturedirichletvalue)(tdy, (faces->centroid[iface].X), &(tdy->T_BND[t_bnd_idx]), tdy->temperaturedirichletvaluectx);CHKERRQ(ierr);
    } else {
      tdy->T_BND[t_bnd_idx] = t[cell_id];
    }

    t_vec_ptr[t_bnd_idx + ncells] = tdy->T_BND[t_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->Temp_P_vec,&t_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

