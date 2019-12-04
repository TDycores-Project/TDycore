#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdysaturationimpl.h>
#include <private/tdypermeabilityimpl.h>
#include <private/tdympfao3Dutilsimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_InternalVertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex, vertex_id;
  PetscInt dim;
  PetscInt irow, icol;
  PetscInt cell_id_up, cell_id_dn, cell_id, icell;
  PetscReal gz,den,fluxm,ukvr;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] != 0) continue;
    PetscInt vOffsetFace = vertices->face_offset[ivertex];
    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];

    PetscInt nflux_in = 12;
    PetscScalar P[vertices->num_internal_cells[ivertex]];
    PetscScalar TtimesP[nflux_in];

    // Build the P vector
    for (icell=0; icell<vertices->num_internal_cells[ivertex]; icell++) {
      cell_id = vertices->internal_cell_ids[vOffsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Compute = T*P
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];

      if (!faces->is_local[face_id]) continue;
      
      TtimesP[irow] = 0.0;

      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    //
    // fluxm_ij = rho_ij * (kr/mu)_{ij,upwind} * [ T ] *  [ P+rho*g*z ]^T
    // where
    //      rho_ij = 0.5*(rho_i + rho_j)
    //      (kr/mu)_{ij,upwind} = (kr/mu)_{i} if velocity is from i to j
    //                          = (kr/mu)_{j} otherwise
    //      T includes product of K and A_{ij}
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      
      if (TtimesP[irow] < 0.0) ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
      else                     ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
      
      den = 0.5*(tdy->rho[cell_id_up] + tdy->rho[cell_id_dn]);
      fluxm = den*ukvr*(-TtimesP[irow]);
      
      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cells->is_local[cell_id_up]) r[cell_id_up] += fluxm;
      if (cells->is_local[cell_id_dn]) r[cell_id_dn] -= fluxm;
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex, vertex_id;
  PetscInt dim;
  PetscInt ncells, ncells_bnd;
  PetscInt npcen, npitf_bc, nflux_bc, nflux_in;
  TDy_subcell    *subcells;
  PetscInt irow, icol;
  PetscInt cell_id_up, cell_id_dn, cell_id, icell;
  PetscReal gz,den,fluxm,ukvr;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  subcells = &mesh->subcells;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells     <  2) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    npcen    = vertices->num_internal_cells[ivertex];
    npitf_bc = vertices->num_boundary_cells[ivertex];
    nflux_bc = npitf_bc/2;

    switch (ncells) {
    case 2:
      nflux_in = 1;
      break;
    case 4:
      nflux_in = 4;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Unsupported number of internal cells.");
      break;
    }

    PetscScalar pBoundary[4];
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

        if (faces->is_internal[face_id] == 0) {

          // Extract pressure value at the boundary
          PetscInt p_bnd_idx;
          if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
          else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

          pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];

          numBoundary++;
        }
      }
    }
    
    PetscReal P[npcen + npitf_bc];

    // Save intenral pressure values
    for (icell=0;icell<npcen;icell++) {
      cell_id = vertices->internal_cell_ids[vOffsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) {
      P[icell+npcen] = pBoundary[icell];
    }

    // Compute T*P
    PetscScalar TtimesP[nflux_in + 2*nflux_bc];
    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<npcen + npitf_bc; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      if (!faces->is_local[face_id]) continue;

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      if (cell_id_up<0 || cell_id_dn<0) continue;

      if (TtimesP[irow] < 0.0) { // up ---> dn
        if (cell_id_up>=0) ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
        else               ukvr = tdy->Kr_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
      } else {
        if (cell_id_dn>=0) ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
        else               ukvr = tdy->Kr_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      fluxm = den*ukvr*(-TtimesP[irow]);
      
      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up>-1 && cells->is_local[cell_id_up]) r[cell_id_up] += fluxm;
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) r[cell_id_dn] -= fluxm;

    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex, vertex_id;
  PetscInt dim;
  TDy_subcell    *subcells;
  PetscInt irow, icol;
  PetscInt isubcell, iface;
  PetscInt cell_id_up, cell_id_dn, cell_id, icell;
  PetscReal gz,den,fluxm,ukvr;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  subcells = &mesh->subcells;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

    // Vertex is on the boundary
    PetscScalar pBoundary[3];
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    cell_id  = vertices->internal_cell_ids[vOffsetCell + 0];
    isubcell = vertices->subcell_ids[vOffsetSubcell + 0];

    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    numBoundary = 0;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Extract pressure value at the boundary
      PetscInt p_bnd_idx;
      if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
      else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

      pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];
      numBoundary++;

    }

    PetscReal P[numBoundary+1];

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) P[icell] = pBoundary[icell];

    // Save internal pressure value
    gz = 0.0;
    ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
    P[numBoundary] = p[cell_id] + tdy->rho[cell_id]*gz;

    // Compute T*P
    PetscScalar TtimesP[numBoundary];
    for (irow=0; irow<numBoundary; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<numBoundary+1; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];


      if (TtimesP[irow] < 0.0) { // up ---> dn
        if (cell_id_up>=0) ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
        else               ukvr = tdy->Kr_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
      } else {
        if (cell_id_dn>=0) ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
        else               ukvr = tdy->Kr_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      fluxm = den*ukvr*(-TtimesP[irow]);

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up>-1 && cells->is_local[cell_id_up]) r[cell_id_up] += fluxm;
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) r[cell_id_dn] -= fluxm;
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {
  
  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*dp_dt,*r;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);

  PetscReal vel_error = 0.0;
  PetscInt count = 0;
  PetscInt iface;

  for (iface=0;iface<mesh->num_faces;iface++) tdy->vel[iface] = 0.0;
  
  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);

  ierr = TDyMPFAOIFunction_InternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscReal dporosity_dP = 0.0;
  PetscReal dmass_dP;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dP * dP/dt * Vol
    dmass_dP = tdy->rho[icell]     * dporosity_dP         * tdy->S[icell] +
               tdy->drho_dP[icell] * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dP[icell];
    r[icell] += dmass_dP * dp_dt[icell] * cells->volume[icell];
    r[icell] -= tdy->source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_InternalVertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt ivertex, vertex_id;
  PetscInt icell, cell_id, cell_id_up, cell_id_dn;
  PetscInt irow, icol;
  PetscInt dim;
  PetscReal gz;
  PetscReal *p;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal T;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] != 0) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    PetscInt nflux_in = 12;
    PetscScalar P[vertices->num_internal_cells[ivertex]];
    PetscScalar TtimesP[nflux_in];

    // Build the P vector
    for (icell=0; icell<vertices->num_internal_cells[ivertex]; icell++) {
      cell_id = vertices->internal_cell_ids[vOffsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Compute = T*P
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      
      if (!faces->is_local[face_id]) continue;
      
      TtimesP[irow] = 0.0;

      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    //
    // fluxm_ij = rho_ij * (kr/mu)_{ij,upwind} * [ T ] *  [ P+rho*g*z ]^T
    // where
    //      rho_ij = 0.5*(rho_i + rho_j)
    //      (kr/mu)_{ij,upwind} = (kr/mu)_{i} if velocity is from i to j
    //                          = (kr/mu)_{j} otherwise
    //      T includes product of K and A_{ij}
    //
    // For i and j, jacobian is given as:
    // d(fluxm_ij)/dP_i = d(rho_ij)/dP_i + (kr/mu)_{ij,upwind}         * [  T  ] *  [    P+rho*g*z   ]^T +
    //                      rho_ij       + d((kr/mu)_{ij,upwind})/dP_i * [  T  ] *  [    P+rho*g*z   ]^T +
    //                      rho_ij       + (kr/mu)_{ij,upwind}         *   T_i   *  (1+d(rho_i)/dP_i*g*z
    //
    // For k not equal to i and j, jacobian is given as:
    // d(fluxm_ij)/dP_k =   rho_ij       + (kr/mu)_{ij,upwind}         *   T_ik  *  (1+d(rho_k)/dP_k*g*z
    //
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      
      dukvr_dPup = 0.0;
      dukvr_dPdn = 0.0;

      if (TtimesP[irow] < 0.0) {
        ukvr       = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
        dukvr_dPup = tdy->dKr_dS[cell_id_up]*tdy->dS_dP[cell_id_up]/tdy->vis[cell_id_up] -
                     tdy->Kr[cell_id_up]/(tdy->vis[cell_id_up]*tdy->vis[cell_id_up])*tdy->dvis_dP[cell_id_up];
      } else {
        ukvr       = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
        dukvr_dPdn = tdy->dKr_dS[cell_id_dn]*tdy->dS_dP[cell_id_dn]/tdy->vis[cell_id_dn] -
                     tdy->Kr[cell_id_dn]/(tdy->vis[cell_id_dn]*tdy->vis[cell_id_dn])*tdy->dvis_dP[cell_id_dn];
      }

      den = 0.5*(tdy->rho[cell_id_up] + tdy->rho[cell_id_dn]);
      dden_dPup = tdy->drho_dP[cell_id_up];
      dden_dPdn = tdy->drho_dP[cell_id_dn];

      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        cell_id = vertices->internal_cell_ids[vOffsetCell + icol];
        
        T = tdy->Trans[vertex_id][irow][icol];
        
        ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

        if (cell_id == cell_id_up) {
          Jac =
            dden_dPup * ukvr       * TtimesP[irow] +
            den       * dukvr_dPup * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dPup*gz) ;
        } else if (cell_id == cell_id_dn) {
          Jac =
            dden_dPdn * ukvr       * TtimesP[irow] +
            den       * dukvr_dPdn * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dPdn*gz) ;
        } else {
          Jac = den * ukvr * T * (1.0 + 0.*gz);
        }
        if (fabs(Jac)<PETSC_MACHINE_EPSILON) Jac = 0.0;

        // Changing sign when bringing the term from RHS to LHS of the equation
        Jac = -Jac;

        if (cells->is_local[cell_id_up]) {
          ierr = MatSetValuesLocal(A,1,&cell_id_up,1,&cell_id,&Jac,ADD_VALUES);CHKERRQ(ierr);
        }

        if (cells->is_local[cell_id_dn]) {
          Jac = -Jac;
          ierr = MatSetValuesLocal(A,1,&cell_id_dn,1,&cell_id,&Jac,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt ncells, ncells_bnd;
  PetscInt npcen, npitf_bc, nflux_bc, nflux_in;
  PetscInt fStart, fEnd;
  TDy_subcell    *subcells;
  PetscInt dim;
  PetscInt icell, ivertex;
  PetscInt cell_id, cell_id_up, cell_id_dn, vertex_id;
  PetscInt irow, icol;
  PetscReal T;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal *p;
  PetscReal gz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  subcells = &mesh->subcells;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells     <  2) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    npcen    = vertices->num_internal_cells[ivertex];
    npitf_bc = vertices->num_boundary_cells[ivertex];
    nflux_bc = npitf_bc/2;

    switch (ncells) {
    case 2:
      nflux_in = 1;
      break;
    case 4:
      nflux_in = 4;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Unsupported number of internal cells.");
      break;
    }

    PetscScalar pBoundary[4];
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

        if (faces->is_internal[face_id] == 0) {

          // Extract pressure value at the boundary
          PetscInt p_bnd_idx;
          if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
          else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

          pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];

          numBoundary++;
        }
      }
    }
    
    PetscReal P[npcen + npitf_bc];

    // Save intenral pressure values
    for (icell=0;icell<npcen;icell++) {
      cell_id = vertices->internal_cell_ids[vOffsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) {
      P[icell+npcen] = pBoundary[icell];
    }

    // Compute T*P
    PetscScalar TtimesP[nflux_in + 2*nflux_bc];
    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<npcen + npitf_bc; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      if (!faces->is_local[face_id]) continue;

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      dukvr_dPup = 0.0;
      dukvr_dPdn = 0.0;

      if (TtimesP[irow] < 0.0) {
        // Flow: up --> dn
        if (cell_id_up>=0) {
          // "up" is an internal cell
          ukvr       = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
          dukvr_dPup = tdy->dKr_dS[cell_id_up]*tdy->dS_dP[cell_id_up]/tdy->vis[cell_id_up] -
                       tdy->Kr[cell_id_up]/(tdy->vis[cell_id_up]*tdy->vis[cell_id_up])*tdy->dvis_dP[cell_id_up];
        } else {
          // "up" is boundary cell
          ukvr       = tdy->Kr_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
          dukvr_dPup = 0.0;//tdy->dKr_dS[-cell_id_up-1]*tdy->dS_dP_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
        }

      } else {
        // Flow: up <--- dn
        if (cell_id_dn>=0) {
          // "dn" is an internal cell
          ukvr       = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
          dukvr_dPdn = tdy->dKr_dS[cell_id_dn]*tdy->dS_dP[cell_id_dn]/tdy->vis[cell_id_dn] -
                       tdy->Kr[cell_id_dn]/(tdy->vis[cell_id_dn]*tdy->vis[cell_id_dn])*tdy->dvis_dP[cell_id_dn];
        } else {
          // "dn" is a boundary cell
          ukvr       = tdy->Kr_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
          dukvr_dPdn = 0.0;//tdy->dKr_dS_BND[-cell_id_dn-1]*tdy->dS_dP_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
        }
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      if (cell_id_up<0) cell_id_up = cell_id_dn;
      if (cell_id_dn<0) cell_id_dn = cell_id_up;

      dden_dPup = 0.0;
      dden_dPdn = 0.0;

      if (cell_id_up>=0) dden_dPup = tdy->drho_dP[cell_id_up];
      if (cell_id_dn>=0) dden_dPdn = tdy->drho_dP[cell_id_dn];

      // Deriviates will be computed only w.r.t. internal pressure
      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        cell_id = vertices->internal_cell_ids[vOffsetCell + icol];
        
        T = tdy->Trans[vertex_id][irow][icol];
        
        ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

        if (cell_id_up>-1 && cell_id == cell_id_up) {
          Jac =
            dden_dPup * ukvr       * TtimesP[irow] +
            den       * dukvr_dPup * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dPup*gz) ;
        } else if (cell_id_dn>-1 && cell_id == cell_id_dn) {
          Jac =
            dden_dPdn * ukvr       * TtimesP[irow] +
            den       * dukvr_dPdn * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dPdn*gz) ;
        } else {
          Jac = den * ukvr * T * (1.0 + 0.0*gz);
        }
        if (fabs(Jac)<PETSC_MACHINE_EPSILON) Jac = 0.0;

        // Changing sign when bringing the term from RHS to LHS of the equation
        Jac = -Jac;

        if (cell_id_up >-1 && cells->is_local[cell_id_up]) {
          ierr = MatSetValuesLocal(A,1,&cell_id_up,1,&cell_id,&Jac,ADD_VALUES);CHKERRQ(ierr);
        }

        if (cell_id_dn >-1 && cells->is_local[cell_id_dn]) {
          Jac = -Jac;
          ierr = MatSetValuesLocal(A,1,&cell_id_dn,1,&cell_id,&Jac,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt fStart, fEnd;
  TDy_subcell    *subcells;
  PetscInt dim;
  PetscInt icell, ivertex;
  PetscInt isubcell, iface;
  PetscInt cell_id, cell_id_up, cell_id_dn, vertex_id;
  PetscInt irow, icol;
  PetscReal T;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal *p;
  PetscReal gz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  subcells = &mesh->subcells;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

    // Vertex is on the boundary
    PetscScalar pBoundary[3];
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    cell_id  = vertices->internal_cell_ids[vOffsetCell + 0];
    isubcell = vertices->subcell_ids[vOffsetSubcell + 0];

    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    numBoundary = 0;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      // Extract pressure value at the boundary
      PetscInt p_bnd_idx;
      if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
      else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

      pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];
      numBoundary++;

    }

    PetscReal P[numBoundary+1];

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) P[icell] = pBoundary[icell];

    // Save internal pressure value
    gz = 0.0;
    ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
    P[numBoundary] = p[cell_id] + tdy->rho[cell_id]*gz;

    // Compute T*P
    PetscScalar TtimesP[numBoundary];
    for (irow=0; irow<numBoundary; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<numBoundary+1; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt sOffsetFace = subcells->face_offset[subcell_id];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      dukvr_dPup = 0.0;
      dukvr_dPdn = 0.0;

      if (TtimesP[irow] < 0.0) { // up ---> dn
        if (cell_id_up>=0) {
          ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
          dukvr_dPup = tdy->dKr_dS[cell_id_up]*tdy->dS_dP[cell_id_up]/tdy->vis[cell_id_up] -
                       tdy->Kr[cell_id_up]/(tdy->vis[cell_id_up]*tdy->vis[cell_id_up])*tdy->dvis_dP[cell_id_up];
        } else {
          ukvr = tdy->Kr_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
          dukvr_dPup = 0.0;//tdy->dKr_dS[-cell_id_up-1]*tdy->dS_dP_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
        }
      } else {
        if (cell_id_dn>=0) {
          ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
          dukvr_dPdn = tdy->dKr_dS[cell_id_dn]*tdy->dS_dP[cell_id_dn]/tdy->vis[cell_id_dn] -
                       tdy->Kr[cell_id_dn]/(tdy->vis[cell_id_dn]*tdy->vis[cell_id_dn])*tdy->dvis_dP[cell_id_dn];
        } else {
          ukvr = tdy->Kr_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
          dukvr_dPdn = 0.0;//tdy->dKr_dS_BND[-cell_id_dn-1]*tdy->dS_dP_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
        }
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      //fluxm = den*ukvr*(-TtimesP[irow]);

      // fluxm > 0 implies flow is from 'up' to 'dn'
      //if (cell_id_up>-1 && cells[cell_id_up].is_local) r[cell_id_up] += fluxm;
      //if (cell_id_dn>-1 && cells[cell_id_dn].is_local) r[cell_id_dn] -= fluxm;

      dden_dPup = 0.0;
      dden_dPdn = 0.0;

      // Deriviates will be computed only w.r.t. internal pressure
      //for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
      icol = numBoundary;
      if (cell_id_up>-1) cell_id = cell_id_up;
      else               cell_id = cell_id_dn;

      irow = iface;
      T = tdy->Trans[vertex_id][irow][icol];

      PetscReal dden_dP = 0.0;
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

      if (cell_id_up>-1 && cell_id == cell_id_up) {
        Jac =
          dden_dPup * ukvr       * TtimesP[irow] +
          den       * dukvr_dPup * TtimesP[irow] +
          den       * ukvr       * T * (1.0 + dden_dP*gz) ;
      } else if (cell_id_dn>-1 && cell_id == cell_id_dn) {
        Jac =
          dden_dPdn * ukvr       * TtimesP[irow] +
          den       * dukvr_dPdn * TtimesP[irow] +
          den       * ukvr       * T * (1.0 + dden_dP*gz) ;
      }
      if (fabs(Jac)<PETSC_MACHINE_EPSILON) Jac = 0.0;

      // Changing sign when bringing the term from RHS to LHS of the equation
      Jac = -Jac;

      if (cell_id_up >-1 && cells->is_local[cell_id_up]) {
        ierr = MatSetValuesLocal(A,1,&cell_id_up,1,&cell_id,&Jac,ADD_VALUES);CHKERRQ(ierr);
      }

      if (cell_id_dn >-1 && cells->is_local[cell_id_dn]) {
        Jac = -Jac;
        ierr = MatSetValuesLocal(A,1,&cell_id_dn,1,&cell_id,&Jac,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Accumulation_3DMesh(Vec Ul,Vec Udotl,PetscReal shift,Mat A,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  PetscInt icell;
  PetscReal *dp_dt;
  PetscErrorCode ierr;

  PetscReal dporosity_dP = 0.0, d2porosity_dP2 = 0.0;
  PetscReal drho_dP, d2rho_dP2;
  PetscReal dmass_dP, d2mass_dP2;

  PetscFunctionBegin;

  mesh = tdy->mesh;
  cells = &mesh->cells;

  ierr = VecGetArray(Udotl,&dp_dt); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    drho_dP = tdy->drho_dP[icell];
    d2rho_dP2 = tdy->d2rho_dP2[icell];

    // d(rho*phi*s)/dP * dP/dt * Vol
    dmass_dP = tdy->rho[icell] * dporosity_dP         * tdy->S[icell] +
               drho_dP         * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell] * tdy->porosity[icell] * tdy->dS_dP[icell];

    d2mass_dP2 = (
      tdy->dS_dP[icell]   * tdy->rho[icell]        * dporosity_dP         +
      tdy->S[icell]       * drho_dP                * dporosity_dP         +
      tdy->S[icell]       * tdy->rho[icell]        * d2porosity_dP2       +
      tdy->dS_dP[icell]   * drho_dP                * tdy->porosity[icell] +
      tdy->S[icell]       * d2rho_dP2              * tdy->porosity[icell] +
      tdy->S[icell]       * drho_dP                * dporosity_dP         +
      tdy->d2S_dP2[icell] * tdy->rho[icell]        * tdy->porosity[icell] +
      tdy->dS_dP[icell]   * drho_dP                * tdy->porosity[icell] +
      tdy->dS_dP[icell]   * tdy->rho[icell]        * dporosity_dP
       );

    PetscReal value = (shift*dmass_dP + d2mass_dP2*dp_dt[icell])*cells->volume[icell];

    ierr = MatSetValuesLocal(A,1,&icell,1,&icell,&value,ADD_VALUES);CHKERRQ(ierr);

  }

  ierr = VecRestoreArray(Udotl,&dp_dt); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  DM             dm;
  Vec Ul, Udotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm = tdy->dm;

  ierr = MatZeroEntries(A); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Udotl); CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U_t,INSERT_VALUES,Udotl); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U_t,INSERT_VALUES,Udotl); CHKERRQ(ierr);

  ierr = TDyMPFAOIJacobian_InternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh(Ul, A, ctx);
  //ierr = TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_Accumulation_3DMesh(Ul, Udotl, shift, A, ctx);

  if (A !=B ) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


