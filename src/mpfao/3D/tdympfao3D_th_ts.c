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
PetscErrorCode TDyMPFAOIFunction_InternalVertices_3DMesh_TH(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *r;
  PetscInt ivertex;
  PetscInt dim;
  PetscInt irow;
  PetscInt cell_id_up, cell_id_dn;
  PetscReal den,fluxm,ukvr,fluxe,flow_rate,uh;
  PetscScalar *TtimesP_vec_ptr, *Temp_TtimesP_vec_ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);


  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (vertices->num_boundary_cells[ivertex] != 0) continue;
    PetscInt vOffsetFace = vertices->face_offset[ivertex];

    PetscInt nflux_in = 12;
    PetscScalar TtimesP[nflux_in], Temp_TtimesP[nflux_in];

    // Compute = T*P
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      Temp_TtimesP[irow] = Temp_TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
      if (fabs(Temp_TtimesP[irow])<PETSC_MACHINE_EPSILON) Temp_TtimesP[irow] = 0.0;
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
      
      if (TtimesP[irow] < 0.0) { 
        ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
        uh   = tdy->h[cell_id_up];
      }
      else {
        ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
        uh   = tdy->h[cell_id_dn];
      }                    
      
      den = 0.5*(tdy->rho[cell_id_up] + tdy->rho[cell_id_dn]);
      flow_rate = ukvr*(-TtimesP[irow]); // flow_rate is darcy flux times area

      fluxm = den*flow_rate;
      fluxe = den*flow_rate*uh;  //Advection term

      // Conduction term in energy equation
      fluxe += -Temp_TtimesP[irow];
      
      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cells->is_local[cell_id_up]) {
        r[cell_id_up*2]   += fluxm;
        r[cell_id_up*2+1] += fluxe;
      }
      if (cells->is_local[cell_id_dn]) {
        r[cell_id_dn*2]   -= fluxm;
        r[cell_id_dn*2+1] -= fluxe;
      }
    }
  }

  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh_TH(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex;
  PetscInt dim;
  PetscInt ncells,ncells_bnd;
  PetscInt npitf_bc,nflux_bc,nflux_in;
  PetscInt irow;
  PetscInt cell_id_up,cell_id_dn;
  PetscReal den,fluxm,ukvr,fluxe,uh,flow_rate;
  PetscScalar *TtimesP_vec_ptr, *p_vec_ptr;
  PetscScalar *Temp_TtimesP_vec_ptr;
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
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->P_vec,&p_vec_ptr);CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells     <  2) continue;
    if (!vertices->is_local[ivertex]) continue;

    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

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

    // Compute T*P
    PetscScalar TtimesP[nflux_in + 2*nflux_bc];
    PetscScalar Temp_TtimesP[nflux_in + 2*nflux_bc];

    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      TtimesP[irow]     = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      Temp_TtimesP[irow] = Temp_TtimesP_vec_ptr[face_id*num_subfaces + subface_id];

      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
      if (fabs(Temp_TtimesP[irow])<PETSC_MACHINE_EPSILON) Temp_TtimesP[irow] = 0.0;
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

      if (TtimesP[irow] < 0.0) { // up ---> dn
        if (cell_id_up>=0) uh = tdy->h[cell_id_up];
        else               uh = tdy->h_BND[-cell_id_up-1];
      } else {
        if (cell_id_dn>=0) uh = tdy->h[cell_id_dn];
        else               uh = tdy->h_BND[-cell_id_dn-1];
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      flow_rate = ukvr*(-TtimesP[irow]);
      fluxm = den*flow_rate;
      fluxe = den*flow_rate*uh;  //Advection term

      fluxe += -Temp_TtimesP[irow];


      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {
        r[cell_id_up*2]   += fluxm;
        r[cell_id_up*2+1] += fluxe;
      }
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) {
        r[cell_id_dn*2]   -= fluxm;
        r[cell_id_dn*2+1] -= fluxe;
      }

    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh_TH(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex;
  PetscInt dim;
  TDy_subcell    *subcells;
  PetscInt irow;
  PetscInt isubcell,iface;
  PetscInt cell_id_up,cell_id_dn,cell_id;
  PetscReal den,fluxm,ukvr,fluxe,uh,flow_rate;
  PetscReal *TtimesP_vec_ptr,*Temp_TtimesP_vec_ptr;
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
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetFace = vertices->face_offset[ivertex];

    // Vertex is on the boundary
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    cell_id  = vertices->internal_cell_ids[vOffsetCell + 0];
    isubcell = vertices->subcell_ids[vOffsetSubcell + 0];

    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    numBoundary = subcells->num_faces[subcell_id];

    // Compute T*P
    PetscScalar TtimesP[numBoundary];
    PetscScalar Temp_TtimesP[numBoundary];
    for (irow=0; irow<numBoundary; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      TtimesP[irow]      = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      Temp_TtimesP[irow] = Temp_TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
      if (fabs(Temp_TtimesP[irow])<PETSC_MACHINE_EPSILON) Temp_TtimesP[irow] = 0.0;
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

      if (TtimesP[irow] < 0.0) { // up ---> dn
        if (cell_id_up>=0) uh = tdy->h[cell_id_up];
        else               uh = tdy->h[-cell_id_up-1];
      } else {
        if (cell_id_dn>=0) uh = tdy->h[cell_id_dn];
        else               uh = tdy->h[-cell_id_dn-1];
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      flow_rate = ukvr*(-TtimesP[irow]);
      fluxm = den*flow_rate;
      fluxe = den*flow_rate*uh;  //Advection term

      fluxe += -Temp_TtimesP[irow];


      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {
        r[cell_id_up*2]   += fluxm;
        r[cell_id_up*2+1] += fluxe;
      }
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) {
        r[cell_id_dn*2]   -= fluxm;
        r[cell_id_dn*2+1] -= fluxe;
      }
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_3DMesh_TH(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {
  
  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*du_dt,*r,*temp,*u_p,*dp_dt,*dtemp_dt;
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
  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy,u_p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyMPFAO_SetBoundaryTemperature(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(tdy->Trans_mat,tdy->P_vec,tdy->TtimesP_vec);
  ierr = MatMult(tdy->Temp_Trans_mat,tdy->Temp_P_vec,tdy->Temp_TtimesP_vec);

#if 0
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Temp_Trans_mat.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(tdy->Temp_Trans_mat,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Temp_TtimesP_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(tdy->Temp_TtimesP_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Temp_P_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(tdy->Temp_P_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  PetscInt iface;

  // Fluxes
  ierr = TDyMPFAOIFunction_InternalVertices_3DMesh_TH(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh_TH(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh_TH(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscInt c,cStart,cEnd;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(Vec),&dp_dt);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(Vec),&dtemp_dt);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(Vec),&p);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(Vec),&temp);CHKERRQ(ierr);

  for (c=0;c<cEnd-cStart;c++) {
    dp_dt[c]    = du_dt[c*2];
    p[c]        = u_p[c*2];
    dtemp_dt[c] = du_dt[c*2+1];
    temp[c]     = u_p[c*2+1];
  }


  PetscReal dporosity_dP = 0.0;
  PetscReal dporosity_dT = 0.0;
  PetscReal dmass_dP,dmass_dT;
  PetscInt icell;

  // Accumulation and source/sink contributions
  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) break;

    // A_M = d(rho*phi*s)/dP * dP_dtime * Vol + d(rho*phi*s)/dT * dT_dtime * Vol
    dmass_dP = tdy->rho[icell]     * dporosity_dP         * tdy->S[icell] +
               tdy->drho_dP[icell] * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dP[icell];
    dmass_dT = tdy->rho[icell]     * dporosity_dT         * tdy->S[icell] +
               tdy->drho_dT[icell] * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dT[icell];

    // A_E = [d(rho*phi*s*U)/dP + d(rho*(1-phi)*T)/dP] * dP_dtime *Vol + 
    //       [d(rho*phi*s*U)/dT + d(rho*(1-phi)*T)/dT] * dT_dtime *Vol
    // denergy_dP = dden_dP     * por        * sat     * u     + &
    //              den         * dpor_dP    * sat     * u     + &
    //              den         * por        * dsat_dP * u     + &
    //              den         * por        * sat     * du_dP + &
    //              rock_dencpr * (-dpor_dP) * temp

    // denergy_dt = dden_dt     * por        * sat     * u     + &
    //              den         * dpor_dt    * sat     * u     + &
    //              den         * por        * dsat_dt * u     + &
    //              den         * por        * sat     * du_dt + &
    //              rock_dencpr * (-dpor_dt) * temp            + &
    //              rock_dencpr * (1-por)

    PetscReal rock_dencpr = tdy->rhor[icell]*tdy->Cr[icell];
    PetscReal denergy_dP,denergy_dT;

    denergy_dP = tdy->drho_dP[icell] * tdy->porosity[icell] * tdy->S[icell]     * tdy->u[icell]     +
                 tdy->rho[icell]     * dporosity_dP         * tdy->S[icell]     * tdy->u[icell]     +
                 tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dP[icell] * tdy->u[icell]     +
                 tdy->rho[icell]     * tdy->porosity[icell] * tdy->S[icell]     * tdy->du_dP[icell] +
                 rock_dencpr         * (-dporosity_dP)      * temp[icell];

    denergy_dT = tdy->drho_dT[icell] * tdy->porosity[icell] * tdy->S[icell]     * tdy->u[icell]     +
                 tdy->rho[icell]     * dporosity_dT         * tdy->S[icell]     * tdy->u[icell]     +
                 tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dT[icell] * tdy->u[icell]     +
                 tdy->rho[icell]     * tdy->porosity[icell] * tdy->S[icell]     * tdy->du_dT[icell] +
                 rock_dencpr         * (-dporosity_dT)      * temp[icell]                           +
                 rock_dencpr         * (1.0 - tdy->porosity[icell]);

    r[icell*2]   += dmass_dP * dp_dt[icell] * cells->volume[icell] + dmass_dT * dtemp_dt[icell] * cells->volume[icell];
    r[icell*2+1] += denergy_dP * dp_dt[icell] * cells->volume[icell] + denergy_dT * dtemp_dt[icell] * cells->volume[icell];
    r[icell*2]   -= tdy->source_sink[icell] * cells->volume[icell];
    r[icell*2+1] -= tdy->energy_source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_InternalVertices_3DMesh_TH(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt ivertex, vertex_id;
  PetscInt cell_id, cell_id_up, cell_id_dn;
  PetscInt irow, icol;
  PetscInt dim;
  PetscReal gz;
  PetscReal *p;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal T;
  PetscScalar *TtimesP_vec_ptr;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] != 0) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    PetscInt nflux_in = 12;
    PetscScalar TtimesP[nflux_in];

    // Compute = T*P
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;
      
      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
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
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh_TH(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt ncells, ncells_bnd;
  PetscInt npitf_bc, nflux_bc, nflux_in;
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
  PetscScalar *TtimesP_vec_ptr;
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
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells     <  2) continue;
    if (!vertices->is_local[ivertex]) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

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

        if (faces->is_internal[face_id] == 0) {

          numBoundary++;
        }
      }
    }
    
    // Compute T*P
    PetscScalar TtimesP[nflux_in + 2*nflux_bc];
    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];

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
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh_TH(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt fStart, fEnd;
  TDy_subcell    *subcells;
  PetscInt dim;
  PetscInt ivertex;
  PetscInt isubcell, iface;
  PetscInt cell_id, cell_id_up, cell_id_dn, vertex_id;
  PetscInt irow, icol;
  PetscReal T;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal *p;
  PetscReal gz;
  PetscScalar *TtimesP_vec_ptr;
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
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    // Vertex is on the boundary
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    cell_id  = vertices->internal_cell_ids[vOffsetCell + 0];
    isubcell = vertices->subcell_ids[vOffsetSubcell + 0];

    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    numBoundary = subcells->num_faces[subcell_id];

    // Compute T*P
    PetscScalar TtimesP[numBoundary];
    for (irow=0; irow<numBoundary; irow++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
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
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Accumulation_3DMesh_TH(Vec Ul,Vec Udotl,PetscReal shift,Mat A,void *ctx) {

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
PetscErrorCode TDyMPFAOIJacobian_3DMesh_TH(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

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

  ierr = TDyMPFAOIJacobian_InternalVertices_3DMesh_TH(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh_TH(Ul, A, ctx);
  //ierr = TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_Accumulation_3DMesh_TH(Ul, Udotl, shift, A, ctx);

  if (A !=B ) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


