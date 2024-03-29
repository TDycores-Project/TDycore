#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdydiscretizationimpl.h>

//#define DEBUG
#if defined(DEBUG)
PetscInt icount_f = 0;
PetscInt icount_j = 0;
PetscInt max_count = 5;
#endif

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Vertices_TH(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  PetscReal *r;
  PetscInt ivertex;
  PetscInt dim;
  PetscInt irow;
  PetscInt cell_id_up, cell_id_dn;
  PetscInt npitf_bc, nflux_in;
  PetscReal den,fluxm,ukvr,fluxe,flow_rate,uh;
  PetscScalar *TtimesP_vec_ptr, *Temp_TtimesP_vec_ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    PetscInt *face_ids, num_faces;
    PetscInt *subface_ids, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

    npitf_bc = vertices->num_boundary_faces[ivertex];
    nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    PetscScalar TtimesP[nflux_in + npitf_bc], Temp_TtimesP[nflux_in + npitf_bc];

    // Compute = T*P
    for (irow=0; irow<nflux_in + npitf_bc; irow++) {

      PetscInt face_id = face_ids[irow];
      PetscInt subface_id = subface_ids[irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      Temp_TtimesP[irow] = Temp_TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
      if (fabs(Temp_TtimesP[irow])<PETSC_MACHINE_EPSILON) Temp_TtimesP[irow] = 0.0;


    //
    // fluxm_ij = rho_ij * (kr/mu)_{ij,upwind} * [ T ] *  [ P+rho*g*z ]^T
    // where
    //      rho_ij = 0.5*(rho_i + rho_j)
    //      (kr/mu)_{ij,upwind} = (kr/mu)_{i} if velocity is from i to j
    //                          = (kr/mu)_{j} otherwise
    //      T includes product of K and A_{ij}
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];

      if (TtimesP[irow] < 0.0) { // up ---> dn
         // Is the cell_id_up an internal or boundary cell?
        if (cell_id_up >= 0) {
          ukvr = mpfao->Kr[cell_id_up]/mpfao->vis[cell_id_up];
          uh   = mpfao->h[cell_id_up];
        }
        else {
          ukvr = mpfao->Kr_bnd[-cell_id_up-1]/mpfao->vis_bnd[-cell_id_up-1];
          uh   = mpfao->h_bnd[-cell_id_up-1];
        }
      }
      else {
         // Is the cell_id_dn an internal or boundary cell?
         if (cell_id_dn >= 0) {
          ukvr = mpfao->Kr[cell_id_dn]/mpfao->vis[cell_id_dn];
          uh   = mpfao->h[cell_id_dn];
         }
         else {
          ukvr = mpfao->Kr_bnd[-cell_id_dn-1]/mpfao->vis_bnd[-cell_id_dn-1];
          uh   = mpfao->h_bnd[-cell_id_dn-1];
         }
      }

      den = 0.0;
      if (cell_id_up>=0) den += mpfao->rho[cell_id_up];
      else               den += mpfao->rho_bnd[-cell_id_up-1];
      if (cell_id_dn>=0) den += mpfao->rho[cell_id_dn];
      else               den += mpfao->rho_bnd[-cell_id_dn-1];
      den *= 0.5;
      flow_rate = ukvr*(-TtimesP[irow]); // flow_rate is darcy flux times area

      fluxm = den*flow_rate;
      fluxe = den*flow_rate*uh;  //Advection term

      // Conduction term in energy equation
      fluxe += -Temp_TtimesP[irow];

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
        r[cell_id_up*2]   += fluxm;
        r[cell_id_up*2+1] += fluxe;
      }
      if (cell_id_dn >=0 && cells->is_local[cell_id_dn]) {
        r[cell_id_dn*2]   -= fluxm;
        r[cell_id_dn*2+1] -= fluxe;
      }
    }
  }

  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_TH(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {

  TDy        tdy = (TDy)ctx;
  TDyMPFAO  *mpfao = tdy->context;
  TDyMesh   *mesh = mpfao->mesh;
  TDyCell   *cells = &mesh->cells;
  PetscReal *du_dt,*r,*u_p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

#if defined(DEBUG)
  PetscViewer viewer;
  char word[32];
  sprintf(word,"U%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

#if defined(DEBUG)
  sprintf(word,"Ul%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = VecView(Ul,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  ierr = TDyGlobalToLocal(tdy,U,tdy->soln_loc); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(tdy->soln_loc,&u_p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy,u_p,mesh->num_cells); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_loc,&u_p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyMPFAO_SetBoundaryTemperature(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyMPFAOUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(mpfao->Trans_mat,mpfao->P_vec,mpfao->TtimesP_vec);
  ierr = MatMult(mpfao->Temp_Trans_mat,mpfao->Temp_P_vec,mpfao->Temp_TtimesP_vec);

#if 0
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Temp_Trans_mat.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(mpfao->Temp_Trans_mat,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Temp_TtimesP_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(mpfao->Temp_TtimesP_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Temp_P_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(mpfao->Temp_P_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  // Fluxes
  ierr = TDyMPFAOIFunction_Vertices_TH(tdy->soln_loc,R,ctx); CHKERRQ(ierr);


  ierr = VecGetArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscInt c,cStart,cEnd;
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  PetscReal dp_dt[cEnd-cStart],
            temp[cEnd-cStart], dtemp_dt[cEnd-cStart];
  ierr = VecGetArray(tdy->soln_loc,&u_p); CHKERRQ(ierr);
  for (c=0;c<cEnd-cStart;c++) {
    dp_dt[c]    = du_dt[c*2];
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
    dmass_dP = mpfao->rho[icell]     * dporosity_dP         * mpfao->S[icell] +
               mpfao->drho_dP[icell] * mpfao->porosity[icell] * mpfao->S[icell] +
               mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->dS_dP[icell];
    dmass_dT = mpfao->rho[icell]     * dporosity_dT         * mpfao->S[icell] +
               mpfao->drho_dT[icell] * mpfao->porosity[icell] * mpfao->S[icell] +
               mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->dS_dT[icell];

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

    PetscReal rock_dencpr = mpfao->rho_soil[icell]*mpfao->c_soil[icell];
    PetscReal denergy_dP,denergy_dT;

    denergy_dP = mpfao->drho_dP[icell] * mpfao->porosity[icell] * mpfao->S[icell]     * mpfao->u[icell]     +
                 mpfao->rho[icell]     * dporosity_dP         * mpfao->S[icell]     * mpfao->u[icell]     +
                 mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->dS_dP[icell] * mpfao->u[icell]     +
                 mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->S[icell]     * mpfao->du_dP[icell] +
                 rock_dencpr         * (-dporosity_dP)      * temp[icell];

    denergy_dT = mpfao->drho_dT[icell] * mpfao->porosity[icell] * mpfao->S[icell]     * mpfao->u[icell]     +
                 mpfao->rho[icell]     * dporosity_dT         * mpfao->S[icell]     * mpfao->u[icell]     +
                 mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->dS_dT[icell] * mpfao->u[icell]     +
                 mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->S[icell]     * mpfao->du_dT[icell] +
                 rock_dencpr         * (-dporosity_dT)      * temp[icell]                           +
                 rock_dencpr         * (1.0 - mpfao->porosity[icell]);

    r[icell*2]   += dmass_dP * dp_dt[icell] * cells->volume[icell] + dmass_dT * dtemp_dt[icell] * cells->volume[icell];
    r[icell*2+1] += denergy_dP * dp_dt[icell] * cells->volume[icell] + denergy_dT * dtemp_dt[icell] * cells->volume[icell];
    r[icell*2]   -= mpfao->source_sink[icell] * cells->volume[icell];
    r[icell*2+1] -= mpfao->energy_source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(tdy->soln_loc,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

#if defined(DEBUG)
  sprintf(word,"Function%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  icount_f++;
#endif

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Vertices_TH(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  PetscInt ivertex, vertex_id;
  PetscInt npitf_bc, nflux_in;
  PetscInt cell_id, cell_id_up, cell_id_dn;
  PetscInt irow, icol;
  PetscInt dim;
  PetscReal gz;
  PetscReal ukvr, den, uh;
  PetscReal dukvr_dPup, dukvr_dPdn;
  PetscReal duh_dPup, duh_dPdn;
  PetscReal dukvr_dTup, dukvr_dTdn;
  PetscReal duh_dTup, duh_dTdn;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal dden_dTup, dden_dTdn;
  PetscReal T, Temp_T;
  PetscScalar *TtimesP_vec_ptr, *Temp_TtimesP_vec_ptr;
  PetscScalar Jac[4];
  PetscInt size_jac = 4;
  PetscInt ijac;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_faces[ivertex] > 0 && vertices->num_internal_cells[ivertex] < 2) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];

    PetscInt *face_ids, num_faces;
    PetscInt *subface_ids, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

    npitf_bc = vertices->num_boundary_faces[ivertex];
    nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];


    PetscScalar TtimesP[nflux_in + npitf_bc], Temp_TtimesP[nflux_in + npitf_bc];

    // Compute = T*P
    for (irow=0; irow<nflux_in + npitf_bc; irow++) {

      PetscInt face_id = face_ids[irow];
      PetscInt subface_id = subface_ids[irow];
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
    //
    // For i and j, jacobian is given as:
    // d(fluxm_ij)/dP_i = d(rho_ij)/dP_i + (kr/mu)_{ij,upwind}         * [  T  ] *  [    P+rho*g*z   ]^T +
    //                      rho_ij       + d((kr/mu)_{ij,upwind})/dP_i * [  T  ] *  [    P+rho*g*z   ]^T +
    //                      rho_ij       + (kr/mu)_{ij,upwind}         *   T_i   *  (1+d(rho_ij)/dP_i*g*z
    //
    // For k not equal to i and j, jacobian is given as:
    // d(fluxm_ij)/dP_k =   rho_ij       + (kr/mu)_{ij,upwind}         *   T_ik  *  (1+d(rho_ij)/dP_k*g*z
    //
    for (irow=0; irow<nflux_in + npitf_bc; irow++) {

      PetscInt face_id = face_ids[irow];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];

      dukvr_dPup = 0.0;
      dukvr_dPdn = 0.0;
      dukvr_dTup = 0.0;
      dukvr_dTdn = 0.0;
      duh_dPup = 0.0;
      duh_dPdn = 0.0;
      duh_dTup = 0.0;
      duh_dTdn = 0.0;

      if (TtimesP[irow] < 0.0) {
        // Flow: up --> dn
        if (cell_id_up>=0) {
          // "up" is an internal cell
          ukvr       = mpfao->Kr[cell_id_up]/mpfao->vis[cell_id_up];
          dukvr_dPup = mpfao->dKr_dS[cell_id_up]*mpfao->dS_dP[cell_id_up]/mpfao->vis[cell_id_up] -
                      mpfao->Kr[cell_id_up]/(mpfao->vis[cell_id_up]*mpfao->vis[cell_id_up])*mpfao->dvis_dP[cell_id_up];
          dukvr_dTup = mpfao->dKr_dS[cell_id_up]*mpfao->dS_dT[cell_id_up]/mpfao->vis[cell_id_up] -
                      mpfao->Kr[cell_id_up]/(mpfao->vis[cell_id_up]*mpfao->vis[cell_id_up])*mpfao->dvis_dT[cell_id_up];
          uh         = mpfao->h[cell_id_up];
          duh_dPup   = mpfao->dh_dP[cell_id_up];
          duh_dTup   = mpfao->dh_dT[cell_id_up];
        } else {
           // "up" is boundary cell
          ukvr       = mpfao->Kr_bnd[-cell_id_up-1]/mpfao->vis_bnd[-cell_id_up-1];
          dukvr_dPup = 0.0;
          dukvr_dTup = 0.0;
          uh         = mpfao->h_bnd[-cell_id_up-1];
          duh_dPup   = 0.0;
          duh_dTup   = 0.0;
        }

      } else {
        // Flow: up <--- dn
        if (cell_id_dn >= 0) {
          ukvr       = mpfao->Kr[cell_id_dn]/mpfao->vis[cell_id_dn];
            dukvr_dPdn = mpfao->dKr_dS[cell_id_dn]*mpfao->dS_dP[cell_id_dn]/mpfao->vis[cell_id_dn] -
                        mpfao->Kr[cell_id_dn]/(mpfao->vis[cell_id_dn]*mpfao->vis[cell_id_dn])*mpfao->dvis_dP[cell_id_dn];
            dukvr_dTdn = mpfao->dKr_dS[cell_id_dn]*mpfao->dS_dT[cell_id_dn]/mpfao->vis[cell_id_dn] -
                        mpfao->Kr[cell_id_dn]/(mpfao->vis[cell_id_dn]*mpfao->vis[cell_id_dn])*mpfao->dvis_dT[cell_id_dn];
            uh         = mpfao->h[cell_id_dn];
            duh_dPdn   = mpfao->dh_dP[cell_id_dn];
            duh_dTdn   = mpfao->dh_dT[cell_id_dn];
        } else {
          // "dn" is a boundary cell
          ukvr       = mpfao->Kr_bnd[-cell_id_dn-1]/mpfao->vis_bnd[-cell_id_dn-1];
          dukvr_dPdn = 0.0;
          dukvr_dTdn = 0.0;
          uh         = mpfao->h_bnd[-cell_id_dn-1];
          duh_dPdn   = 0.0;
          duh_dTdn   = 0.0;
        }
      }

      den = 0.0;
      if (cell_id_up>=0) den += mpfao->rho[cell_id_up];
      else               den += mpfao->rho_bnd[-cell_id_up-1];
      if (cell_id_dn>=0) den += mpfao->rho[cell_id_dn];
      else               den += mpfao->rho_bnd[-cell_id_dn-1];
      den *= 0.5;

      // If one of the cell is on the boundary
      if (cell_id_up<0) cell_id_up = cell_id_dn;
      if (cell_id_dn<0) cell_id_dn = cell_id_up;

      dden_dPup = 0.0;
      dden_dPdn = 0.0;
      dden_dTup = 0.0;
      dden_dTdn = 0.0;

      if (cell_id_up >= 0) dden_dPup = 0.5*mpfao->drho_dP[cell_id_up];
      if (cell_id_dn >= 0) dden_dPdn = 0.5*mpfao->drho_dP[cell_id_dn];
      if (cell_id_up >= 0) dden_dTup = 0.5*mpfao->drho_dT[cell_id_up];
      if (cell_id_dn >= 0) dden_dTdn = 0.5*mpfao->drho_dT[cell_id_dn];


      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        cell_id = vertices->internal_cell_ids[vOffsetCell + icol];

        T      = mpfao->Trans[vertex_id][irow][icol];
        Temp_T = mpfao->Temp_Trans[vertex_id][irow][icol];

        ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

      // Jac[0] is dfluxm/dP
      // Jac[1] is dfluxm/dT
      // Jac[2] is dfluxe/dP
      // Jac[3] is dfluxe/dT

	for (ijac=0; ijac<size_jac; ijac++) Jac[ijac] = 0.0;

      // Advection terms
        if (cell_id_up > -1 && cell_id == cell_id_up) {
          Jac[0] =
            dden_dPup * ukvr       * TtimesP[irow] +
            den       * dukvr_dPup * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dPup*gz) ;
          Jac[2] =
            dden_dTup * ukvr       * TtimesP[irow] +
            den       * dukvr_dTup * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dTup*gz) ;
          Jac[1] =
            dden_dPup * ukvr       * TtimesP[irow]            * uh +
            den       * dukvr_dPup * TtimesP[irow]            * uh +
            den       * ukvr       * T * (1.0 + dden_dPup*gz) * uh +
            den       * ukvr       * TtimesP[irow]            * duh_dPup;
          Jac[3] =
            dden_dTup * ukvr       * TtimesP[irow]            * uh +
            den       * dukvr_dTup * TtimesP[irow]            * uh +
            den       * ukvr       * T * (1.0 + dden_dTup*gz) * uh +
            den       * ukvr       * TtimesP[irow]            * duh_dTup;
        } else if (cell_id_dn > -1 && cell_id == cell_id_dn) {
          Jac[0] =
            dden_dPdn * ukvr       * TtimesP[irow] +
            den       * dukvr_dPdn * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dPdn*gz) ;
          Jac[2] =
            dden_dTdn * ukvr       * TtimesP[irow] +
            den       * dukvr_dTdn * TtimesP[irow] +
            den       * ukvr       * T * (1.0 + dden_dTdn*gz) ;
          Jac[1] =
            dden_dPdn * ukvr       * TtimesP[irow]            * uh +
            den       * dukvr_dPdn * TtimesP[irow]            * uh +
            den       * ukvr       * T * (1.0 + dden_dPdn*gz) * uh +
            den       * ukvr       * TtimesP[irow]            * duh_dPdn;
          Jac[3] =
            dden_dTdn * ukvr       * TtimesP[irow]            * uh +
            den       * dukvr_dTdn * TtimesP[irow]            * uh +
            den       * ukvr       * T * (1.0 + dden_dTdn*gz) * uh +
            den       * ukvr       * TtimesP[irow]            * duh_dTdn;
        } else {
          Jac[0] = den * ukvr * T * (1.0 + 0.*gz);
          Jac[2] = den * ukvr * T * (0.0 + 0.*gz); // derivative with temperature is zero
          Jac[1] = den * ukvr * T * (1.0 + 0.*gz) * uh;
          Jac[3] = den * ukvr * T * (0.0 + 0.*gz) * uh; // derivative with temperature is zero
        }

      // conduction term
        Jac[3] += Temp_T;


        for (ijac=0; ijac<size_jac; ijac++) {
          if (fabs(Jac[ijac])<PETSC_MACHINE_EPSILON) Jac[ijac] = 0.0;;
        }


        // Changing sign when bringing the term from RHS to LHS of the equation
        for (ijac=0; ijac<size_jac; ijac++) {Jac[ijac] = -Jac[ijac];}

        if (cells->is_local[cell_id_up]) {
          ierr = MatSetValuesBlockedLocal(A,1,&cell_id_up,1,&cell_id,Jac,ADD_VALUES);CHKERRQ(ierr);
        }

        if (cells->is_local[cell_id_dn]) {
          for (ijac=0; ijac<size_jac; ijac++) {Jac[ijac] = -Jac[ijac];}
          ierr = MatSetValuesBlockedLocal(A,1,&cell_id_dn,1,&cell_id,Jac,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecRestoreArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->Temp_TtimesP_vec,&Temp_TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

#if 0
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"jac_flux.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(A,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Accumulation_TH(Vec Ul,Vec Udotl,PetscReal shift,Mat A,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscInt icell;
  PetscReal *du_dt, *u_p;
  PetscErrorCode ierr;

  PetscReal porosity, dporosity_dP, dporosity_dT;
  PetscReal d2porosity_dP2, d2porosity_dT2, d2porosity_dTdP, d2porosity_dPdT;
  PetscReal rho, drho_dP, drho_dT;
  PetscReal d2rho_dP2, d2rho_dT2, d2rho_dTdP, d2rho_dPdT;
  PetscReal sat, dsat_dP, dsat_dT;
  PetscReal d2sat_dP2, d2sat_dT2, d2sat_dTdP, d2sat_dPdT;
  PetscReal u, du_dP, du_dT;
  PetscReal d2u_dP2, d2u_dT2, d2u_dTdP, d2u_dPdT;
  PetscReal dmass_dP, d2mass_dP2, d2mass_dPdT;
  PetscReal dmass_dT, d2mass_dT2, d2mass_dTdP;
  PetscReal denergy_dP, d2energy_dP2, d2energy_dPdT;
  PetscReal denergy_dT, d2energy_dT2, d2energy_dTdP;
  PetscScalar Jac[4];
  PetscInt size_jac = 4;
  PetscInt ijac;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  cells = &mesh->cells;

  ierr = VecGetArray(Udotl,&du_dt); CHKERRQ(ierr);
  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);

  PetscInt c,cStart,cEnd;
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  PetscReal dp_dt[cEnd-cStart], dT_dt[cEnd-cStart], temp[cEnd-cStart];
  for (c=0;c<cEnd-cStart;c++) {
    dp_dt[c] = du_dt[c*2];
    dT_dt[c] = du_dt[c*2+1];
    temp[c]  = u_p[c*2+1];
  }

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    for (ijac=0; ijac<size_jac; ijac++) {Jac[ijac] = 0.0;}

    // Porosity
    porosity = mpfao->porosity[icell];
    dporosity_dP = 0.0;
    d2porosity_dP2 = 0.0;
    dporosity_dT = 0.0;
    d2porosity_dT2 = 0.0;
    d2porosity_dTdP = 0.0;
    d2porosity_dPdT = d2porosity_dTdP;

    // Density
    rho = mpfao->rho[icell];
    drho_dP = mpfao->drho_dP[icell];
    drho_dT = mpfao->drho_dT[icell];
    d2rho_dP2 = mpfao->d2rho_dP2[icell];
    d2rho_dT2 = 0.0;
    d2rho_dTdP = 0.0;
    d2rho_dPdT = d2rho_dTdP;

    // Saturation
    sat = mpfao->S[icell];
    dsat_dP = mpfao->dS_dP[icell];
    dsat_dT = mpfao->dS_dT[icell];
    d2sat_dP2 = mpfao->d2S_dP2[icell];
    d2sat_dT2 = 0.0;
    d2sat_dTdP = 0.0;
    d2sat_dPdT = d2sat_dTdP;

    // Internal Energy
    u = mpfao->u[icell];
    du_dP = mpfao->du_dP[icell];
    du_dT = mpfao->du_dT[icell];
    d2u_dT2 = 0.0;
    d2u_dP2 = 0.0;
    d2u_dTdP = 0.0;
    d2u_dPdT = d2u_dTdP;

    PetscReal rock_dencpr = mpfao->rho_soil[icell]*mpfao->c_soil[icell];


    // A_M = d(rho*phi*s)/dP * dP_dtime * Vol + d(rho*phi*s)/dT * dT_dtime * Vol

    // Jlocal(1,1) = shift*d(A_M)/d(Pdot) + d(A_M)/d(P)
    //             = shift*d(rho*phi*s)/dP*Vol + d2(rho*phi*s)/dP2*dP_dtime*Vol +
    //               d2(rho*phi*s)/dTdP*dT_dtime*Vol

    dmass_dP = rho     * dporosity_dP * sat       +
               drho_dP * porosity     * sat       +
               rho     * porosity     * dsat_dP;

    d2mass_dP2 = (
      dsat_dP   * rho        * dporosity_dP    +
      sat       * drho_dP    * dporosity_dP    +
      sat       * rho        * d2porosity_dP2  +
      dsat_dP   * drho_dP    * porosity        +
      sat       * d2rho_dP2  * porosity        +
      sat       * drho_dP    * dporosity_dP    +
      d2sat_dP2 * rho        * porosity        +
      dsat_dP   * drho_dP    * porosity        +
      dsat_dP   * rho        * dporosity_dP
       );

    d2mass_dPdT = (
      dsat_dT     * drho_dP     * porosity        +
      sat         * d2rho_dTdP  * porosity        +
      sat         * drho_dP     * dporosity_dT    +
      d2sat_dTdP  * rho         * porosity        +
      dsat_dP     * drho_dT     * porosity        +
      dsat_dP     * rho         * dporosity_dT    +
      dsat_dT     * rho         * dporosity_dP    +
      sat         * drho_dT     * dporosity_dP    +
      sat         * rho         * d2porosity_dTdP
      );

    d2mass_dTdP = d2mass_dPdT;

    Jac[0] = (shift*dmass_dP + d2mass_dP2*dp_dt[icell] + d2mass_dTdP*dT_dt[icell])*cells->volume[icell];

    // Jlocal(1,2) = shift*d(A_M)/d(Tdot) + d(A_M)/d(T)
    //             = shift*d(rho*phi*s)/dT*Vol + d2(rho*phi*s)/dT2*dT_dtime*Vol +
    //              d2(rho*phi*s)/dPdT*dP_dtime*Vol

    dmass_dT = (
      sat     * drho_dT * porosity     +
      dsat_dT * rho     * porosity     +
      sat     * rho     * dporosity_dT
      );

    d2mass_dT2 = (
      dsat_dT   * drho_dT   * porosity       +
      sat       * d2rho_dT2 * porosity       +
      sat       * drho_dT   * dporosity_dT   +
      d2sat_dT2 * rho       * porosity       +
      dsat_dT   * drho_dT   * porosity       +
      dsat_dT   * rho       * dporosity_dT   +
      dsat_dT   * rho       * dporosity_dT   +
      sat       * drho_dT   * dporosity_dT   +
      sat       * rho       * d2porosity_dT2
      );

    Jac[2] = (shift*dmass_dT + d2mass_dT2*dT_dt[icell] + d2mass_dPdT*dp_dt[icell])*cells->volume[icell];

  //   A_E = [d(rho*phi*s*U)/dP + d(rock_dencpr*(1-phi)*T)/dP] * dP_dtime *Vol +
  //        [d(rho*phi*s*U)/dT + d(rock_dencpr*(1-phi)*T)/dT] * dT_dtime *Vol


  //  Jlocal(2,1) = shift*d(A_E)/d(Pdot) + d(A_E)/d(P)
  //              = shift*[d(rho*phi*s*U)/dP + d(rock_dencpr*(1-phi)*T)/dP]*Vol +
  //                [d2(rho*phi*s*U)/dP2 + d2(rock_dencpr*(1-phi)*T)/dP2]*dP_dtime*Vol +
  //                [d2(rho*phi*s*U)/dTdP + d2(rock_dencpr*(1-phi)*T)/dTdP]*dT_dtime*Vol

    denergy_dP = drho_dP     * porosity        * sat     * u     +
                 rho         * dporosity_dP    * sat     * u     +
                 rho         * porosity        * dsat_dP * u     +
                 rho         * porosity        * sat     * du_dP +
                 rock_dencpr * (-dporosity_dP) * temp[icell];

    d2energy_dP2 = d2rho_dP2   * porosity          * sat       * u       +
                   drho_dP     * dporosity_dP      * sat       * u       +
                   drho_dP     * porosity          * dsat_dP   * u       +
                   drho_dP     * porosity          * sat       * du_dP   +
                   drho_dP     * dporosity_dP      * sat       * u       +
                   rho         * d2porosity_dP2    * sat       * u       +
                   rho         * dporosity_dP      * dsat_dP   * u       +
                   rho         * dporosity_dP      * sat       * du_dP   +
                   drho_dP     * porosity          * dsat_dP   * u       +
                   rho         * dporosity_dP      * dsat_dP   * u       +
                   rho         * porosity          * d2sat_dP2 * u       +
                   rho         * porosity          * dsat_dP   * du_dP   +
                   drho_dP     * porosity          * sat       * du_dP   +
                   rho         * dporosity_dP      * sat       * du_dP   +
                   rho         * porosity          * dsat_dP   * du_dP   +
                   rho         * porosity          * sat       * d2u_dP2 +
                   rock_dencpr * (-d2porosity_dP2) * temp[icell];

    d2energy_dPdT = d2rho_dPdT  * porosity           * sat        * u        +
                    drho_dP     * dporosity_dT       * sat        * u        +
                    drho_dP     * porosity           * dsat_dT    * u        +
                    drho_dP     * porosity           * sat        * du_dT    +
                    drho_dT     * dporosity_dP       * sat        * u        +
                    rho         * d2porosity_dPdT    * sat        * u        +
                    rho         * dporosity_dP       * dsat_dT    * u        +
                    rho         * dporosity_dP       * sat        * du_dT    +
                    drho_dT     * porosity           * dsat_dP    * u        +
                    rho         * dporosity_dT       * dsat_dP    * u        +
                    rho         * porosity           * d2sat_dPdT * u        +
                    rho         * porosity           * dsat_dP    * du_dT    +
                    drho_dT     * porosity           * sat        * du_dP    +
                    rho         * dporosity_dT       * sat        * du_dP    +
                    rho         * porosity           * dsat_dT    * du_dP    +
                    rho         * porosity           * sat        * d2u_dPdT +
                    rock_dencpr * (-d2porosity_dPdT) * temp[icell]           +
                    rock_dencpr * (-dporosity_dP);

    d2energy_dTdP = d2energy_dPdT;


    Jac[1] = (shift*denergy_dP + d2energy_dP2*dp_dt[icell] + d2energy_dTdP*dT_dt[icell])*cells->volume[icell];


  //  Jlocal(2,2) = shift*d(A_E)/d(Tdot) + d(A_E)/d(T)
  //              = shift*[d(rho*phi*s*U)/dT + d(rock_dencpr*(1-phi)*T)/dT]*Vol +
  //                [d2(rho*phi*s*U)/dPdt + d2(rock_dencpr*(1-phi)*T)/dPdt]*dP_dtime*Vol +
  //                [d2(rho*phi*s*U)/dT2 + d2(rock_dencpr*(1-phi)*T)/dT2]*dT_dtime*Vol

    denergy_dT = drho_dT     * porosity        * sat     * u     +
                 rho         * dporosity_dT    * sat     * u     +
                 rho         * porosity        * dsat_dT * u     +
                 rho         * porosity        * sat     * du_dT +
                 rock_dencpr * (-dporosity_dT) * temp[icell]     +
                 rock_dencpr * (1-porosity);

    d2energy_dT2 = d2rho_dT2   * porosity          * sat       * u       +
                   drho_dT     * dporosity_dT      * sat       * u       +
                   drho_dT     * porosity          * dsat_dT   * u       +
                   drho_dT     * porosity          * sat       * du_dT   +
                   drho_dT     * dporosity_dT      * sat       * u       +
                   rho         * d2porosity_dT2    * sat       * u       +
                   rho         * dporosity_dT      * dsat_dT   * u       +
                   rho         * dporosity_dT      * sat       * du_dT   +
                   drho_dT     * porosity          * dsat_dT   * u       +
                   rho         * dporosity_dT      * dsat_dT   * u       +
                   rho         * porosity          * d2sat_dT2 * u       +
                   rho         * porosity          * dsat_dT   * du_dT   +
                   drho_dT     * porosity          * sat       * du_dT   +
                   rho         * dporosity_dT      * sat       * du_dT   +
                   rho         * porosity          * dsat_dT   * du_dT   +
                   rho         * porosity          * sat       * d2u_dT2 +
                   rock_dencpr * (-d2porosity_dT2) * temp[icell]         +
                   rock_dencpr * (-dporosity_dT)                         +
                   rock_dencpr * (-dporosity_dT);


    Jac[3] = (shift*denergy_dT + d2energy_dT2*dT_dt[icell] + d2energy_dPdT*dp_dt[icell])*cells->volume[icell];

    ierr = MatSetValuesBlockedLocal(A,1,&icell,1,&icell,Jac,ADD_VALUES);CHKERRQ(ierr);

  }

  ierr = VecRestoreArray(Udotl,&du_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

#if 0
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"jac_accum.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(A,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_TH(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  Vec Udotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);

  ierr = MatZeroEntries(B); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Udotl); CHKERRQ(ierr);

  ierr = TDyGlobalToLocal(tdy,U,tdy->soln_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,U_t,INSERT_VALUES,Udotl); CHKERRQ(ierr);

  ierr = TDyMPFAOIJacobian_Vertices_TH(tdy->soln_loc,B,ctx);
  ierr = TDyMPFAOIJacobian_Accumulation_TH(tdy->soln_loc,Udotl,shift,B,ctx);

  if (A !=B ) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

#if defined(DEBUG)
  PetscViewer viewer;
  char word[32];
  sprintf(word,"Jacobian%d.mat",icount_j);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = MatView(A,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  icount_j++;
  if (icount_j == max_count) exit(0);
#endif

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}


