#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdydiscretization.h>
#include <private/tdycharacteristiccurvesimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Vertices(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;

  PetscReal *r_ptr;
  PetscScalar *TtimesP_vec_ptr;
  PetscScalar *GravDis_ptr;
  PetscErrorCode ierr;

  PetscBool set_flow_to_zero = PETSC_FALSE;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = VecGetArray(R,&r_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    PetscInt *face_ids, *subface_ids;
    PetscInt num_faces, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

    PetscInt npitf_bc = vertices->num_boundary_faces[ivertex];
    PetscInt nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    // Compute = T*P
    PetscScalar TtimesP[nflux_in + npitf_bc];

    for (PetscInt irow=0; irow<nflux_in + npitf_bc; irow++) {
      set_flow_to_zero = PETSC_FALSE;
      PetscInt face_id = face_ids[irow];
      PetscInt subface_id = subface_ids[irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];

       //
       // fluxm_ij = rho_ij * (kr/mu)_{ij,upwind} * [ T ] *  [ P+rho*g*z ]^T
       // where
       //      rho_ij = 0.5*(rho_i + rho_j)
       //      (kr/mu)_{ij,upwind} = (kr/mu)_{i} if velocity is from i to j
       //                          = (kr/mu)_{j} otherwise
       //      T includes product of K and A_{ij}

      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);
      PetscInt cell_id_up = cell_ids[0];
      PetscInt cell_id_dn = cell_ids[1];

      PetscReal den_aveg = 0.0;
      if (cell_id_up>=0) {
        den_aveg += mpfao->rho[cell_id_up];
      } else {
        den_aveg += mpfao->rho_bnd[-cell_id_up-1];
      }

      if (cell_id_dn>=0) {
        den_aveg += mpfao->rho[cell_id_dn];
      } else {
        den_aveg += mpfao->rho_bnd[-cell_id_dn-1];
      }

      den_aveg *= 0.5;

      PetscReal G = GravDis_ptr[face_id*num_subfaces + subface_id];

      // Upwind the 'ukvr'
      PetscReal ukvr = 0.0;
      if (TtimesP[irow] + den_aveg * G < 0.0) { // up ---> dn
        // Is the cell_id_up an internal or boundary cell?
        if (cell_id_up>=0) {
          PetscReal Kr = mpfao->Kr[cell_id_up];
          PetscReal vis = mpfao->vis[cell_id_up];

          ukvr = Kr/vis;
        } else {
          PetscReal Kr = mpfao->Kr_bnd[-cell_id_up-1];
          PetscReal vis = mpfao->vis_bnd[-cell_id_up-1];

          ukvr = Kr/vis;
          if (mpfao->bc_type == SEEPAGE_BC && mpfao->P_bnd[-cell_id_up-1] <= mpfao->Pref) {
            set_flow_to_zero = PETSC_TRUE;
          }
        }
      } else {
        // Is the cell_id_dn an internal or boundary cell?
        if (cell_id_dn>=0) {
          PetscReal Kr = mpfao->Kr[cell_id_dn];
          PetscReal vis = mpfao->vis[cell_id_dn];

          ukvr = Kr/vis;
        } else {
          PetscReal Kr = mpfao->Kr_bnd[-cell_id_dn-1];
          PetscReal vis = mpfao->vis_bnd[-cell_id_dn-1];

          ukvr = Kr/vis;
	        if (mpfao->bc_type == SEEPAGE_BC && mpfao->P_bnd[-cell_id_dn-1] <= mpfao->Pref) {
            set_flow_to_zero = PETSC_TRUE;
          }
        }
      }

      PetscReal fluxm = 0.0;
      if (set_flow_to_zero == PETSC_FALSE) {
        fluxm = den_aveg*ukvr*(-TtimesP[irow]);
        fluxm += - pow(den_aveg,2.0) * ukvr * G;
      }

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
        r_ptr[cell_id_up] += fluxm;
      }

      if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
        r_ptr[cell_id_dn] -= fluxm;
      }

    }
  }

  ierr = VecRestoreArray(R,&r_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscReal *p,*dp_dt,*r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

//#define DEBUG
#if defined(DEBUG)
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"IU.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  ierr = TDyGlobalToLocal(tdy,U,tdy->soln_loc); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(tdy->soln_loc,&p); CHKERRQ(ierr);
ierr = TDyUpdateState(tdy, p, mesh->num_cells); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_loc,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyMPFAOUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(mpfao->Trans_mat, mpfao->P_vec, mpfao->TtimesP_vec);

  ierr = TDyMPFAOIFunction_Vertices(tdy->soln_loc,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (PetscInt icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dP * dP/dt * Vol
    PetscReal rho = mpfao->rho[icell];
    PetscReal drho_dP = mpfao->drho_dP[icell];
    PetscReal porosity = mpfao->porosity[icell];
    PetscReal dporosity_dP = 0.0;
    PetscReal S = mpfao->S[icell];
    PetscReal dS_dP = mpfao->dS_dP[icell];

    PetscReal dmass_dP =
                rho     * dporosity_dP * S   +
                drho_dP * porosity     * S   +
                rho     * porosity     * dS_dP;

    PetscReal volume = cells->volume[icell];

    r[icell] += dmass_dP * dp_dt[icell] * volume;
    r[icell] -= mpfao->source_sink[icell] * volume;
  }

  /* Cleanup */
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

#if defined(DEBUG)
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"IFunction.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Computes the jacobian for the Richards equation
///
/// @param [in] Ul current value of the iterate
/// @param [out] A matrix for the jacobian
/// @param [in] ctx user-defined context
PetscErrorCode TDyMPFAOIJacobian_Vertices(Vec Ul, Mat A, void *ctx) {

  TDy tdy = ctx;
  TDyMPFAO* mpfao = tdy->context;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  TDyMesh *mesh  = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;

  PetscErrorCode ierr;

  PetscBool set_jac_to_zero = PETSC_FALSE;

  PetscScalar *TtimesP_vec_ptr, *GravDis_ptr;
  ierr = VecGetArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    PetscInt vertex_id = ivertex;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];

    PetscInt *face_ids, *subface_ids;
    PetscInt num_faces, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, vertex_id, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, vertex_id, &subface_ids, &num_subfaces); CHKERRQ(ierr);

    PetscInt npitf_bc = vertices->num_boundary_faces[ivertex];
    PetscInt nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    // Compute T*P
    PetscScalar TtimesP[nflux_in + npitf_bc];
    for (PetscInt irow=0; irow < nflux_in + npitf_bc; irow++) {

      PetscInt face_id = face_ids[irow];
      PetscInt subface_id = subface_ids[irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
    }

    //
    // fluxm_ij = rho_ij * (kr/mu)_{ij,upwind} * [T] * [P]  + rho_ij * (kr/mu)_{ij,upwind} * rho_ij * G
    //
    // For k = i or j, the jacobian is given as:
    // d(fluxm_ij)/dP_k =
    //                   d(rho_ij)/dP_k +   (kr/mu)_{ij,upwind}       * [T] * [P] +
    //                     rho_ij       + d((kr/mu)_{ij,upwind})/dP_k * [T] * [P] +
    //                     rho_ij       +   (kr/mu)_{ij,upwind}       * T_k       +
    //                  2 * rho_ij * d(rho_ij)/dP_k *   (kr/mu)_{ij,upwind}       * G  +
    //                      rho_ij *   rho_ij       * d((kr/mu)_{ij,upwind})/dP_k * G
    //
    // For k /= i or j, the jacobian is given as:
    // d(fluxm_ij)/dP_k =
    //                     rho_ij       +   (kr/mu)_{ij,upwind}       * T_k
    //

    for (PetscInt irow=0; irow<nflux_in + npitf_bc; irow++) {
      set_jac_to_zero = PETSC_FALSE;
      PetscInt face_id = face_ids[irow];
      PetscInt subface_id = subface_ids[irow];
      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      PetscInt cell_id_up = cell_ids[0];
      PetscInt cell_id_dn = cell_ids[1];

      // If using neumann bc (which is currently no-flux), then skip the face
      if ( mpfao->bc_type == NEUMANN_BC  && (cell_id_up<0 || cell_id_dn <0))  continue;

      PetscReal dukvr_dPup = 0.0;
      PetscReal dukvr_dPdn = 0.0;
      PetscReal ukvr;

      PetscReal den_aveg = 0.0;
      PetscReal dden_aveg_dPup = 0.0;
      PetscReal dden_aveg_dPdn = 0.0;

      if (cell_id_up>=0) {
        den_aveg += 0.5*mpfao->rho[cell_id_up];
        dden_aveg_dPup = 0.5*mpfao->drho_dP[cell_id_up];
      } else {
        den_aveg += 0.5*mpfao->rho_bnd[-cell_id_up-1];
        dden_aveg_dPup = 0.0;
      }

      if (cell_id_dn>=0) {
        den_aveg += 0.5*mpfao->rho[cell_id_dn];
        dden_aveg_dPdn = 0.5*mpfao->drho_dP[cell_id_dn];
      } else {
        den_aveg += 0.5*mpfao->rho_bnd[-cell_id_dn-1];
        dden_aveg_dPdn = 0.0;
      }

      PetscInt num_subfaces = 4;
      PetscReal G = GravDis_ptr[face_id*num_subfaces + subface_id];

      if (TtimesP[irow] + den_aveg * G < 0.0) { // up ---> dn
        // Flow: up --> dn
        if (cell_id_up>=0) {
          // "up" is an internal cell
          PetscReal Kr = mpfao->Kr[cell_id_up];
          PetscReal dKr_dS = mpfao->dKr_dS[cell_id_up];
          PetscReal vis = mpfao->vis[cell_id_up];
          PetscReal dvis_dP = mpfao->dvis_dP[cell_id_up];
          PetscReal dS_dP = mpfao->dS_dP[cell_id_up];

          ukvr       = Kr/vis;
          dukvr_dPup = dKr_dS*dS_dP/vis - Kr/(vis*vis)*dvis_dP;

        } else {
          // "up" is boundary cell
          PetscReal Kr = mpfao->Kr_bnd[-cell_id_up-1];
          PetscReal vis = mpfao->vis_bnd[-cell_id_up-1];

          ukvr = Kr/vis;
          dukvr_dPup = 0.0;

          if (mpfao->bc_type == SEEPAGE_BC && mpfao->P_bnd[-cell_id_up-1] <= mpfao->Pref) {
            set_jac_to_zero = PETSC_TRUE;
          }
        }

      } else {
        // Flow: up <--- dn
        if (cell_id_dn>=0) {
          // "dn" is an internal cell
          PetscReal Kr = mpfao->Kr[cell_id_dn];
          PetscReal dKr_dS = mpfao->dKr_dS[cell_id_dn];
          PetscReal vis = mpfao->vis[cell_id_dn];
          PetscReal dvis_dP = mpfao->dvis_dP[cell_id_dn];
          PetscReal dS_dP = mpfao->dS_dP[cell_id_dn];

          ukvr       = Kr/vis;
          dukvr_dPdn = dKr_dS*dS_dP/vis - Kr/(vis*vis)*dvis_dP;

        } else {
          // "dn" is a boundary cell
          PetscReal Kr = mpfao->Kr_bnd[-cell_id_dn-1];
          PetscReal vis = mpfao->vis_bnd[-cell_id_dn-1];

          ukvr       = Kr/vis;
          dukvr_dPdn = 0.0;

          if (mpfao->bc_type == SEEPAGE_BC && mpfao->P_bnd[-cell_id_dn-1] <= mpfao->Pref) {
            set_jac_to_zero = PETSC_TRUE;
          }
        }
      }

      PetscInt num_int_cells = vertices->num_internal_cells[ivertex];
      PetscInt up_cols[num_int_cells], dn_cols[num_int_cells];
      PetscReal up_Jac[num_int_cells], dn_Jac[num_int_cells];
      for (PetscInt icol=0; icol < num_int_cells; ++icol) {
        up_cols[icol] = -1;
        dn_cols[icol] = -1;
        PetscInt cell_id = vertices->internal_cell_ids[vOffsetCell + icol];

        PetscReal T = mpfao->Trans[vertex_id][irow][icol];

        PetscReal Jac;

        if (set_jac_to_zero == PETSC_TRUE){
          Jac = 0.0;
        } else if (cell_id_up>-1 && cell_id == cell_id_up) {
          Jac =
            dden_aveg_dPup * ukvr       * TtimesP[irow] +
            den_aveg       * dukvr_dPup * TtimesP[irow] +
            den_aveg       * ukvr       * T             +
            2*den_aveg *dden_aveg_dPup * ukvr       * G +
            pow(den_aveg,2.0)     * dukvr_dPup * G;
        } else if (cell_id_dn>-1 && cell_id == cell_id_dn) {
          Jac =
            dden_aveg_dPdn * ukvr       * TtimesP[irow] +
            den_aveg       * dukvr_dPdn * TtimesP[irow] +
            den_aveg       * ukvr       * T             +
            2*den_aveg *dden_aveg_dPdn * ukvr       * G +
            pow(den_aveg,2.0)     * dukvr_dPdn * G;
        } else {
          Jac = den_aveg * ukvr * T;
        }

        // Changing sign when bringing the term from RHS to LHS of the equation
        Jac = -Jac;

        if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
          up_cols[icol] = cell_id;
          up_Jac[icol] = Jac;
        }

        if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
          dn_cols[icol] = cell_id;
          dn_Jac[icol] = -Jac;
        }
      }

      // Set rows for the upward and downward fluxes.
      if (cell_id_up >= 0)  {
        ierr = MatSetValuesLocal(A,1,&cell_id_up,num_int_cells,up_cols,up_Jac,ADD_VALUES); CHKERRQ(ierr);
      }
      if (cell_id_dn >= 0) {
        ierr = MatSetValuesLocal(A,1,&cell_id_dn,num_int_cells,dn_cols,dn_Jac,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = VecRestoreArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Accumulation(Vec Ul,Vec Udotl,PetscReal shift,Mat A,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  PetscInt icell;
  PetscReal *dp_dt;
  PetscErrorCode ierr;

  PetscReal dporosity_dP = 0.0, d2porosity_dP2 = 0.0;
  PetscReal drho_dP, d2rho_dP2;
  PetscReal dmass_dP, d2mass_dP2;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  cells = &mesh->cells;

  ierr = VecGetArray(Udotl,&dp_dt); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    drho_dP = mpfao->drho_dP[icell];
    d2rho_dP2 = mpfao->d2rho_dP2[icell];

    // d(rho*phi*s)/dP * dP/dt * Vol
    dmass_dP = mpfao->rho[icell] * dporosity_dP         * mpfao->S[icell] +
               drho_dP         * mpfao->porosity[icell] * mpfao->S[icell] +
               mpfao->rho[icell] * mpfao->porosity[icell] * mpfao->dS_dP[icell];

    d2mass_dP2 = (
      mpfao->dS_dP[icell]   * mpfao->rho[icell]        * dporosity_dP         +
      mpfao->S[icell]       * drho_dP                * dporosity_dP         +
      mpfao->S[icell]       * mpfao->rho[icell]        * d2porosity_dP2       +
      mpfao->dS_dP[icell]   * drho_dP                * mpfao->porosity[icell] +
      mpfao->S[icell]       * d2rho_dP2              * mpfao->porosity[icell] +
      mpfao->S[icell]       * drho_dP                * dporosity_dP         +
      mpfao->d2S_dP2[icell] * mpfao->rho[icell]        * mpfao->porosity[icell] +
      mpfao->dS_dP[icell]   * drho_dP                * mpfao->porosity[icell] +
      mpfao->dS_dP[icell]   * mpfao->rho[icell]        * dporosity_dP
       );

    PetscReal value = (shift*dmass_dP + d2mass_dP2*dp_dt[icell])*cells->volume[icell];

    ierr = MatSetValuesLocal(A,1,&icell,1,&icell,&value,ADD_VALUES);CHKERRQ(ierr);

  }

  ierr = VecRestoreArray(Udotl,&dp_dt); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

  TDy tdy = (TDy)ctx;
  DM dm = (&tdy->tdydm)->dm;
  Vec Udotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  ierr = MatZeroEntries(B); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Udotl); CHKERRQ(ierr);

  ierr = TDyGlobalToLocal(tdy,U,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U_t,Udotl); CHKERRQ(ierr);

  ierr = TDyMPFAOIJacobian_Vertices(tdy->soln_loc, B, ctx); CHKERRQ(ierr);

  //ierr = TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_Accumulation(tdy->soln_loc, Udotl, shift, B, ctx);

  if (A !=B ) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

#if defined(DEBUG)
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"IJacobian.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(A,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}


