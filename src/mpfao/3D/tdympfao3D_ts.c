#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>

#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfao3Dutilsimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Vertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;

  PetscReal *r_ptr;
  PetscScalar *TtimesP_vec_ptr;
  PetscScalar *GravDis_ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  CharacteristicCurve *cc = tdy->cc;
  CharacteristicCurve *cc_bnd = tdy->cc_bnd;

  ierr = VecGetArray(R,&r_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;
    //if (vertices->num_boundary_faces[ivertex] != 0) continue;
    PetscInt vOffsetFace = vertices->face_offset[ivertex];

    PetscInt npitf_bc = vertices->num_boundary_faces[ivertex];
    PetscInt nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    // Compute = T*P
    PetscScalar TtimesP[nflux_in + npitf_bc];
    for (PetscInt irow=0; irow<nflux_in + npitf_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
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

      PetscInt fOffsetCell = faces->cell_offset[face_id];
      PetscInt cell_id_up = faces->cell_ids[fOffsetCell + 0];
      PetscInt cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      PetscReal den_aveg = 0.0;
      if (cell_id_up>=0) {
        den_aveg += tdy->rho[cell_id_up];
      } else {
        den_aveg += tdy->rho_BND[-cell_id_up-1];
      }

      if (cell_id_dn>=0) {
        den_aveg += tdy->rho[cell_id_dn];
      } else {
        den_aveg += tdy->rho_BND[-cell_id_dn-1];
      }

      den_aveg *= 0.5;

      PetscReal G = GravDis_ptr[face_id*num_subfaces + subface_id];

      // Upwind the 'ukvr'
      PetscReal ukvr = 0.0;
      if (TtimesP[irow] + den_aveg * G < 0.0) { // up ---> dn
        // Is the cell_id_up an internal or boundary cell?
        if (cell_id_up>=0) {
          PetscReal Kr = cc->Kr[cell_id_up];
          PetscReal vis = tdy->vis[cell_id_up];

          ukvr = Kr/vis;
        } else {
          PetscReal Kr = cc_bnd->Kr[-cell_id_up-1];
          PetscReal vis = tdy->vis_BND[-cell_id_up-1];

          ukvr = Kr/vis;
        }
      } else {
        // Is the cell_id_dn an internal or boundary cell?
        if (cell_id_dn>=0) {
          PetscReal Kr = cc->Kr[cell_id_dn];
          PetscReal vis = tdy->vis[cell_id_dn];

          ukvr = Kr/vis;
        } else {
          PetscReal Kr = cc_bnd->Kr[-cell_id_dn-1];
          PetscReal vis = tdy->vis_BND[-cell_id_dn-1];

          ukvr = Kr/vis;
        }
      }

      PetscReal fluxm = 0.0;
      fluxm = den_aveg*ukvr*(-TtimesP[irow]);
      fluxm += - pow(den_aveg,2.0) * ukvr * G;

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
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*dp_dt,*r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  CharacteristicCurve *cc = tdy->cc;
  MaterialProp *matprop = tdy->matprop;

//#define DEBUG
#if defined(DEBUG)
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"IU.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

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
  ierr = MatMult(tdy->Trans_mat, tdy->P_vec, tdy->TtimesP_vec);

  ierr = TDyMPFAOIFunction_Vertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (PetscInt icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dP * dP/dt * Vol
    PetscReal rho = tdy->rho[icell];
    PetscReal drho_dP = tdy->drho_dP[icell];
    PetscReal porosity = matprop->porosity[icell];
    PetscReal dporosity_dP = 0.0;
    PetscReal S = cc->S[icell];
    PetscReal dS_dP = cc->dS_dP[icell];

    PetscReal dmass_dP = 
                rho     * dporosity_dP * S   +
                drho_dP * porosity     * S   +
                rho     * porosity     * dS_dP;

    PetscReal volume = cells->volume[icell];

    r[icell] += dmass_dP * dp_dt[icell] * volume;
    r[icell] -= tdy->source_sink[icell] * volume;
  }

  /* Cleanup */
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

#if defined(DEBUG)
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"IFunction.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif
  
  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Vertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  TDyMesh *mesh  = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  DM dm = tdy->dm;
  CharacteristicCurve *cc = tdy->cc;
  CharacteristicCurve *cc_bnd = tdy->cc_bnd;

  PetscInt dim;
  PetscErrorCode ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscScalar *p, *TtimesP_vec_ptr, *GravDis_ptr;

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    PetscInt vertex_id = ivertex;

    if (vertices->num_boundary_faces[ivertex] > 0 && vertices->num_internal_cells[ivertex] < 2) continue;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    PetscInt npitf_bc = vertices->num_boundary_faces[ivertex];
    PetscInt nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    // Compute T*P
    PetscScalar TtimesP[nflux_in + npitf_bc];
    for (PetscInt irow=0; irow < nflux_in + npitf_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
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

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      PetscInt cell_id_up = faces->cell_ids[fOffsetCell + 0];
      PetscInt cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      PetscReal dukvr_dPup = 0.0;
      PetscReal dukvr_dPdn = 0.0;
      PetscReal ukvr;

      if (TtimesP[irow] < 0.0) {
        // Flow: up --> dn
        if (cell_id_up>=0) {
          // "up" is an internal cell
          ukvr       = cc->Kr[cell_id_up]/tdy->vis[cell_id_up];
          dukvr_dPup = cc->dKr_dS[cell_id_up]*cc->dS_dP[cell_id_up]/tdy->vis[cell_id_up] -
                       cc->Kr[cell_id_up]/(tdy->vis[cell_id_up]*tdy->vis[cell_id_up])*tdy->dvis_dP[cell_id_up];
        } else {
          // "up" is boundary cell
          ukvr = cc_bnd->Kr[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
          dukvr_dPup = 0.0;
        }

      } else {
        // Flow: up <--- dn
        if (cell_id_dn>=0) {
          // "dn" is an internal cell
          ukvr       = cc->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
          dukvr_dPdn = cc->dKr_dS[cell_id_dn]*cc->dS_dP[cell_id_dn]/tdy->vis[cell_id_dn] -
                       cc->Kr[cell_id_dn]/(tdy->vis[cell_id_dn]*tdy->vis[cell_id_dn])*tdy->dvis_dP[cell_id_dn];
        } else {
          // "dn" is a boundary cell
          ukvr       = cc_bnd->Kr[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
          dukvr_dPdn = 0.0;
        }
      }

      PetscReal den_aveg = 0.0;
      if (cell_id_up>=0) den_aveg += tdy->rho[cell_id_up];
      else               den_aveg += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den_aveg += tdy->rho[cell_id_dn];
      else               den_aveg += tdy->rho_BND[-cell_id_dn-1];
      den_aveg *= 0.5;

      // If one of the cell is on the boundary
      if (cell_id_up<0) cell_id_up = cell_id_dn;
      if (cell_id_dn<0) cell_id_dn = cell_id_up;

      PetscReal dden_aveg_dPup = 0.0;
      PetscReal dden_aveg_dPdn = 0.0;

      if (cell_id_up>=0) dden_aveg_dPup = 0.5*tdy->drho_dP[cell_id_up];
      if (cell_id_dn>=0) dden_aveg_dPdn = 0.5*tdy->drho_dP[cell_id_dn];

      PetscInt num_int_cells = vertices->num_internal_cells[ivertex];
      PetscInt up_cols[num_int_cells], dn_cols[num_int_cells];
      PetscReal up_Jac[num_int_cells], dn_Jac[num_int_cells];
      for (PetscInt icol=0; icol < num_int_cells; ++icol) {
        up_cols[icol] = -1;
        dn_cols[icol] = -1;
        PetscInt cell_id = vertices->internal_cell_ids[vOffsetCell + icol];

        PetscReal T = tdy->Trans[vertex_id][irow][icol];

        PetscReal gz;
        ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

        PetscReal Jac;
        if (cell_id_up>-1 && cell_id == cell_id_up) {
          Jac =
            dden_aveg_dPup * ukvr       * TtimesP[irow] +
            den_aveg       * dukvr_dPup * TtimesP[irow] +
            den_aveg       * ukvr       * T * (1.0 + dden_aveg_dPup*gz) ;
        } else if (cell_id_dn>-1 && cell_id == cell_id_dn) {
          Jac =
            dden_aveg_dPdn * ukvr       * TtimesP[irow] +
            den_aveg       * dukvr_dPdn * TtimesP[irow] +
            den_aveg       * ukvr       * T * (1.0 + dden_aveg_dPdn*gz) ;
        } else {
          Jac = den_aveg * ukvr * T * (1.0 + 0.0*gz);
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
      if (cell_id_up >= 0){
        ierr = MatSetValuesLocal(A,1,&cell_id_up,num_int_cells,up_cols,up_Jac,ADD_VALUES); CHKERRQ(ierr);
      }
      if (cell_id_dn >= 0){
        ierr = MatSetValuesLocal(A,1,&cell_id_dn,num_int_cells,dn_cols,dn_Jac,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Computes the jacobian for the Richards equation for two-point flux
///
/// @param [in] Ul current value of the iterate
/// @param [out] A matrix for the jacobian
/// @param [in] ctx user-defined context
PetscErrorCode TDyMPFAOIJacobian_Vertices_3DMesh_TPF(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  TDyMesh *mesh  = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  CharacteristicCurve *cc = tdy->cc;
  CharacteristicCurve *cc_bnd = tdy->cc_bnd;

  PetscErrorCode ierr;

  PetscScalar *TtimesP_vec_ptr, *GravDis_ptr;
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    PetscInt vertex_id = ivertex;

    PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
    PetscInt vOffsetFace    = vertices->face_offset[ivertex];

    PetscInt npitf_bc = vertices->num_boundary_faces[ivertex];
    PetscInt nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    // Compute T*P
    PetscScalar TtimesP[nflux_in + npitf_bc];
    for (PetscInt irow=0; irow < nflux_in + npitf_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
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

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      PetscInt cell_id_up = faces->cell_ids[fOffsetCell + 0];
      PetscInt cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      // If using neumann bc (which is currently no-flux), then skip the face
      if ( tdy->mpfao_bc_type == MPFAO_NEUMANN_BC  && (cell_id_up<0 || cell_id_dn <0))  continue;

      PetscReal dukvr_dPup = 0.0;
      PetscReal dukvr_dPdn = 0.0;
      PetscReal ukvr;

      PetscReal den_aveg = 0.0;
      PetscReal dden_aveg_dPup = 0.0;
      PetscReal dden_aveg_dPdn = 0.0;

      if (cell_id_up>=0) {
        den_aveg += 0.5*tdy->rho[cell_id_up];
        dden_aveg_dPup = 0.5*tdy->drho_dP[cell_id_up];
      } else {
        den_aveg += 0.5*tdy->rho_BND[-cell_id_up-1];
        dden_aveg_dPup = 0.0;
      }

      if (cell_id_dn>=0) {
        den_aveg += 0.5*tdy->rho[cell_id_dn];
        dden_aveg_dPup = 0.5*tdy->drho_dP[cell_id_dn];
      } else {
        den_aveg += 0.5*tdy->rho_BND[-cell_id_dn-1];
        dden_aveg_dPup = 0.0;
      }

      PetscInt num_subfaces = 4;
      PetscReal G = GravDis_ptr[face_id*num_subfaces + subface_id];

      if (TtimesP[irow] + den_aveg * G < 0.0) { // up ---> dn
        // Flow: up --> dn
        if (cell_id_up>=0) {
          // "up" is an internal cell
          PetscReal Kr = cc->Kr[cell_id_up];
          PetscReal dKr_dS = cc->dKr_dS[cell_id_up];
          PetscReal vis = tdy->vis[cell_id_up];
          PetscReal dvis_dP = tdy->dvis_dP[cell_id_up];
          PetscReal dS_dP = cc->dS_dP[cell_id_up];

          ukvr       = Kr/vis;
          dukvr_dPup = dKr_dS*dS_dP/vis - Kr/(vis*vis)*dvis_dP;
        } else {
          // "up" is boundary cell
          PetscReal Kr = cc_bnd->Kr[-cell_id_up-1];
          PetscReal vis = tdy->vis_BND[-cell_id_up-1];

          ukvr = Kr/vis;
          dukvr_dPup = 0.0;
        }

      } else {
        // Flow: up <--- dn
        if (cell_id_dn>=0) {
          // "dn" is an internal cell
          PetscReal Kr = cc->Kr[cell_id_dn];
          PetscReal dKr_dS = cc->dKr_dS[cell_id_dn];
          PetscReal vis = tdy->vis[cell_id_dn];
          PetscReal dvis_dP = tdy->dvis_dP[cell_id_dn];
          PetscReal dS_dP = cc->dS_dP[cell_id_dn];

          ukvr       = Kr/vis;
          dukvr_dPdn = dKr_dS*dS_dP/vis - Kr/(vis*vis)*dvis_dP;

        } else {
          // "dn" is a boundary cell
          PetscReal Kr = cc_bnd->Kr[-cell_id_dn-1];
          PetscReal vis = tdy->vis_BND[-cell_id_dn-1];

          ukvr       = Kr/vis;
          dukvr_dPdn = 0.0;
        }
      }

      PetscInt num_int_cells = vertices->num_internal_cells[ivertex];
      PetscInt up_cols[num_int_cells], dn_cols[num_int_cells];
      PetscReal up_Jac[num_int_cells], dn_Jac[num_int_cells];
      for (PetscInt icol=0; icol < num_int_cells; ++icol) {
        up_cols[icol] = -1;
        dn_cols[icol] = -1;
        PetscInt cell_id = vertices->internal_cell_ids[vOffsetCell + icol];

        PetscReal T = tdy->Trans[vertex_id][irow][icol];

        PetscReal Jac;

        if (cell_id_up>-1 && cell_id == cell_id_up) {
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

  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Accumulation_3DMesh(Vec Ul,Vec Udotl,PetscReal shift,Mat A,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscInt icell;
  PetscReal *dp_dt;
  PetscErrorCode ierr;

  PetscReal dporosity_dP = 0.0, d2porosity_dP2 = 0.0;
  PetscReal drho_dP, d2rho_dP2;
  PetscReal dmass_dP, d2mass_dP2;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  cells = &mesh->cells;
  CharacteristicCurve *cc = tdy->cc;
  MaterialProp *matprop = tdy->matprop;

  ierr = VecGetArray(Udotl,&dp_dt); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    drho_dP = tdy->drho_dP[icell];
    d2rho_dP2 = tdy->d2rho_dP2[icell];

    // d(rho*phi*s)/dP * dP/dt * Vol
    dmass_dP = tdy->rho[icell] * dporosity_dP         * cc->S[icell] +
               drho_dP         * matprop->porosity[icell] * cc->S[icell] +
               tdy->rho[icell] * matprop->porosity[icell] * cc->dS_dP[icell];

    d2mass_dP2 = (
      cc->dS_dP[icell]   * tdy->rho[icell]        * dporosity_dP         +
      cc->S[icell]       * drho_dP                * dporosity_dP         +
      cc->S[icell]       * tdy->rho[icell]        * d2porosity_dP2       +
      cc->dS_dP[icell]   * drho_dP                * matprop->porosity[icell] +
      cc->S[icell]       * d2rho_dP2              * matprop->porosity[icell] +
      cc->S[icell]       * drho_dP                * dporosity_dP         +
      cc->d2S_dP2[icell] * tdy->rho[icell]        * matprop->porosity[icell] +
      cc->dS_dP[icell]   * drho_dP                * matprop->porosity[icell] +
      cc->dS_dP[icell]   * tdy->rho[icell]        * dporosity_dP
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
PetscErrorCode TDyMPFAOIJacobian_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  DM             dm = tdy->dm;
  Vec Ul, Udotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  ierr = MatZeroEntries(B); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Udotl); CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U_t,INSERT_VALUES,Udotl); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U_t,INSERT_VALUES,Udotl); CHKERRQ(ierr);

  ierr = TDyMPFAOIJacobian_Vertices_3DMesh_TPF(Ul, B, ctx); CHKERRQ(ierr);

  //ierr = TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_Accumulation_3DMesh(Ul, Udotl, shift, B, ctx);

  if (A !=B ) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
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


