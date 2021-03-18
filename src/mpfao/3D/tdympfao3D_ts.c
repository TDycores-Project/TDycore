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
  printf("TDyMPFAOIFunction_Vertices_3DMesh\n");

  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;
    //if (vertices->num_boundary_faces[ivertex] != 0) continue;
    PetscInt vOffsetFace = vertices->face_offset[ivertex];
    PetscInt print_info = 0;
    //if (ivertex == 0 || ivertex == 3 || ivertex == 12) print_info = 1;
    //if (ivertex == 0) print_info = 1;

    PetscInt npitf_bc = vertices->num_boundary_faces[ivertex];
    PetscInt nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];
    if (print_info) printf("ivertex = %03d\n",ivertex);

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
      if (print_info) printf("   [%d] face_id = %03d; %+03d %+03d",face_id*num_subfaces + subface_id, face_id, cell_id_up, cell_id_dn);

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

      PetscReal G;
      if (tdy->mpfao_gmatrix_method == MPFAO_GMATRIX_TPF) {
        G = GravDis_ptr[face_id*num_subfaces + subface_id];
      } else {
        G = 0.0;
      }

      // Upwind the 'ukvr'
      PetscReal ukvr = 0.0;
      if (print_info) 
        printf("   G_index = %03d TtimesP[%03d * %03d + %03d] + den_aveg * G = %19.18e + %19.18e = %19.18e", 
          face_id * num_subfaces +  subface_id, face_id, num_subfaces, subface_id, TtimesP[irow],  den_aveg * G, TtimesP[irow] + den_aveg * G);
      if (TtimesP[irow] + den_aveg * G < 0.0) { // up ---> dn
        // Is the cell_id_up an internal or boundary cell?
        if (print_info) printf("  up --> dn\n");
        if (cell_id_up>=0) {
          PetscReal Kr = cc->Kr[cell_id_up];
          PetscReal vis = tdy->vis[cell_id_up];

          ukvr = Kr/vis;
          if (print_info) printf("   (a) Kr = %+19.18e; vis = %+19.18e ",Kr,vis);
        } else {
          PetscReal Kr = cc_bnd->Kr[-cell_id_up-1];
          PetscReal vis = tdy->vis_BND[-cell_id_up-1];

          ukvr = Kr/vis;
          if (print_info) printf("   (b) Kr = %+19.18e; vis = %+19.18e ",Kr,vis);
        }
      } else {
        if (print_info) printf("  up <-- dn\n");
        // Is the cell_id_dn an internal or boundary cell?
        if (cell_id_dn>=0) {
          PetscReal Kr = cc->Kr[cell_id_dn];
          PetscReal vis = tdy->vis[cell_id_dn];

          ukvr = Kr/vis;
          if (print_info) printf("   (c) Kr = %+19.18e; vis = %+19.18e ",Kr,vis);
        } else {
          PetscReal Kr = cc_bnd->Kr[-cell_id_dn-1];
          PetscReal vis = tdy->vis_BND[-cell_id_dn-1];

          ukvr = Kr/vis;
          if (print_info) printf("   (d) Kr = %+19.18e; vis = %+19.18e ",Kr,vis);
        }
      }

      PetscReal fluxm = 0.0;
      fluxm = den_aveg*ukvr*(-TtimesP[irow]);
      fluxm += - pow(den_aveg,2.0) * ukvr * G;
      if (print_info) printf("\n       flux_m = %+19.18e; den = %+19.18e; ukvr = %+19.18e;\n       G = %+19.18e TxP = %+19.18e\n\n",
        fluxm, den_aveg, ukvr, G, (TtimesP[irow]));

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
        r_ptr[cell_id_up] += fluxm;
      }

      if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
        r_ptr[cell_id_dn] -= fluxm;
      }

    }
    //exit(0);
  }
  //exit(0);

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
  PetscViewer viewer;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"T.bin",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = MatView(tdy->Trans_mat,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"P.bin",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = VecView(tdy->P_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

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

    //r[icell] += dmass_dP * dp_dt[icell] * volume;
    //r[icell] -= tdy->source_sink[icell] * volume;
  }

  /* Cleanup */
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"R.bin",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  //exit(0);

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
    printf("  nflux_in = %d; npitf_bc = %d\n",nflux_in, npitf_bc);
    for (PetscInt irow=0; irow < nflux_in + npitf_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;
      printf("    face_id = %d; subface_id = %d\n",face_id,subface_id);

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
      printf("  [%d] face_id = %03d; cell_up = %+03d; cell_dn = %+03d\n",irow, face_id, cell_id_up, cell_id_dn);

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
      printf("  Updated cell_up = %+03d; cell_dn = %+03d\n",cell_id_up,cell_id_dn);

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
        printf("    [%d] cell_id = %+03d\n",icol,cell_id);

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
      ierr = MatSetValuesLocal(A,1,&cell_id_up,num_int_cells,up_cols,up_Jac,ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&cell_id_dn,num_int_cells,dn_cols,dn_Jac,ADD_VALUES);CHKERRQ(ierr);
    }
    exit(0);
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

  printf("In Jacobian\n");
  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    PetscInt vertex_id = ivertex;

    //if (vertices->num_boundary_faces[ivertex] > 0 && vertices->num_internal_cells[ivertex] < 2) continue;
    //printf("ivertex = %d\n",ivertex);
    /*{
      for (PetscInt ii=0; ii<5; ii++) {
        for (PetscInt jj=0; jj<5; jj++) //printf("%+19.18e ",tdy->Trans[vertex_id][ii][jj]);
        //printf("\n");
      }
    }*/

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
    if (ivertex == 3){
    printf("TT: \n");
    for (PetscInt irow=0; irow<nflux_in + npitf_bc; irow++) {
         for (PetscInt icol=0; icol < vertices->num_internal_cells[ivertex] + vertices->num_boundary_faces[ivertex] ; ++icol) {
           printf("%+19.18e ",tdy->Trans[vertex_id][irow][icol]);
         }
         printf("\n");
    }
    }

    for (PetscInt irow=0; irow<nflux_in + npitf_bc; irow++) {

      PetscInt face_id = vertices->face_ids[vOffsetFace + irow];
      PetscInt subface_id = vertices->subface_ids[vOffsetFace + irow];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      PetscInt cell_id_up = faces->cell_ids[fOffsetCell + 0];
      PetscInt cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      PetscInt print_info = 0;
      if (ivertex == 3) print_info = 1;
      //if (ivertex == 2 || ivertex == 3 || ivertex == 6 || ivertex == 7) print_info = 1;
      //if (ivertex == 18 || ivertex == 19 || ivertex == 22 || ivertex == 23) print_info = 1;
      //if (cell_id_up == 0 && cell_id_dn == 1) print_info = 1;
      //if ((cell_id_up == 2 || cell_id_dn == 2) && (cell_id_up < 0 || cell_id_dn < 0)) print_info = 1;
      //if ((cell_id_up == 2 || cell_id_dn == 2) && (cell_id_up >= 0 && cell_id_dn >= 0)) print_info = 1;

      if (print_info) {
        printf("vertex_id: %d\n",vertex_id);
        printf("  [irow = %d]face_id = %03d cell_id_up/dn = %+03d %+03d\n",irow,face_id,cell_id_up,cell_id_dn);
      }

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
        //printf("  (a) rho_up = %+19.18e; drho_dPup = %+19.18e\n",tdy->rho[cell_id_up],tdy->drho_dP[cell_id_up]);
      } else {
        den_aveg += 0.5*tdy->rho_BND[-cell_id_up-1];
        dden_aveg_dPup = 0.0;
        //printf("  (b) rho_up = %+19.18e; drho_dPup = %+19.18e\n",tdy->rho_BND[-cell_id_up-1],0.0);
      }

      if (cell_id_dn>=0) {
        den_aveg += 0.5*tdy->rho[cell_id_dn];
        dden_aveg_dPup = 0.5*tdy->drho_dP[cell_id_dn];
        //printf("  (c) rho_dn = %+19.18e; drho_dPup = %+19.18e\n",tdy->rho[cell_id_dn],tdy->drho_dP[cell_id_dn]);
      } else {
        den_aveg += 0.5*tdy->rho_BND[-cell_id_dn-1];
        dden_aveg_dPup = 0.0;
        //printf("  (d) rho_dn = %+19.18e; drho_dPup = %+19.18e\n",tdy->rho_BND[-cell_id_dn-1],0.0);
      }

      PetscInt num_subfaces = 4;
      PetscReal G = GravDis_ptr[face_id*num_subfaces + subface_id];
      //printf("  G = %+19.18e\n",G);

      if (TtimesP[irow] + den_aveg * G < 0.0) { // up ---> dn
        if (print_info) printf("up --> dn\n");
        // Flow: up --> dn
        //printf("  up --> dn\n");
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
          if (print_info) printf("  (2) Kr = %+19.18e; vis = %+19.18e; ukvr = %+19.18e\n",Kr,vis,ukvr);
        }

      } else {
        // Flow: up <--- dn
        if (print_info) printf("up --> dn\n");
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
          if (print_info) printf("  (2) Kr = %+19.18e; vis = %+19.18e; ukvr = %+19.18e\n",Kr,vis,ukvr);
        }
      }

      // If one of the cell is on the boundary
      /*
      if (cell_id_up<0) {
        cell_id_up = cell_id_dn;
      }

      if (cell_id_dn<0) {
        cell_id_dn = cell_id_up;
      }

      if (cell_id_up>=0) {
        dden_aveg_dPup = 0.5*tdy->drho_dP[cell_id_up];
      }
      if (cell_id_dn>=0) {
        dden_aveg_dPdn = 0.5*tdy->drho_dP[cell_id_dn];
      }
      */

      PetscInt num_int_cells = vertices->num_internal_cells[ivertex];
      PetscInt up_cols[num_int_cells], dn_cols[num_int_cells];
      PetscReal up_Jac[num_int_cells], dn_Jac[num_int_cells];
      for (PetscInt icol=0; icol < num_int_cells; ++icol) {
        up_cols[icol] = -1;
        dn_cols[icol] = -1;
        PetscInt cell_id = vertices->internal_cell_ids[vOffsetCell + icol];
        if (print_info) printf("    [icol = %d] cell_id = %d\n",icol,cell_id);

        PetscReal T = tdy->Trans[vertex_id][irow][icol];

        PetscReal Jac;

        if (cell_id_up>-1 && cell_id == cell_id_up) {
          Jac =
            dden_aveg_dPup * ukvr       * TtimesP[irow] +
            den_aveg       * dukvr_dPup * TtimesP[irow] +
            den_aveg       * ukvr       * T             +
            2*den_aveg *dden_aveg_dPup * ukvr       * G +
            pow(den_aveg,2.0)     * dukvr_dPup * G;
            if (print_info && cell_id == 2) {
            //if (cell_id == -2) {
            printf("    (a) Jac = %+19.18e\n",Jac);
            printf("      term 1 = %+19.18e\n",dden_aveg_dPdn * ukvr       * TtimesP[irow]);
            printf("      term 2 = %+19.18e\n",den_aveg       * dukvr_dPdn * TtimesP[irow]);
            printf("      term 3 = %+19.18e\n",den_aveg       * ukvr       * T);
            printf("      term 4 = %+19.18e\n",2*den_aveg *dden_aveg_dPup * ukvr       * G);
            printf("      term 5 = %+19.18e\n",pow(den_aveg,2.0)     * dukvr_dPup * G);
            printf("      TxP    = %+19.18e\n",TtimesP[irow]);
            printf("      ukvr   = %+19.18e\n",ukvr);
            printf("      T      = %+19.18e\n",T);
            printf("      G      = %+19.18e\n",G);
            }
        } else if (cell_id_dn>-1 && cell_id == cell_id_dn) {
          Jac =
            dden_aveg_dPdn * ukvr       * TtimesP[irow] +
            den_aveg       * dukvr_dPdn * TtimesP[irow] +
            den_aveg       * ukvr       * T             +
            2*den_aveg *dden_aveg_dPdn * ukvr       * G +
            pow(den_aveg,2.0)     * dukvr_dPdn * G;
            if (print_info && cell_id == 2) {
            //if (cell_id == -2) {
            printf("    (b) Jac = %+19.18e\n",Jac);
            printf("      term 1 = %+19.18e\n",dden_aveg_dPdn * ukvr       * TtimesP[irow]);
            printf("      term 2 = %+19.18e\n",den_aveg       * dukvr_dPdn * TtimesP[irow]);
            printf("      term 3 = %+19.18e\n",den_aveg       * ukvr       * T);
            printf("      term 4 = %+19.18e\n",2*den_aveg *dden_aveg_dPup * ukvr       * G);
            printf("      term 5 = %+19.18e\n",pow(den_aveg,2.0)     * dukvr_dPup * G);
            printf("      TxP    = %+19.18e\n",TtimesP[irow]);
            printf("      ukvr   = %+19.18e\n",ukvr);
            printf("      T      = %+19.18e\n",T);
            printf("      G      = %+19.18e\n",G);
            }
            
        } else {
          Jac = den_aveg * ukvr * T;
          //if (print_info && cell_id == 2) printf("    (c) Jac = %+19.18e\n",Jac);
          if (cell_id == -2) printf("    (c) Jac = %+19.18e\n",Jac);
        }
        //if (face_id == 72 && ivertex == 1) exit(0);
        if (print_info) printf("\n");

        // Changing sign when bringing the term from RHS to LHS of the equation
        Jac = -Jac;
        PetscInt cell_id_of_interest = 2;

        if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
          up_cols[icol] = cell_id;
          up_Jac[icol] = Jac;
          //if (cell_id_up == 2 || cell_id == 2) printf("up Jac = %+19.18e\n",Jac);
          if (cell_id_up == cell_id_of_interest && cell_id == cell_id_of_interest) 
            printf("(%d, %d) up Jac = %+19.18e; vertex_id = %02d face_id = %03d; cell_up/dn = %03d %03d; cell_id = %02d; T = %+19.18e\n",
              irow,icol,Jac,ivertex,face_id,cell_id_up,cell_id_dn,cell_id,T);
        }

        if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
          dn_cols[icol] = cell_id;
          dn_Jac[icol] = -Jac;
          //if (cell_id_dn == 2 || cell_id == 2) printf("dn Jac = %+19.18e\n",-Jac);
          if (cell_id_dn == cell_id_of_interest && cell_id == cell_id_of_interest) 
            printf("(%d, %d) dn Jac = %+19.18e; vertex_id = %02d face_id = %03d; cell_up/dn = %03d %03d; cell_id = %02d; T = %+19.18e\n",
              irow,icol,Jac,ivertex,face_id,cell_id_up,cell_id_dn,cell_id,T);
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
    //exit(0);
  }
//  exit(0);

  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"J.bin",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = MatView(A,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  //exit(0);

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

  switch (tdy->mpfao_gmatrix_method){
    case MPFAO_GMATRIX_DEFAULT:
      ierr = TDyMPFAOIJacobian_Vertices_3DMesh(Ul, B, ctx); CHKERRQ(ierr);
      break;
    case MPFAO_GMATRIX_TPF:
      ierr = TDyMPFAOIJacobian_Vertices_3DMesh_TPF(Ul, B, ctx); CHKERRQ(ierr);
      break;
  }
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


