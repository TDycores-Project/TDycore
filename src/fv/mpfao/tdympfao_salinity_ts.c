#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoimpl.h>
#include <private/tdympfaosalinitytsimpl.h>
#include <private/tdydiscretizationimpl.h>

//#define DEBUG
#if defined(DEBUG)
PetscInt icount_f = 0;
PetscInt icount_j = 0;
PetscInt max_count = 5;
#endif

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Vertices_Salinity(Vec Ul, Vec R, void *ctx) {

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
  PetscReal den,fluxm,ukvr,fluxt,flow_rate;
  PetscScalar *TtimesP_vec_ptr, *TtimesPsi_vec_ptr;
  PetscErrorCode ierr;
  PetscScalar *GravDis_ptr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(mpfao->TtimesPsi_vec,&TtimesPsi_vec_ptr); CHKERRQ(ierr); //CHANGE
  ierr = VecGetArray(mpfao->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    PetscInt *face_ids, num_faces;
    PetscInt *subface_ids, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

    npitf_bc = vertices->num_boundary_faces[ivertex];
    nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_faces[ivertex];

    PetscScalar TtimesP[nflux_in + npitf_bc], TtimesPsi[nflux_in + npitf_bc];

    // Compute = T*P
    for (irow=0; irow<nflux_in + npitf_bc; irow++) {

      PetscInt face_id = face_ids[irow];
      PetscInt subface_id = subface_ids[irow];
      PetscInt num_subfaces = 4;

      if (!faces->is_local[face_id]) continue;

      TtimesP[irow] = TtimesP_vec_ptr[face_id*num_subfaces + subface_id];
      TtimesPsi[irow] = TtimesPsi_vec_ptr[face_id*num_subfaces + subface_id];

      PetscInt *cell_ids, num_cells;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &cell_ids, &num_cells); CHKERRQ(ierr);

      cell_id_up = cell_ids[0];
      cell_id_dn = cell_ids[1];

      if (TtimesP[irow] < 0.0) { // up ---> dn
                                 // Is the cell_id_up an internal or boundary cell?
        if (cell_id_up >= 0) {
          ukvr = mpfao->Kr[cell_id_up]/mpfao->vis[cell_id_up]; //vis based on salinity
        }
        else {
          ukvr = mpfao->Kr_bnd[-cell_id_up-1]/mpfao->vis_bnd[-cell_id_up-1];
        }
      }
      else {
        // Is the cell_id_dn an internal or boundary cell?
        if (cell_id_dn >= 0) {
          ukvr = mpfao->Kr[cell_id_dn]/mpfao->vis[cell_id_dn];
        }
        else {
          ukvr = mpfao->Kr_bnd[-cell_id_dn-1]/mpfao->vis_bnd[-cell_id_dn-1];
        }
      }

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

      flow_rate = ukvr*(-TtimesP[irow]); // flow_rate is darcy flux times area

      fluxm = den_aveg*flow_rate;
      fluxt = den_aveg*flow_rate;  //Advection term

      // Diffusion term in transport equation?
      fluxt += -den_aveg*TtimesPsi[irow];
      fluxm += - pow(den_aveg,2.0) * ukvr * G;

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
        r[cell_id_up*2]   += fluxm;
        r[cell_id_up*2+1] += fluxt;
      }
      if (cell_id_dn >=0 && cells->is_local[cell_id_dn]) {
        r[cell_id_dn*2]   -= fluxm;
        r[cell_id_dn*2+1] -= fluxt;
      }
    }
  }

  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->TtimesPsi_vec,&TtimesPsi_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(mpfao->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Salinity(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {

  TDy       tdy   = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh  *mesh  = mpfao->mesh;
  TDyCell  *cells = &mesh->cells;
  DM        dm;
  Vec       Ul;
  PetscReal *du_dt,*r,*u_p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  #if defined(DEBUG)
  PetscViewer viewer;
  char word[32];
  sprintf(word,"U%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"U2.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  #endif

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U,Ul); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy,u_p, mesh->num_cells); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyMPFAO_SetBoundarySalinity(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyMPFAOUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(mpfao->Trans_mat,mpfao->P_vec,mpfao->TtimesP_vec);
  ierr = MatMult(mpfao->Psi_Trans_mat,mpfao->Psi_vec,mpfao->TtimesPsi_vec);

  #if 0
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psi_Trans_mat.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(mpfao->Psi_Trans_mat,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psi_TtimesP_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(mpfao->TtimesPsi_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psi_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(mpfao->Psi_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"P_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(mpfao->P_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  #endif

  // Fluxes
  ierr = TDyMPFAOIFunction_Vertices_Salinity(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscInt c,cStart,cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscReal p[cEnd-cStart], dp_dt[cEnd-cStart],
            Psi[cEnd-cStart], dPsi_dt[cEnd-cStart];

  for (c=0;c<cEnd-cStart;c++) {
    dp_dt[c]    = du_dt[c*2];
    p[c]        = u_p[c*2];
    dPsi_dt[c] = du_dt[c*2+1];
    Psi[c]     = u_p[c*2+1];
  }

  MaterialProp *matprop = tdy->matprop;

  PetscReal dporosity_dP = 0.0;
  PetscReal dporosity_dPsi = 0.0;
  PetscReal dS_dPsi = 0.0;
  PetscReal dPsi_dP = 0.0;
  PetscReal dmass_dP,dmass_dPsi;
  PetscInt icell;

  // Accumulation and source/sink contributions
  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) break;

    // A_M = d(rho*phi*s)/dP * dP_dtime * Vol + d(rho*phi*s)/dT * dT_dtime * Vol //change
    dmass_dP = mpfao->rho[icell]       * dporosity_dP           * mpfao->S[icell] +
               mpfao->drho_dP[icell]   * mpfao->porosity[icell] * mpfao->S[icell] +
               mpfao->rho[icell]       * mpfao->porosity[icell] * mpfao->dS_dP[icell];
    dmass_dPsi = mpfao->rho[icell]       * dporosity_dPsi         * mpfao->S[icell] +
                 mpfao->drho_dPsi[icell] * mpfao->porosity[icell] * mpfao->S[icell] +
                 mpfao->rho[icell]       * mpfao->porosity[icell] * dS_dPsi;

    PetscReal dtrans_dP, dtrans_dPsi;

    dtrans_dP =
      mpfao->drho_dP[icell] * mpfao->porosity[icell] * mpfao->S[icell]     * Psi[icell] +
      mpfao->rho[icell]     * dporosity_dP           * mpfao->S[icell]     * Psi[icell] +
      mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->dS_dP[icell] * Psi[icell] +
      mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->S[icell]     * dPsi_dP;
    dtrans_dPsi =
      mpfao->drho_dPsi[icell] * mpfao->porosity[icell] * mpfao->S[icell] * Psi[icell] +
      mpfao->rho[icell]       * dporosity_dPsi         * mpfao->S[icell] * Psi[icell] +
      mpfao->rho[icell]       * mpfao->porosity[icell] * dS_dPsi         * Psi[icell] +
      mpfao->rho[icell]       * mpfao->porosity[icell] * mpfao->S[icell];

    r[icell*2]   += dmass_dP * dp_dt[icell] * cells->volume[icell] + dmass_dPsi * dPsi_dt[icell] * cells->volume[icell];
    r[icell*2+1] += dtrans_dP * dp_dt[icell] * cells->volume[icell] + dtrans_dPsi * dPsi_dt[icell] * cells->volume[icell];
    r[icell*2]   -= mpfao->source_sink[icell] * cells->volume[icell];
    r[icell*2+1] -= mpfao->salinity_source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

#if defined(DEBUG)
  sprintf(word,"Function%d.vec");
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Function.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  icount_f++;
#endif

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

