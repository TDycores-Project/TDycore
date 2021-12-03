#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdydiscretization.h>

//#define DEBUG
#if defined(DEBUG)
PetscInt icount_f = 0;
PetscInt icount_j = 0;
PetscInt max_count = 5;
#endif

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Vertices_Salinity(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  DM dm = tdy->dm;
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


  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->TtimesPsi_vec,&TtimesPsi_vec_ptr); CHKERRQ(ierr); //CHANGE
  ierr = VecGetArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);
  
   CharacteristicCurve *cc = tdy->cc;
   CharacteristicCurve *cc_bnd = tdy->cc_bnd;

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
          ukvr = cc->Kr[cell_id_up]/tdy->vis[cell_id_up]; //vis based on salinity
        }
        else {
          ukvr = cc_bnd->Kr[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
        }
      }
      else {
         // Is the cell_id_dn an internal or boundary cell?
         if (cell_id_dn >= 0) {
          ukvr = cc->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
         }
         else {
          ukvr = cc_bnd->Kr[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
         }
      }

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
      
      flow_rate = ukvr*(-TtimesP[irow]); // flow_rate is darcy flux times area

      fluxm = den_aveg*flow_rate;
      fluxt = den_aveg*flow_rate;  //Advection term

      //printf("den = %f\n",den_aveg);
      //   printf("flow = %f\n",flow_rate);

      // Diffusion term in transport equation?
      fluxt += -den_aveg * TtimesPsi[irow];
      fluxm += - pow(den_aveg,2.0) * ukvr * G;

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
        r[cell_id_up*2]   += fluxm;   //check index here
        r[cell_id_up*2+1] += fluxt;
      }
      if (cell_id_dn >=0 && cells->is_local[cell_id_dn]) {
        r[cell_id_dn*2]   -= fluxm;
        r[cell_id_dn*2+1] -= fluxt;
      }
    }
  }

  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->TtimesP_vec,&TtimesP_vec_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->TtimesPsi_vec,&TtimesPsi_vec_ptr); CHKERRQ(ierr);
ierr = VecRestoreArray(tdy->GravDisVec, &GravDis_ptr); CHKERRQ(ierr);
 
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_Salinity(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*du_dt,*r,*Psi,*u_p,*dp_dt,*dPsi_dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  //#if defined(DEBUG)
  PetscViewer viewer;
  // char word[32];
  // sprintf(word,"U%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"U2.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  //#endif

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U,Ul); CHKERRQ(ierr);
  
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Ul.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(Ul,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  
  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy,u_p); CHKERRQ(ierr);
  //  ierr = VecRestoreArray(Ul,&U_p); CHKERRQ(ierr);
  
  ierr = TDyMPFAO_SetBoundaryPressure(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyMPFAO_SetBoundaryConcentration(tdy,Ul); CHKERRQ(ierr); //write concentration
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(tdy->Trans_mat,tdy->P_vec,tdy->TtimesP_vec);
  //add in matmult for trans * rho, or maybe not
  ierr = MatMult(tdy->Psi_Trans_mat,tdy->Psi_vec,tdy->TtimesPsi_vec); //todo multply in sat

  
  //#if 0
  // PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psi_Trans_mat.mat",&viewer); CHKERRQ(ierr);
  ierr = MatView(tdy->Psi_Trans_mat,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psi_TtimesP_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(tdy->TtimesPsi_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psi_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(tdy->Psi_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"P_vec.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(tdy->P_vec,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  //#endif
  
  
  // Fluxes
  ierr = TDyMPFAOIFunction_Vertices_Salinity(Ul,R,ctx); CHKERRQ(ierr);


  ierr = VecGetArray(U_t,&du_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscInt c,cStart,cEnd;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&dp_dt);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&dPsi_dt);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&p);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&Psi);CHKERRQ(ierr);

  for (c=0;c<cEnd-cStart;c++) {
    dp_dt[c]    = du_dt[c*2];
    p[c]        = u_p[c*2];
    dPsi_dt[c] = du_dt[c*2+1];
    Psi[c]     = u_p[c*2+1];
  }

  CharacteristicCurve *cc = tdy->cc;
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
    dmass_dP = 0.0; //tdy->rho[icell]     * dporosity_dP         * cc->S[icell] +
      //     tdy->drho_dP[icell] * matprop->porosity[icell] * cc->S[icell] +
      //      tdy->rho[icell]     * matprop->porosity[icell] * cc->dS_dP[icell];
    dmass_dPsi = 0.0; //tdy->rho[icell]     * dporosity_dPsi         * cc->S[icell] +
      //   tdy->drho_dPsi[icell] * matprop->porosity[icell] * cc->S[icell] +
      //  tdy->rho[icell]     * matprop->porosity[icell] * dS_dPsi;

    //CHANGE
    // A_E = [d(rho*phi*s*U)/dP + d(rho*(1-phi)*T)/dP] * dP_dtime *Vol + //change
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

    PetscReal dtrans_dP, dtrans_dPsi;

    dtrans_dP = tdy->drho_dP[icell] * matprop->porosity[icell] * cc->S[icell]     * Psi[icell]     +
             tdy->rho[icell]     * dporosity_dP         * cc->S[icell]     * Psi[icell]     +
             tdy->rho[icell]     * matprop->porosity[icell] * cc->dS_dP[icell] * Psi[icell]     +
             tdy->rho[icell]     * matprop->porosity[icell] * cc->S[icell]     * dPsi_dP;
    dtrans_dPsi = tdy->drho_dPsi[icell] * matprop->porosity[icell] * cc->S[icell]     * Psi[icell]     +
               tdy->rho[icell]     * dporosity_dPsi         * cc->S[icell]     * Psi[icell]     +
               tdy->rho[icell]     * matprop->porosity[icell] * dS_dPsi * Psi[icell]     +
      tdy->rho[icell]     * matprop->porosity[icell] * cc->S[icell];

    r[icell*2]   += dmass_dP * dp_dt[icell] * cells->volume[icell] + dmass_dPsi * dPsi_dt[icell] * cells->volume[icell];
    r[icell*2+1] += dtrans_dP * dp_dt[icell] * cells->volume[icell] + dtrans_dPsi * dPsi_dt[icell] * cells->volume[icell];
    r[icell*2]   -= tdy->source_sink[icell] * cells->volume[icell];
    r[icell*2+1] -= tdy->salinity_source_sink[icell] * cells->volume[icell];
    printf("rho = %f\n",tdy->rho[icell]);
     printf("icell = %d\n",icell);
    //  printf("psi = %f\n",Psi[icell]);
    //  printf("drho_dp = %f\n",tdy->drho_dP[icell]);
    //    printf("drho_dpsi = %f\n",tdy->drho_dPsi[icell]);
  }

  /* Cleanup */
  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr); //chck
  ierr = VecRestoreArray(U_t,&du_dt); CHKERRQ(ierr); //check
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

  //PetscViewer viewer;
// sprintf(word,"Function%d.vec");
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Function.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"dudt.vec",&viewer); CHKERRQ(ierr);
  ierr = VecView(U_t,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  //  icount_f++;


  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

