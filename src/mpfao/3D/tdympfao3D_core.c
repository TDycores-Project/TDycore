#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdysaturationimpl.h>
#include <private/tdypermeabilityimpl.h>
#include <private/tdympfao3Dutilsimpl.h>

/* ---------------------------------------------------------------- */
PetscErrorCode TDyComputeEntryOfGMatrix3D(PetscReal area, PetscReal n[3],
                                       PetscReal K[3][3], PetscReal v[3],
                                       PetscReal T, PetscInt dim, PetscReal *g) {

  PetscFunctionBegin;

  PetscInt i, j;
  PetscReal Kv[3];

  *g = 0.0;

  for (i=0; i<dim; i++) {
    Kv[i] = 0.0;
    for (j=0; j<dim; j++) {
      Kv[i] += K[i][j] * v[j];
    }
  }

  for (i=0; i<dim; i++) {
    (*g) += n[i] * Kv[i];
  }
  (*g) *= 1.0/(T)*area;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyComputeGMatrixFor3DMesh(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt dim,icell;
  PetscErrorCode ierr;

  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_subcell *subcells;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // extract permeability tensor
    PetscInt ii,jj;
    PetscReal K[3][3];

    for (ii=0; ii<dim; ii++) {
      for (jj=0; jj<dim; jj++) {
        K[ii][jj] = tdy->matprop_K0[icell*dim*dim + ii*dim + jj];
      }
    }

    // extract thermal conductivity tensor
    PetscReal Kappa[3][3];

    if (tdy->mode == TH) {
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = tdy->Kappa0[icell*dim*dim + ii*dim + jj];
        }
      }
    }

    PetscInt isubcell;

    for (isubcell=0; isubcell<cells->num_subcells[icell]; isubcell++) {

      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->face_offset[subcell_id];

      PetscInt ii,jj;

      for (ii=0;ii<subcells->num_faces[subcell_id];ii++) {

        PetscReal area;
        PetscReal normal[3];

        PetscInt face_id = subcells->face_ids[sOffsetFace + ii];

        area = subcells->face_area[sOffsetFace + ii];

        ierr = TDyFace_GetNormal(faces, face_id, dim, &normal[0]); CHKERRQ(ierr);

        for (jj=0;jj<subcells->num_faces[subcell_id];jj++) {
          PetscReal nu[dim];

          switch (tdy->mpfao_gmatrix_method){
          case MPFAO_GMATRIX_DEFAULT:
             ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

             ierr = TDyComputeEntryOfGMatrix3D(area, normal, K, nu, subcells->T[subcell_id], dim,
                                            &(tdy->subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
             break;

          case MPFAO_GMATRIX_TPF:
            if (ii == jj) {
              ierr = TDySubCell_GetIthNuStarVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);
              ierr = TDyComputeEntryOfGMatrix3D(area, normal, K, nu, subcells->T[subcell_id], dim,
                                         &(tdy->subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
            } else {
              tdy->subc_Gmatrix[icell][isubcell][ii][jj] = 0.0;
            }
          }

          if (tdy->mode == TH) {
            switch (tdy->mpfao_gmatrix_method){
            case MPFAO_GMATRIX_DEFAULT:
               ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

              ierr = TDyComputeEntryOfGMatrix3D(area, normal, Kappa,
                                  nu, subcells->T[subcell_id], dim,
                                  &(tdy->Temp_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
               break;

            case MPFAO_GMATRIX_TPF:
               if (ii == jj) {
                ierr = TDySubCell_GetIthNuStarVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

                ierr = TDyComputeEntryOfGMatrix3D(area, normal, Kappa, nu, subcells->T[subcell_id], dim,
                                           &(tdy->Temp_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
                } else {
                  tdy->Temp_subc_Gmatrix[icell][isubcell][ii][jj] = 0.0;
                }
                break;
            }
          }
        }
      }
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeAinvB(PetscInt A_nrow, PetscReal *A,
    PetscInt B_ncol, PetscReal *B, PetscReal *AinvB) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt m, n;
  PetscBLASInt info, *pivots;
  PetscErrorCode ierr;

  m = A_nrow; n = A_nrow;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);

  LAPACKgetrf_(&m, &n, A, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  ierr = PetscMemcpy(AinvB,B,sizeof(PetscReal)*(A_nrow*B_ncol));
  CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1 * B) by back-substitution
  m = A_nrow; n = B_ncol;
  LAPACKgetrs_("N", &m, &n, A, &m, pivots, AinvB, &m, &info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeCtimesAinvB(PetscInt C_nrow, PetscInt AinvB_ncol, PetscInt C_ncol,
    PetscReal *C, PetscReal *AinvB, PetscReal *CtimesAinvB) {

  PetscFunctionBegin;

  PetscInt m, n, k;
  PetscScalar zero = 0.0, one = 1.0;

  // Compute (C * AinvB)
  m = C_nrow; n = AinvB_ncol; k = C_ncol;
  BLASgemm_("N","N", &m, &n, &k, &one, C, &m, AinvB, &k, &zero,
            CtimesAinvB, &m);

  PetscFunctionReturn(0);

}

 /* -------------------------------------------------------------------------- */
PetscErrorCode ComputeCandFmatrix(TDy tdy, PetscInt ivertex, PetscInt varID,
  PetscReal **Cup, PetscReal **Cdn, PetscReal **Fup, PetscReal **Fdn) {

  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  TDy_vertex *vertices = &tdy->mesh->vertices;
  TDy_cell   *cells    = &tdy->mesh->cells;
  TDy_subcell *subcells = &tdy->mesh->subcells;

  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  PetscInt npcen = vertices->num_internal_cells[ivertex];
  PetscReal **Gmatrix;
  PetscInt i, ndim;

  ierr = DMGetDimension(tdy->dm, &ndim); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim   , ndim   ); CHKERRQ(ierr);

  for (i=0; i<npcen; i++) {
    PetscInt icell    = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    if (varID == VAR_PRESSURE) {
      ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);
    } else if (varID == VAR_TEMPERATURE){
      ierr = ExtractTempSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);
    }

    PetscInt idx_interface_p0, idx_interface_p1, idx_interface_p2;

    idx_interface_p0 = subcells->face_unknown_idx[sOffsetFace +0];
    idx_interface_p1 = subcells->face_unknown_idx[sOffsetFace +1];
    idx_interface_p2 = subcells->face_unknown_idx[sOffsetFace +2];

    PetscInt idx_flux, iface;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscBool upwind_entries;

      upwind_entries = (subcells->is_face_up[sOffsetFace + iface]==1);

      if (upwind_entries) {
        idx_flux = subcells->face_flux_idx[sOffsetFace + iface];

        Cup[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cup[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cup[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];

        Fup[idx_flux][i] = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];

      } else {
        idx_flux = subcells->face_flux_idx[sOffsetFace + iface];

        Cdn[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cdn[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cdn[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];

        Fdn[idx_flux][i] = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];

      }
    }

  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode DetermineNumberOfUpAndDownBoundaryFaces(TDy tdy, PetscInt ivertex, PetscInt *nflux_bc_up, PetscInt *nflux_bc_dn) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  TDy_vertex *vertices = &tdy->mesh->vertices;
  TDy_subcell *subcells = &tdy->mesh->subcells;
  TDy_cell *cells = &tdy->mesh->cells;
  TDy_face *faces = &tdy->mesh->faces;

  PetscInt npcen = vertices->num_internal_cells[ivertex];
  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  PetscInt i, iface;

  *nflux_bc_up = 0;
  *nflux_bc_dn = 0;

  for (i=0; i<npcen; i++) {
    PetscInt icell    = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt faceID = subcells->face_ids[sOffsetFace+iface];
      if (faces->is_internal[faceID]) continue;

      if ((subcells->is_face_up[sOffsetFace + iface]==1)) {
        (*nflux_bc_up)++;
      } else {
        (*nflux_bc_dn)++;
      }
    }

  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ExtractSubMatrix(PetscReal **M, PetscInt rStart, PetscInt rEnd,
                                PetscInt cStart, PetscInt cEnd, PetscReal **Msub){

  PetscFunctionBegin;

  PetscInt irow, icol;

  for (irow = rStart; irow < rEnd; irow++) {
    for (icol = cStart; icol < cEnd; icol++) {
      Msub[irow-rStart][icol-cStart] = M[irow][icol];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ExtractsubCMatrices(PetscInt nrow, PetscInt ncol, PetscReal **C,
  PetscInt nrow_1, PetscInt nrow_2, PetscInt nrow_3,
  PetscInt ncol_1, PetscInt ncol_2, PetscInt ncol_3,
  PetscReal ***C_11, PetscReal ***C_12, PetscReal ***C_13,
  PetscReal ***C_21, PetscReal ***C_22, PetscReal ***C_23,
  PetscReal ***C_31, PetscReal ***C_32, PetscReal ***C_33){

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscErrorCode ierr;

  ierr = TDyAllocate_RealArray_2D(C_11, nrow_1, ncol_1); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(C_12, nrow_1, ncol_2); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(C_13, nrow_1, ncol_3); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(C_21, nrow_2, ncol_1); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(C_22, nrow_2, ncol_2); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(C_23, nrow_2, ncol_3); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(C_31, nrow_3, ncol_1); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(C_32, nrow_3, ncol_2); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(C_33, nrow_3, ncol_3); CHKERRQ(ierr);

  ierr = ExtractSubMatrix(C, 0, nrow_1, 0            , ncol_1              , *C_11); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(C, 0, nrow_1, ncol_1       , ncol_1+ncol_2       , *C_12); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(C, 0, nrow_1, ncol_1+ncol_2, ncol_1+ncol_2+ncol_3, *C_13); CHKERRQ(ierr);

  ierr = ExtractSubMatrix(C, nrow_1, nrow_1+nrow_2, 0            , ncol_1              , *C_21); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(C, nrow_1, nrow_1+nrow_2, ncol_1       , ncol_1+ncol_2       , *C_22); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(C, nrow_1, nrow_1+nrow_2, ncol_1+ncol_2, ncol_1+ncol_2+ncol_3, *C_23); CHKERRQ(ierr);

  ierr = ExtractSubMatrix(C, nrow_1+nrow_2, nrow_1+nrow_2+nrow_3, 0            , ncol_1              , *C_31); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(C, nrow_1+nrow_2, nrow_1+nrow_2+nrow_3, ncol_1       , ncol_1+ncol_2       , *C_32); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(C, nrow_1+nrow_2, nrow_1+nrow_2+nrow_3, ncol_1+ncol_2, ncol_1+ncol_2+ncol_3, *C_33); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ExtractsubFMatrices(PetscReal **F, PetscInt nrow, PetscInt ncol, PetscInt nrow_1, PetscInt nrow_2, PetscInt nrow_3,
PetscReal ***F_1, PetscReal ***F_2, PetscReal ***F_3){

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscErrorCode ierr;

  ierr = TDyAllocate_RealArray_2D(F_1 , nrow_1, ncol); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(F_2 , nrow_2, ncol); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(F_3 , nrow_3, ncol); CHKERRQ(ierr);

  ierr = ExtractSubMatrix(F, 0, nrow_1, 0, ncol, *F_1); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(F, nrow_1, nrow_1 + nrow_2, 0, ncol, *F_2); CHKERRQ(ierr);
  ierr = ExtractSubMatrix(F, nrow_1 + nrow_2, nrow_1 + nrow_2 + nrow_3, 0, ncol, *F_3); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix_ForNonCornerVertex(TDy tdy,
    PetscInt ivertex, TDy_cell *cells, PetscInt varID) {

  TDy_vertex *vertices;
  TDy_subcell *subcells;
  TDy_face *faces;
  PetscInt icell;
  PetscReal **Gmatrix;
  PetscReal **Fup_all, **Cup_all, **Fdn_all, **Cdn_all;
  PetscReal *AINBCxINBC_1d, *BINBCxCDBC_1d;
  PetscReal *AinvB_1d;
  PetscReal *CupINBCxINBC_1d, *CupDBCxIn_1d, *CdnDBCxIn_1d;
  PetscReal *CupInxIntimesAinvB_1d, *CupBCxIntimesAinvB_1d, *CdnBCxIntimesAinvB_1d;
  PetscInt idx, vertex_id;
  PetscInt ndim;
  PetscInt i,j;
  PetscReal ****Trans;
  Mat *Trans_mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  vertices = &tdy->mesh->vertices;
  faces = &tdy->mesh->faces;
  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt vOffsetFace = vertices->face_offset[ivertex];

  subcells = &tdy->mesh->subcells;
  vertex_id = ivertex;

  ierr = DMGetDimension(tdy->dm, &ndim); CHKERRQ(ierr);

  PetscInt npcen        = vertices->num_internal_cells[ivertex];
  PetscInt npitf_bc_all = vertices->num_boundary_faces[ivertex];

  PetscInt nflux_all_bc_up, nflux_all_bc_dn;
  PetscInt nflux_dir_bc_up, nflux_dir_bc_dn;
  PetscInt nflux_neu_bc_up, nflux_neu_bc_dn;
  ierr = DetermineNumberOfUpAndDownBoundaryFaces(tdy, ivertex, &nflux_all_bc_up, &nflux_all_bc_dn);

  PetscInt npitf_dir_bc_all, npitf_neu_bc_all;

  if (tdy->mpfao_bc_type == MPFAO_DIRICHLET_BC) {
    nflux_dir_bc_up = nflux_all_bc_up;
    nflux_dir_bc_dn = nflux_all_bc_dn;
    npitf_dir_bc_all= npitf_bc_all;

    nflux_neu_bc_up = 0;
    nflux_neu_bc_dn = 0;
    npitf_neu_bc_all= 0;
  } else {
    nflux_dir_bc_up = 0;
    nflux_dir_bc_dn = 0;
    npitf_dir_bc_all= 0;

    nflux_neu_bc_up = nflux_all_bc_up;
    nflux_neu_bc_dn = nflux_all_bc_dn;
    npitf_neu_bc_all= npitf_bc_all;
  }

  PetscInt nflux_neu_bc_all = npitf_neu_bc_all;

  PetscInt nptif_dir_bc = nflux_dir_bc_up + nflux_dir_bc_dn;
  PetscInt nptif_neu_bc = nflux_neu_bc_up + nflux_neu_bc_dn;

  // Determine:
  //  (1) number of internal and boudnary fluxes,
  //  (2) number of internal unknown pressure values and known boundary pressure values

  PetscInt nflux_in = vertices->num_faces[ivertex] - nptif_dir_bc - nptif_neu_bc;

  PetscInt nflux_in_plus_bcs_up = nflux_in + nflux_dir_bc_up + nflux_neu_bc_up;
  PetscInt nflux_in_plus_bcs_dn = nflux_in + nflux_dir_bc_dn + nflux_neu_bc_dn;

  PetscInt npitf_in = nflux_in;
  PetscInt npitf = npitf_in + npitf_bc_all;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim, ndim);

  ierr = TDyAllocate_RealArray_2D(&Fup_all, nflux_in_plus_bcs_up, npcen);
  ierr = TDyAllocate_RealArray_2D(&Cup_all, nflux_in_plus_bcs_up, npitf);
  ierr = TDyAllocate_RealArray_2D(&Fdn_all, nflux_in_plus_bcs_dn, npcen);
  ierr = TDyAllocate_RealArray_2D(&Cdn_all, nflux_in_plus_bcs_dn, npitf);

  ierr = TDyAllocate_RealArray_1D(&AINBCxINBC_1d  , (nflux_in+nflux_neu_bc_all)*(npitf_in + nflux_neu_bc_all));
  ierr = TDyAllocate_RealArray_1D(&BINBCxCDBC_1d  , (nflux_in+nflux_neu_bc_all)*(npcen    + npitf_dir_bc_all));
  ierr = TDyAllocate_RealArray_1D(&CupINBCxINBC_1d, (nflux_in+nflux_neu_bc_all)*(npitf_in + npitf_neu_bc_all));

  ierr = TDyAllocate_RealArray_1D(&CupDBCxIn_1d, nflux_dir_bc_up*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&CdnDBCxIn_1d, nflux_dir_bc_dn*npitf_in);

  ierr = TDyAllocate_RealArray_1D(&AinvB_1d             , (nflux_in + nflux_neu_bc_all)*(npcen + npitf_dir_bc_all) );
  ierr = TDyAllocate_RealArray_1D(&CupInxIntimesAinvB_1d, (nflux_in + nflux_neu_bc_all)*(npcen + npitf_dir_bc_all) );
  ierr = TDyAllocate_RealArray_1D(&CupBCxIntimesAinvB_1d, nflux_dir_bc_up              *(npcen + npitf_dir_bc_all) );
  ierr = TDyAllocate_RealArray_1D(&CdnBCxIntimesAinvB_1d, nflux_dir_bc_dn              *(npcen + npitf_dir_bc_all) );


  ierr = ComputeCandFmatrix(tdy, ivertex, varID, Cup_all, Cdn_all, Fup_all, Fdn_all); CHKERRQ(ierr);

  PetscReal **Fup_in, **Fup_dir_bc, **Fup_neu_bc;
  PetscReal **Fdn_in, **Fdn_dir_bc, **Fdn_neu_bc;

  ierr = ExtractsubFMatrices(Fup_all, nflux_in_plus_bcs_up, npcen, nflux_in, nflux_dir_bc_up, nflux_neu_bc_up, &Fup_in, &Fup_dir_bc, &Fup_neu_bc);
  ierr = ExtractsubFMatrices(Fdn_all, nflux_in_plus_bcs_up, npcen, nflux_in, nflux_dir_bc_dn, nflux_neu_bc_dn, &Fdn_in, &Fdn_dir_bc, &Fdn_neu_bc);

  PetscReal **Cup_11, **Cup_12, **Cup_13, **Cup_21, **Cup_22, **Cup_23, **Cup_31, **Cup_32, **Cup_33;
  PetscReal **Cdn_11, **Cdn_12, **Cdn_13, **Cdn_21, **Cdn_22, **Cdn_23, **Cdn_31, **Cdn_32, **Cdn_33;

  ierr = ExtractsubCMatrices(nflux_in_plus_bcs_up, npitf, Cup_all,
    nflux_in, nflux_dir_bc_up, nflux_neu_bc_up, // rows
    npitf_in, npitf_dir_bc_all, nflux_neu_bc_all, // cols
    &Cup_11, &Cup_12, &Cup_13,
    &Cup_21, &Cup_22, &Cup_23,
    &Cup_31, &Cup_32, &Cup_33);

  ierr = ExtractsubCMatrices(nflux_in_plus_bcs_up, npitf, Cdn_all,
    nflux_in, nflux_dir_bc_dn, nflux_neu_bc_dn, // rows
    npitf_in, npitf_dir_bc_all, nflux_neu_bc_all, // cols
    &Cdn_11, &Cdn_12, &Cdn_13,
    &Cdn_21, &Cdn_22, &Cdn_23,
    &Cdn_31, &Cdn_32, &Cdn_33);

  idx = 0;
  for (j=0; j<npcen; j++) {
    for (i=0; i<nflux_in; i++) {
      BINBCxCDBC_1d[idx] = -Fup_in[i][j]+Fdn_in[i][j];
      idx++;
    }
    for (i=0; i<nflux_neu_bc_up; i++) {
      BINBCxCDBC_1d[idx] = -Fup_neu_bc[i][j];
      idx++;
    }
    for (i=0; i<nflux_neu_bc_dn; i++) {
      BINBCxCDBC_1d[idx] = -Fdn_neu_bc[i][j];
      idx++;
    }
  }

  for (j=0; j< npitf_dir_bc_all; j++) {
    for (i=0; i<nflux_in; i++) {
      BINBCxCDBC_1d[idx] = Cup_12[i][j] - Cdn_12[i][j];
      idx++;
    }
    for (i=0; i<nflux_neu_bc_up; i++) {
      BINBCxCDBC_1d[idx] = -Cup_32[i][j];
      idx++;
    }
    for (i=0; i<nflux_neu_bc_dn; i++) {
      BINBCxCDBC_1d[idx] = -Cdn_32[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_in; i++) {
      AINBCxINBC_1d[idx]   = -Cup_11[i][j] + Cdn_11[i][j];
      CupINBCxINBC_1d[idx] = Cup_11[i][j];
      idx++;
    }

    for (i=0; i<nflux_neu_bc_up; i++) {
      AINBCxINBC_1d[idx]   = -Cup_31[i][j];
      CupINBCxINBC_1d[idx] = Cup_31[i][j];
      idx++;
    }

    for (i=0; i<nflux_neu_bc_dn; i++) {
      AINBCxINBC_1d[idx]   = -Cdn_31[i][j];
      CupINBCxINBC_1d[idx] = Cdn_31[i][j];
      idx++;
    }

  }

  for (j=0; j<nflux_neu_bc_all; j++) {
    for (i=0; i<nflux_in; i++) {
      AINBCxINBC_1d[idx]   = -Cup_13[i][j] + Cdn_13[i][j];
      CupINBCxINBC_1d[idx] = Cup_13[i][j];
      idx++;
    }

    for (i=0; i<nflux_neu_bc_up; i++) {
      AINBCxINBC_1d[idx]   = -Cup_33[i][j];
      CupINBCxINBC_1d[idx] = Cup_33[i][j];
      idx++;
    }

    for (i=0; i<nflux_neu_bc_dn; i++) {
      AINBCxINBC_1d[idx]   = -Cdn_33[i][j];
      CupINBCxINBC_1d[idx] = Cdn_33[i][j];
      idx++;
    }
  }


  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_dir_bc_up; i++) {
      CupDBCxIn_1d[idx] = Cup_21[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_dir_bc_dn; i++) {
      CdnDBCxIn_1d[idx] = Cdn_21[i][j];
      idx++;
    }
  }

  // Solve A^-1 * B
  ierr = ComputeAinvB(nflux_in + nflux_neu_bc_all, AINBCxINBC_1d, npcen+npitf_dir_bc_all, BINBCxCDBC_1d, AinvB_1d);

  // Solve C * (A^-1 * B) for internal, upwind, and downwind fluxes
  ierr = ComputeCtimesAinvB(
    nflux_in + nflux_neu_bc_all, // nrow for CupINBCxINBC_1d
    npcen    + npitf_dir_bc_all, // ncol for AinvB
    npitf_in + npitf_neu_bc_all, // ncol for CupINBCxINBC_1d
    CupINBCxINBC_1d, AinvB_1d, CupInxIntimesAinvB_1d); CHKERRQ(ierr);

  if (nflux_dir_bc_up > 0) {
    ierr = ComputeCtimesAinvB(
      nflux_dir_bc_up             , // nrow for CupDBCxIn_1d
      npcen    + npitf_dir_bc_all , // ncol for AinvB
      npitf_in                    , // ncol CupDBCxIn_1d
      CupDBCxIn_1d, AinvB_1d, CupBCxIntimesAinvB_1d);  CHKERRQ(ierr);
    }

  if (nflux_dir_bc_dn > 0) {
    ierr = ComputeCtimesAinvB(
      nflux_dir_bc_dn             , // nrow for CdnDBCxIn_1d
      npcen    + npitf_dir_bc_all , // ncol for AinvB
      npitf_in                    , // nrow for CdnDBCxIn_1d
      CdnDBCxIn_1d, AinvB_1d, CdnBCxIntimesAinvB_1d); CHKERRQ(ierr);
    }

  if (varID == VAR_PRESSURE) {
    Trans = &tdy->Trans;
    Trans_mat = &tdy->Trans_mat;
  } else if (varID == VAR_TEMPERATURE) {
    Trans = &tdy->Temp_Trans;
    Trans_mat = &tdy->Temp_Trans_mat;
  }

  // Save transmissiblity matrix for internal fluxes including contribution from unknown P @ cell centers
  // and known P @ boundaries
  idx = 0;
  for (j=0;j<npcen+npitf_dir_bc_all;j++) {
    for (i=0;i<nflux_in;i++) {
      if (j<npcen) {
        (*Trans)[vertex_id][i][j] = CupInxIntimesAinvB_1d[idx] - Fup_in[i][j];
      } else {
        (*Trans)[vertex_id][i][j] = CupInxIntimesAinvB_1d[idx] + Cup_12[i][j-npcen];
      }
      idx++;
    }
    idx = idx + npitf_neu_bc_all;
  }

  // Save transmissiblity matrix for boundary fluxes (first upwind and then downwind) including
  // contribution from unknown P @ cell centers and known P @ boundaries
  idx = 0;
  for (j=0;j<npcen+npitf_dir_bc_all;j++) {
    for (i=0;i<nflux_dir_bc_up;i++) {
      if (j<npcen) {
        (*Trans)[vertex_id][i+nflux_in][j] = CupBCxIntimesAinvB_1d[idx] - Fup_dir_bc[i][j];
      } else {
        (*Trans)[vertex_id][i+nflux_in][j] = CupBCxIntimesAinvB_1d[idx] + Cup_22[i][j-npcen];
      }
      idx++;
    }
  }

  idx = 0;
  for (j=0;j<npcen+npitf_dir_bc_all;j++) {
    for (i=0;i<nflux_dir_bc_dn;i++) {
      if (j<npcen) {
        (*Trans)[vertex_id][i+nflux_in+nflux_dir_bc_up][j] = CdnBCxIntimesAinvB_1d[idx] - Fdn_dir_bc[i][j];
      } else {
        (*Trans)[vertex_id][i+nflux_in+nflux_dir_bc_up][j] = CdnBCxIntimesAinvB_1d[idx] + Cdn_22[i][j-npcen];
      }
      idx++;
    }
  }

  PetscInt face_id, subface_id;
  PetscInt row, col, ncells;
  PetscInt numBnd, idxBnd[npitf_dir_bc_all];

  ncells = vertices->num_internal_cells[ivertex];
  numBnd = 0;
  if (npitf_dir_bc_all>0){
    for (i=0; i<ncells; i++) {
      icell = vertices->internal_cell_ids[vOffsetCell + i];
      PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->face_offset[subcell_id];

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

	PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
	PetscInt fOffsetCell = faces->cell_offset[face_id];

	if (faces->is_internal[face_id] == 0) {

	  // Extract pressure value at the boundary
	  if (faces->cell_ids[fOffsetCell + 0] >= 0) idxBnd[numBnd] = -faces->cell_ids[fOffsetCell + 1] - 1;
	  else                                       idxBnd[numBnd] = -faces->cell_ids[fOffsetCell + 0] - 1;

	  numBnd++;
	}
      }
    }
  }

  PetscInt num_subfaces = 4;
  for (i=0; i<nflux_in+nflux_dir_bc_up+nflux_dir_bc_dn; i++) {
    face_id = vertices->face_ids[vOffsetFace + i];
    subface_id = vertices->subface_ids[vOffsetFace + i];
    row = face_id*num_subfaces + subface_id;

    for (j=0; j<npcen; j++) {
      col = vertices->internal_cell_ids[vOffsetCell + j];
      ierr = MatSetValues(*Trans_mat,1,&row,1,&col,&(*Trans)[vertex_id][i][j],ADD_VALUES); CHKERRQ(ierr);
    }

    for (j=0; j<npitf_dir_bc_all; j++) {
      col = idxBnd[j] + tdy->mesh->num_cells;
      ierr = MatSetValues(*Trans_mat,1,&row,1,&col,&(*Trans)[vertex_id][i][j+npcen],ADD_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup_all, nflux_in_plus_bcs_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_all, nflux_in_plus_bcs_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn_all, nflux_in_plus_bcs_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_all, nflux_in_plus_bcs_dn); CHKERRQ(ierr);

  ierr = TDyDeallocate_RealArray_2D(Fup_in, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup_dir_bc, nflux_dir_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup_neu_bc, nflux_neu_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn_in, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn_dir_bc, nflux_dir_bc_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn_neu_bc, nflux_neu_bc_dn); CHKERRQ(ierr);

  free(AinvB_1d);
  free(CupInxIntimesAinvB_1d);
  free(CupBCxIntimesAinvB_1d);
  free(CdnBCxIntimesAinvB_1d);
  free(AINBCxINBC_1d             );
  free(CupINBCxINBC_1d           );
  free(CupDBCxIn_1d           );
  free(CdnDBCxIn_1d           );

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(TDy tdy,
    PetscInt ivertex, TDy_cell *cells, PetscInt varID) {
  DM             dm;
  TDy_vertex     *vertices;
  TDy_subcell    *subcells;
  TDy_face       *faces;
  PetscInt       icell;
  PetscInt       iface, isubcell;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscInt       j;
  PetscReal      ****Trans;
  Mat            *Trans_mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  dm       = tdy->dm;
  subcells = &tdy->mesh->subcells;
  vertices = &tdy->mesh->vertices;
  faces = &tdy->mesh->faces;

  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // Vertex is on the boundary

  // For boundary edges, save following information:
  //  - Dirichlet pressure value
  //  - Cell IDs connecting the boundary edge in the direction of unit normal
  PetscInt subcell_id;

  icell    = vertices->internal_cell_ids[vOffsetCell + 0];
  isubcell = vertices->subcell_ids[vOffsetSubcell + 0];
  subcell_id = icell*cells->num_subcells[icell]+isubcell;

  if (varID == VAR_PRESSURE) {
    Trans = &tdy->Trans;
    Trans_mat = &tdy->Trans_mat;
    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);
  } else if (varID == VAR_TEMPERATURE) {
    ierr = ExtractTempSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);
    Trans = &tdy->Temp_Trans;
    Trans_mat = &tdy->Temp_Trans_mat;
  }

  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

    for (j=0; j<dim; j++) {
      (*Trans)[vertices->id[ivertex]][iface][j] = Gmatrix[iface][j];
    }
    (*Trans)[vertices->id[ivertex]][iface][dim] = 0.0;
    for (j=0; j<dim; j++) (*Trans)[vertices->id[ivertex]][iface][dim] -= (Gmatrix[iface][j]);
  }


  PetscInt i, face_id, subface_id;
  PetscInt row, col, ncells;
  PetscInt numBnd, idxBnd[3];
  PetscInt vOffsetFace;

  ncells = vertices->num_internal_cells[ivertex];
  numBnd = 0;
  for (i=0; i<ncells; i++) {
    icell = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    PetscInt iface;
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->cell_offset[face_id];

      if (faces->is_internal[face_id] == 0) {

        // Extract pressure value at the boundary
        if (faces->cell_ids[fOffsetCell + 0] >= 0) idxBnd[numBnd] = -faces->cell_ids[fOffsetCell + 1] - 1;
        else                                       idxBnd[numBnd] = -faces->cell_ids[fOffsetCell + 0] - 1;

        numBnd++;
      }
    }
  }

  icell    = vertices->internal_cell_ids[vOffsetCell + 0];
  isubcell = vertices->subcell_ids[vOffsetSubcell + 0];
  subcell_id = icell*cells->num_subcells[icell]+isubcell;
  vOffsetFace = vertices->face_offset[ivertex];

  for (i=0; i<subcells->num_faces[subcell_id]; i++) {
    face_id = vertices->face_ids[vOffsetFace + i];
    subface_id = vertices->subface_ids[vOffsetFace + i];
    row = face_id * 4 + subface_id;

    for (j=0; j<numBnd; j++) {
      col = idxBnd[j] + tdy->mesh->num_cells;
      ierr = MatSetValues(*Trans_mat,1,&row,1,&col,&(*Trans)[ivertex][i][j],ADD_VALUES); CHKERRQ(ierr);
    }

    icell  = vertices->internal_cell_ids[vOffsetCell + 0];
    col = icell;
    ierr = MatSetValues(*Trans_mat,1,&row,1,&col,&(*Trans)[ivertex][i][numBnd],ADD_VALUES); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyUpdateTransmissibilityMatrix(TDy tdy) {

  TDy_mesh       *mesh = tdy->mesh;
  TDy_face       *faces = &mesh->faces;
  TDyRegion     *region = &mesh->region_connected;
  PetscInt       iface, isubface;
  PetscInt       num_subfaces = 4;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()  

  // If a face is shared by two cells that belong to different
  // regions, zero the rows in the transmissiblity matrix

  for (iface=0; iface<tdy->mesh->num_faces; iface++) {

    PetscInt fOffsetCell = faces->cell_offset[iface];
    PetscInt cell_id_up = faces->cell_ids[fOffsetCell + 0];
    PetscInt cell_id_dn = faces->cell_ids[fOffsetCell + 1];

    if (cell_id_up>=0 && cell_id_dn>=0) {
      if (!TDyRegionAreCellsInTheSameRegion(region, cell_id_up, cell_id_dn)) {
        for (isubface=0; isubface<4; isubface++) {
          PetscInt row[1];
          row[0] = iface*num_subfaces + isubface;
          ierr = MatZeroRows(tdy->Trans_mat,1,row,0.0,0,0); CHKERRQ(ierr);
          if (tdy->mode == TH) {
            ierr = MatZeroRows(tdy->Temp_Trans_mat,1,row,0.0,0,0); CHKERRQ(ierr);
          }
        }
      }
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyComputeTransmissibilityMatrix3DMesh(TDy tdy) {

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    if (vertices->num_boundary_faces[ivertex] == 0 || vertices->num_internal_cells[ivertex] > 1) {
      ierr = ComputeTransmissibilityMatrix_ForNonCornerVertex(tdy, ivertex, cells, 0); CHKERRQ(ierr);
      if (tdy->mode == TH) {
        ierr = ComputeTransmissibilityMatrix_ForNonCornerVertex(tdy, ivertex, cells, 1); CHKERRQ(ierr);
      }
    } else {
      // It is assumed that neumann boundary condition is a zero-flux boundary condition.
      // Thus, compute transmissiblity entries only for dirichlet boundary condition.
      if (tdy->mpfao_bc_type == MPFAO_DIRICHLET_BC) {
        ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(tdy, ivertex, cells, 0); CHKERRQ(ierr);
        if (tdy->mode == TH) {
          ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(tdy, ivertex, cells, 1); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = MatAssemblyBegin(tdy->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tdy->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (tdy->mode == TH) {
    ierr = MatAssemblyBegin(tdy->Temp_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tdy->Temp_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  TDyRegion *region = &mesh->region_connected;
  if (region->num_cells > 0){
    if (tdy->mpfao_gmatrix_method == MPFAO_GMATRIX_TPF ) {
      ierr = TDyUpdateTransmissibilityMatrix(tdy); CHKERRQ(ierr);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"WARNING -- Connected region option is only supported with MPFA-O TPF\n");
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}
