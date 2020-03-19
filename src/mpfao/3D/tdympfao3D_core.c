#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
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
        K[ii][jj] = tdy->K0[icell*dim*dim + ii*dim + jj];
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

          ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);
          
          ierr = TDyComputeEntryOfGMatrix3D(area, normal, K, nu, subcells->T[subcell_id], dim,
                                         &(tdy->subc_Gmatrix[icell][isubcell][ii][jj]));
          CHKERRQ(ierr);
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeAinvB(PetscInt A_nrow, PetscReal *A,
    PetscInt B_ncol, PetscReal *B, PetscReal *AinvB) {

  PetscFunctionBegin;

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

  ierr = PetscMemcpy(AinvB,B,sizeof(PetscScalar)*(A_nrow*B_ncol));
  CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1 * B) by back-substitution
  m = A_nrow; n = B_ncol;
  LAPACKgetrs_("N", &m, &n, A, &m, pivots, AinvB, &m, &info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeCtimesAinvB(PetscInt A_nrow, PetscInt B_ncol, PetscInt C_nrow,
    PetscReal *C, PetscReal *AinvB, PetscReal *CtimesAinvB) {

  PetscFunctionBegin;

  PetscInt m, n, k;
  PetscScalar zero = 0.0, one = 1.0;

  // Compute (C * AinvB)
  m = C_nrow; n = B_ncol; k = A_nrow;
  BLASgemm_("N","N", &m, &n, &k, &one, C, &m, AinvB, &k, &zero,
            CtimesAinvB, &m);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix_ForInternalVertex(TDy tdy,
    PetscInt ivertex, TDy_cell *cells) {

  PetscInt       ncells, icell, isubcell;

  TDy_vertex *vertices;
  TDy_subcell    *subcells;
  PetscInt dim;
  PetscReal **Fup, **Fdn;
  PetscReal **Cup, **Cdn;
  PetscReal *A_1d, *B_1d, *Cup_1d, *AinvB_1d, *CuptimesAinvB_1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscInt i, j, ndim, nfluxes;

  PetscFunctionBegin;

  vertices = &tdy->mesh->vertices;
  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt vOffsetFace = vertices->face_offset[ivertex];

  ndim      = 3;
  ncells    = vertices->num_internal_cells[ivertex];
  vertex_id = ivertex;
  nfluxes   = ncells * 3/2;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim   , ndim   ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Fup    , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Fdn    , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Cup    , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Cdn    , nfluxes, nfluxes); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&A_1d           , nfluxes*nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&B_1d           , nfluxes*ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&Cup_1d          , nfluxes*nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&AinvB_1d        , nfluxes*ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&CuptimesAinvB_1d, nfluxes*ncells ); CHKERRQ(ierr);

  subcells = &tdy->mesh->subcells;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (i=0; i<ncells; i++) {
    icell    = vertices->internal_cell_ids[vOffsetCell + i];
    isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    PetscInt idx_interface_p0, idx_interface_p1, idx_interface_p2;
    idx_interface_p0 = subcells->face_unknown_idx[sOffsetFace + 0];
    idx_interface_p1 = subcells->face_unknown_idx[sOffsetFace + 1];
    idx_interface_p2 = subcells->face_unknown_idx[sOffsetFace + 2];

    PetscInt iface;
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

      PetscBool upwind_entries;
      PetscInt idx_flux;

      upwind_entries = (subcells->is_face_up[sOffsetFace + iface]==1);

      idx_flux = subcells->face_flux_idx[sOffsetFace +iface];

      if (upwind_entries) {
        Cup[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cup[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cup[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];
        Fup[idx_flux][i]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
      } else {
        Cdn[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cdn[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cdn[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];
        Fdn[idx_flux][i]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
      }
      
    }
  }

  idx = 0;
  for (j=0; j<nfluxes; j++) {
    for (i=0; i<nfluxes; i++) {
      A_1d[idx] = -Cup[i][j] + Cdn[i][j];
      Cup_1d[idx] = Cup[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<nfluxes; i++) {
      B_1d[idx]= -Fup[i][j] + Fdn[i][j];
      idx++;
    }
  }

  // Solve A^-1 * B
  ierr = ComputeAinvB(nfluxes, A_1d, ncells, B_1d, AinvB_1d);

  // Solve C * (A^-1 * B)
  ierr = ComputeCtimesAinvB(nfluxes, ncells, nfluxes, Cup_1d, AinvB_1d, CuptimesAinvB_1d);

  // Save Transmissibility matrix
  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<nfluxes; i++) {
      tdy->Trans[vertex_id][i][j] = CuptimesAinvB_1d[idx] - Fup[i][j];
      if (fabs(tdy->Trans[vertex_id][i][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i][j] = 0.0;
      idx++;
    }
  }

  PetscInt face_id, subface_id, num_subfaces = 4;
  PetscInt row, col;
  for (i=0; i<nfluxes; i++) {
    face_id = vertices->face_ids[vOffsetFace + i];
    subface_id = vertices->subface_ids[vOffsetFace + i];
    row = face_id*num_subfaces + subface_id;
    for (j=0; j<ncells; j++) {
      col = vertices->internal_cell_ids[vOffsetCell + j];
      ierr = MatSetValues(tdy->Trans_mat,1,&row,1,&col,&tdy->Trans[vertex_id][i][j],ADD_VALUES); CHKERRQ(ierr);
    }
  }

  // Free up the memory
  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim   ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn    , nfluxes ); CHKERRQ(ierr);

  free(A_1d            );
  free(B_1d            );
  free(Cup_1d           );
  free(CuptimesAinvB_1d );

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix_ForBoundaryVertex_SharedWithInternalVertices(TDy tdy,
    PetscInt ivertex, TDy_cell *cells) {

  TDy_vertex *vertices;
  TDy_subcell *subcells;
  TDy_face *faces;
  PetscInt nflux_in, nflux_bc_up, nflux_bc_dn, nflux_in_up, nflux_in_dn;
  PetscInt npitf_bc, npitf_in, npitf;
  PetscInt npcen;
  PetscInt icell, isubcell;
  PetscReal **Gmatrix;
  PetscReal **Fup, **Cup, **Fdn, **Cdn;
  PetscReal *AInxIn_1d, *BInxCBC_1d;
  PetscReal *AinvB_1d;
  PetscReal *CupInxIn_1d, *CupBcxIn_1d, *CdnBcxIn_1d;
  PetscReal *CupInxIntimesAinvB_1d, *CupBCxIntimesAinvB_1d, *CdnBCxIntimesAinvB_1d;
  PetscInt idx, vertex_id;
  PetscInt ndim;
  PetscInt i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  vertices = &tdy->mesh->vertices;
  faces = &tdy->mesh->faces;
  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];
  PetscInt vOffsetFace = vertices->face_offset[ivertex];

  subcells = &tdy->mesh->subcells;
  vertex_id = ivertex;

  ierr = DMGetDimension(tdy->dm, &ndim); CHKERRQ(ierr);

  npcen = vertices->num_internal_cells[ivertex];
  nflux_bc_up = 0;
  nflux_bc_dn = 0;
  for (i=0; i<npcen; i++) {
    icell    = vertices->internal_cell_ids[vOffsetCell + i];
    isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    PetscInt iface;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      PetscInt faceID = subcells->face_ids[sOffsetFace+iface];
      if (faces->is_internal[faceID]) continue;

      PetscBool upwind_entries;
      upwind_entries = (subcells->is_face_up[sOffsetFace + iface]==1);

      if (upwind_entries) nflux_bc_up++;
      else                nflux_bc_dn++;
    }

  }

  // Determine:
  //  (1) number of internal and boudnary fluxes,
  //  (2) number of internal unknown pressure values and known boundary pressure values
  npcen = vertices->num_internal_cells[ivertex];
  npitf_bc = vertices->num_boundary_cells[ivertex];
  
  nflux_in = vertices->num_faces[ivertex] - vertices->num_boundary_cells[ivertex];

  npitf_in = nflux_in;
  nflux_in_up = nflux_in + nflux_bc_up;
  nflux_in_dn = nflux_in + nflux_bc_dn;
  npitf = npitf_in + npitf_bc;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim, ndim);

  ierr = TDyAllocate_RealArray_2D(&Fup, nflux_in_up, npcen);
  ierr = TDyAllocate_RealArray_2D(&Cup, nflux_in_up, npitf);
  ierr = TDyAllocate_RealArray_2D(&Fdn, nflux_in_dn, npcen);
  ierr = TDyAllocate_RealArray_2D(&Cdn, nflux_in_dn, npitf);

  ierr = TDyAllocate_RealArray_1D(&AInxIn_1d    , nflux_in*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&CupInxIn_1d  , nflux_in*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&BInxCBC_1d   , nflux_in*(npcen+npitf_bc));

  ierr = TDyAllocate_RealArray_1D(&CupBcxIn_1d, nflux_bc_up*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&CdnBcxIn_1d, nflux_bc_dn*npitf_in);

  ierr = TDyAllocate_RealArray_1D(&AinvB_1d, nflux_in*(npcen+npitf_bc)   );
  ierr = TDyAllocate_RealArray_1D(&CupInxIntimesAinvB_1d, nflux_in*(npcen+npitf_bc)   );
  ierr = TDyAllocate_RealArray_1D(&CupBCxIntimesAinvB_1d, nflux_bc_up*(npcen+npitf_bc)   );
  ierr = TDyAllocate_RealArray_1D(&CdnBCxIntimesAinvB_1d, nflux_bc_dn*(npcen+npitf_bc)   );
  

  for (i=0; i<npcen; i++) {
    icell    = vertices->internal_cell_ids[vOffsetCell + i];
    isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->face_offset[subcell_id];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

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

  /*
    Upwind/Downwind
      Flux     =         F            *     Pcenter      +                         C                         *      P_interface
    _       _     _                  _   _           _       _                                             _    _               _
   |         |   |                    | |             |     |                                               |  |  P_(nptif_in,1) |
   | Flux_in |   | F_(nflux_in,npcen) | |             |     | C_(nflux_in,npitf_in)   C_(nflux_in,npitf_bc) |  |                 |
   |         | = |                    | | P_(npcen,1) |  +  |                                               |  |                 |
   | Flux_bc |   | F_(nflux_bc,npcen) | |             |     | C_(nflux_bc,npitf_in)   C_(nflux_in,npitf_bc) |  |  P_(nptif_bc,1) |
   |_       _|   |_                  _| |_           _|     |_                                             _|  |_               _|

                          C
    _                                             _
   |                                               |
   | C_(nflux_in,npitf_in)   C_(nflux_in,npitf_bc) |
   |                                               |
   | C_(nflux_bc,npitf_in)   C_(nflux_in,npitf_bc) |
   |_                                             _|

    _       _     _                                          _   _             _       _                     _    _               _
   |         |   |                                           | |                |     |                       |  |                 |
   | Flux_in |   | F_(nflux_in,npcen)  C_(nflux_in,npitf_bc) | | P_(npcen,1)    |     | C_(nflux_in,npitf_in) |  |                 |
   |         | = |                                           | |                |  +  |                       |  |  P_(nptif_in,1) |
   | Flux_bc |   | F_(nflux_bc,npcen)  C_(nflux_in,npitf_bc) | | P_(nptif_bc,1) |     | C_(nflux_bc,npitf_in) |  |                 |
   |_       _|   |_                                         _| |_              _|     |_                     _|  |_               _|

  */

  idx = 0;
  for (j=0; j<npcen; j++) {
    for (i=0; i<nflux_in; i++) {
      BInxCBC_1d[idx] = -Fup[i][j]+Fdn[i][j];
      idx++;
    }
  }

  idx = npcen*nflux_in;
  for (j=0; j<npitf_bc; j++) {
    for (i=0; i<nflux_in; i++) {
      BInxCBC_1d[idx] = Cup[i][j+npitf_in] - Cdn[i][j+npitf_in];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_in; i++) {
      AInxIn_1d[idx]   = -Cup[i][j] + Cdn[i][j];
      CupInxIn_1d[idx] = Cup[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_bc_up; i++) {
      CupBcxIn_1d[idx] = Cup[i+npitf_in][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_bc_dn; i++) {
      CdnBcxIn_1d[idx] = Cdn[i+npitf_in][j];
      idx++;
    }
  }

  // Solve A^-1 * B
  ierr = ComputeAinvB(nflux_in, AInxIn_1d, npcen+npitf_bc, BInxCBC_1d, AinvB_1d);

  // Solve C * (A^-1 * B) for internal, upwind, and downwind fluxes
  ierr = ComputeCtimesAinvB(nflux_in, npcen+npitf_bc, nflux_in, CupInxIn_1d, AinvB_1d, CupInxIntimesAinvB_1d);
  ierr = ComputeCtimesAinvB(nflux_in, npcen+npitf_bc, nflux_bc_up, CupBcxIn_1d, AinvB_1d, CupBCxIntimesAinvB_1d);
  ierr = ComputeCtimesAinvB(nflux_in, npcen+npitf_bc, nflux_bc_dn, CdnBcxIn_1d, AinvB_1d, CdnBCxIntimesAinvB_1d);

  // Save transmissiblity matrix for internal fluxes including contribution from unknown P @ cell centers
  // and known P @ boundaries
  idx = 0;
  for (j=0;j<npcen+npitf_bc;j++) {
    for (i=0;i<nflux_in;i++) {
      if (j<npcen) {
        tdy->Trans[vertex_id][i][j] = CupInxIntimesAinvB_1d[idx] - Fup[i][j];
      } else {
        tdy->Trans[vertex_id][i][j] = CupInxIntimesAinvB_1d[idx] + Cup[i][j-npcen+npitf_in];
      }
      if (fabs(tdy->Trans[vertex_id][i][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i][j] = 0.0;
      idx++;
    }
  }

  // Save transmissiblity matrix for boundary fluxes (first upwind and then downwind) including
  // contribution from unknown P @ cell centers and known P @ boundaries
  idx = 0;
  for (j=0;j<npcen+npitf_bc;j++) {
    for (i=0;i<nflux_bc_up;i++) {
      if (j<npcen) {
        tdy->Trans[vertex_id][i+nflux_in][j] = CupBCxIntimesAinvB_1d[idx] - Fup[i+nflux_in][j];
      } else {
        tdy->Trans[vertex_id][i+nflux_in][j] = CupBCxIntimesAinvB_1d[idx] + Cup[i+npitf_in][j-npcen+npitf_in];
      }
      if (fabs(tdy->Trans[vertex_id][i+nflux_in         ][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i+nflux_in         ][j] = 0.0;
      idx++;
    }
  }

  idx = 0;
  for (j=0;j<npcen+npitf_bc;j++) {
    for (i=0;i<nflux_bc_dn;i++) {
      if (j<npcen) {
        tdy->Trans[vertex_id][i+nflux_in+nflux_bc_up][j] = CdnBCxIntimesAinvB_1d[idx] - Fdn[i+nflux_in][j];
      } else {
        tdy->Trans[vertex_id][i+nflux_in+nflux_bc_up][j] = CdnBCxIntimesAinvB_1d[idx] + Cdn[i+npitf_in][j-npcen+npitf_in];
      }
      if (fabs(tdy->Trans[vertex_id][i+nflux_in+nflux_bc_up][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i+nflux_in+nflux_bc_up][j] = 0.0;
      idx++;
    }
  }

  PetscInt face_id, subface_id;
  PetscInt row, col, ncells;
  PetscInt numBnd, idxBnd[npitf_bc];

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

  PetscInt num_subfaces = 4;
  for (i=0; i<nflux_in+nflux_bc_up+nflux_bc_dn; i++) {
    face_id = vertices->face_ids[vOffsetFace + i];
    subface_id = vertices->subface_ids[vOffsetFace + i];
    row = face_id*num_subfaces + subface_id;

    for (j=0; j<npcen; j++) {
      col = vertices->internal_cell_ids[vOffsetCell + j];
      ierr = MatSetValues(tdy->Trans_mat,1,&row,1,&col,&tdy->Trans[vertex_id][i][j],ADD_VALUES); CHKERRQ(ierr);
    }

    for (j=0; j<npitf_bc; j++) {
      col = idxBnd[j] + tdy->mesh->num_cells;
      ierr = MatSetValues(tdy->Trans_mat,1,&row,1,&col,&tdy->Trans[vertex_id][i][j+npcen],ADD_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup, nflux_in_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup, nflux_in_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn, nflux_in_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn, nflux_in_dn); CHKERRQ(ierr);

  free(AinvB_1d);
  free(CupInxIntimesAinvB_1d);
  free(CupBCxIntimesAinvB_1d);
  free(CdnBCxIntimesAinvB_1d);
  free(AInxIn_1d             );
  free(CupInxIn_1d           );
  free(CupBcxIn_1d           );
  free(CdnBcxIn_1d           );

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(TDy tdy,
    PetscInt ivertex, TDy_cell *cells) {
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

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

  ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);
  
  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

    for (j=0; j<dim; j++) {
      tdy->Trans[vertices->id[ivertex]][iface][j] = Gmatrix[iface][j];
      if (fabs(tdy->Trans[vertices->id[ivertex]][iface][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertices->id[ivertex]][iface][j] = 0.0;
    }
    tdy->Trans[vertices->id[ivertex]][iface][dim] = 0.0;
    for (j=0; j<dim; j++) tdy->Trans[vertices->id[ivertex]][iface][dim] -= (Gmatrix[iface][j]);
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
      ierr = MatSetValues(tdy->Trans_mat,1,&row,1,&col,&tdy->Trans[ivertex][i][j],ADD_VALUES); CHKERRQ(ierr);
    }

    icell  = vertices->internal_cell_ids[vOffsetCell + 0];
    col = icell;
    ierr = MatSetValues(tdy->Trans_mat,1,&row,1,&col,&tdy->Trans[ivertex][i][numBnd],ADD_VALUES); CHKERRQ(ierr);
  }

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

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    if (vertices->num_boundary_cells[ivertex] == 0) {
      ierr = ComputeTransmissibilityMatrix_ForInternalVertex(tdy, ivertex, cells);
      CHKERRQ(ierr);
    } else {
      if (vertices->num_internal_cells[ivertex] > 1) {
        ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_SharedWithInternalVertices(tdy, ivertex, cells);
        CHKERRQ(ierr);
      } else {
        ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(tdy, ivertex, cells);
      }
    }
  }

  ierr = MatAssemblyBegin(tdy->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tdy->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
