#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>

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
  (*g) *= -1.0/(T)*area;

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
  
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  faces = mesh->faces;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // set pointer to cell
    TDy_cell *cell;

    cell = &(cells[icell]);

    // extract permeability tensor
    PetscInt ii,jj;
    PetscReal K[3][3];

    for (ii=0; ii<dim; ii++) {
      for (jj=0; jj<dim; jj++) {
        K[ii][jj] = tdy->K[icell*dim*dim + ii*dim + jj];
      }
    }

    PetscInt isubcell;

    for (isubcell=0; isubcell<cell->num_subcells; isubcell++) {

      TDy_subcell *subcell;

      subcell = &cell->subcells[isubcell];

      PetscInt ii,jj;

      for (ii=0;ii<subcell->num_faces;ii++) {

        PetscReal area;
        PetscReal normal[3];

        TDy_face *face = &faces[subcell->face_ids[ii]];
        
        area = subcell->face_area[ii];

        ierr = TDyFace_GetNormal(face, dim, &normal[0]); CHKERRQ(ierr);

        for (jj=0;jj<subcell->num_faces;jj++) {
          PetscReal nu[dim];

          ierr = TDySubCell_GetIthNuVector(subcell, jj, dim, &nu[0]); CHKERRQ(ierr);
          
          ierr = TDyComputeEntryOfGMatrix3D(area, normal, K, nu, subcell->T, dim,
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
PetscErrorCode ComputeTransmissibilityMatrixForInternalVertex3DMesh(TDy tdy,
    TDy_vertex *vertex, TDy_cell *cells) {

  PetscInt       ncells, icell, isubcell;

  TDy_cell  *cell;
  TDy_subcell    *subcell;
  TDy_face *faces;
  PetscInt dim;
  PetscReal **Fup, **Fdn;
  PetscReal **Cup, **Cdn;
  PetscReal **A, **B, **AinvB;
  PetscReal *A1d, *B_1d, *Cup1d, *AinvB_1d, *CuptimesAinvB_1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscInt i, j, ndim, nfluxes;

  PetscFunctionBegin;

  ndim      = 3;
  ncells    = vertex->num_internal_cells;
  vertex_id = vertex->id;
  nfluxes   = ncells * 3/2;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim   , ndim   ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Fup    , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Fdn    , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Cup    , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Cdn    , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&A      , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&B      , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&AinvB  , nfluxes, ncells ); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&A1d            , nfluxes*nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&B_1d            , nfluxes*ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&Cup1d          , nfluxes*nfluxes); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&AinvB_1d        , nfluxes*ncells ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&CuptimesAinvB_1d, nfluxes*ncells ); CHKERRQ(ierr);

  faces = tdy->mesh->faces;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (i=0; i<ncells; i++) {
    icell    = vertex->internal_cell_ids[i];
    isubcell = vertex->subcell_ids[i];

    cell = &cells[icell];
    subcell = &cell->subcells[isubcell];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    PetscInt idx_interface_p0, idx_interface_p1, idx_interface_p2;
    idx_interface_p0 = subcell->face_unknown_idx[0];
    idx_interface_p1 = subcell->face_unknown_idx[1];
    idx_interface_p2 = subcell->face_unknown_idx[2];

    PetscInt iface;
    for (iface=0;iface<subcell->num_faces;iface++) {
      TDy_face *face = &faces[subcell->face_ids[iface]];
      
      PetscInt cell_1 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[0]);
      PetscInt cell_2 = TDyReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[1]);
      PetscBool upwind_entries;

      upwind_entries = (subcell->is_face_up[iface]==1);

      if (upwind_entries && cell_1 != i) {
        PetscInt tmp = cell_1;
        cell_1 = cell_2;
        cell_2 = tmp;
      } else if (!upwind_entries && cell_2 != i) {
        PetscInt tmp = cell_1;
        cell_1 = cell_2;
        cell_2 = tmp;
      }

      PetscInt idx_flux;
      idx_flux = subcell->face_unknown_idx[iface];

      if (upwind_entries) {
        Cup[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cup[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cup[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];
        Fup[idx_flux][cell_1]           = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
      } else {
        Cdn[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cdn[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cdn[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];
        Fdn[idx_flux][cell_2]           = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
      }
      
    }
  }

  idx = 0;
  for (j=0; j<nfluxes; j++) {
    for (i=0; i<nfluxes; i++) {
      A[i][j] = -Cup[i][j] + Cdn[i][j];
      A1d[idx]= -Cup[i][j] + Cdn[i][j];
      Cup1d[idx] = Cup[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<nfluxes; i++) {
      B[i][j] = -Fup[i][j] + Fdn[i][j];
      B_1d[idx]= -Fup[i][j] + Fdn[i][j];
      idx++;
    }
  }

  // Solve A^-1 * B
  ierr = ComputeAinvB(nfluxes, A1d, ncells, B_1d, AinvB_1d);

  // Solve C * (A^-1 * B)
  ierr = ComputeCtimesAinvB(nfluxes, ncells, nfluxes, Cup1d, AinvB_1d, CuptimesAinvB_1d);

  // Save Transmissibility matrix
  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<nfluxes; i++) {
      tdy->Trans[vertex_id][i][j] = CuptimesAinvB_1d[idx] - Fup[i][j];
      idx++;
    }
  }

  // Free up the memory
  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim   ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn    , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(A      , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(B      , nfluxes ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(AinvB  , nfluxes ); CHKERRQ(ierr);

  free(A1d             );
  free(B_1d             );
  free(Cup1d           );
  free(CuptimesAinvB_1d );

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForBoundaryVertex3DMesh(TDy tdy,
    TDy_vertex *vertex, TDy_cell *cells) {

  PetscInt nflux_in, nflux_bc, nflux;
  PetscInt npitf_bc, npitf_in, npitf;
  PetscInt npcen;

  PetscInt icell, isubcell;

  PetscReal **Fup, **Cup, **Fdn, **Cdn;
  PetscReal *BInxCBC_1d;
  PetscReal *AinvB_1d, *CupInxCBCtimesAinvB_1d, *CupBCxCBCtimesAinvB_1d, *CdnBCxCBCtimesAinvB_1d;

  PetscReal *AInxIn_1d;
  PetscReal *CupInxIn_1d;
  PetscReal *CupBcxIn_1d, *CdnBcxIn_1d;

  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscInt ndim;
  PetscInt i,j;
  TDy_cell  *cell;
  TDy_face  *faces, *face;
  TDy_subcell    *subcell;

  PetscFunctionBegin;

  ndim      = 3;
  vertex_id = vertex->id;
  faces = tdy->mesh->faces;

  // Determine:
  //  (1) number of internal and boudnary fluxes,
  //  (2) number of internal unknown pressure values and known boundary pressure values
  npcen = vertex->num_internal_cells;
  npitf_bc = vertex->num_boundary_cells;
  nflux_bc = npitf_bc/2;
  
  switch (npcen) {
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

  npitf_in = nflux_in;
  nflux = nflux_in+nflux_bc;
  npitf = npitf_in+npitf_bc;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim, ndim);

  ierr = TDyAllocate_RealArray_2D(&Fup, nflux, npcen);
  ierr = TDyAllocate_RealArray_2D(&Cup, nflux, npitf);
  ierr = TDyAllocate_RealArray_2D(&Fdn, nflux, npcen);
  ierr = TDyAllocate_RealArray_2D(&Cdn, nflux, npitf);

  ierr = TDyAllocate_RealArray_1D(&AInxIn_1d    , nflux_in*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&CupInxIn_1d  , nflux_in*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&BInxCBC_1d   , nflux_in*(npcen+npitf_bc));

  ierr = TDyAllocate_RealArray_1D(&CupBcxIn_1d, nflux_bc*npitf_in);
  ierr = TDyAllocate_RealArray_1D(&CdnBcxIn_1d, nflux_bc*npitf_in);

  ierr = TDyAllocate_RealArray_1D(&AinvB_1d, nflux_in*(npcen+npitf_bc)   );
  ierr = TDyAllocate_RealArray_1D(&CupInxCBCtimesAinvB_1d, nflux_in*(npcen+npitf_bc)   );
  ierr = TDyAllocate_RealArray_1D(&CupBCxCBCtimesAinvB_1d, nflux_bc*(npcen+npitf_bc)   );
  ierr = TDyAllocate_RealArray_1D(&CdnBCxCBCtimesAinvB_1d, nflux_bc*(npcen+npitf_bc)   );
  

  PetscInt nup_bnd_flux=0, ndn_bnd_flux=0;

  for (i=0; i<npcen; i++) {
    icell    = vertex->internal_cell_ids[i];
    isubcell = vertex->subcell_ids[i];

    cell = &cells[icell];
    subcell = &cell->subcells[isubcell];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    PetscInt idx_interface_p0, idx_interface_p1, idx_interface_p2;

    idx_interface_p0 = subcell->face_unknown_idx[0];
    idx_interface_p1 = subcell->face_unknown_idx[1];
    idx_interface_p2 = subcell->face_unknown_idx[2];

    PetscInt idx_flux, iface;
    for (iface=0; iface<subcell->num_faces; iface++) {
      face = &faces[subcell->face_ids[iface]];

      PetscBool upwind_entries;

      upwind_entries = (subcell->is_face_up[iface]==1);

      idx_flux = subcell->face_unknown_idx[iface];

      if (upwind_entries) {
        if (face->is_internal==0) {
          idx_flux = nup_bnd_flux + nflux_in;
          nup_bnd_flux++;
        }
        Cup[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cup[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cup[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];

        if (npcen==4){
          if (idx_flux<4) {
            Fup[idx_flux][i]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else if (idx_flux == 4) {
            Fup[idx_flux][0]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else {
            Fup[idx_flux][3]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          }
        } else {
          if (idx_flux==0) {
            Fup[idx_flux][i]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else if (idx_flux == 1) {
            Fup[idx_flux][0]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else {
            Fup[idx_flux][1]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          }
        }
      } else {
        if (face->is_internal==0) {
          idx_flux = ndn_bnd_flux + nflux_in;
          ndn_bnd_flux++;
        }
        Cdn[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cdn[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cdn[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];

        if (npcen==4){
          if (idx_flux<4) {
            Fdn[idx_flux][i]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else if (idx_flux == 4) {
            Fdn[idx_flux][1]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else{
            Fdn[idx_flux][2]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          }
        } else {
          if (idx_flux==0) {
            Fdn[idx_flux][i]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else if (idx_flux == 1) {
            Fdn[idx_flux][0]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          } else {
            Fdn[idx_flux][1]                = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];
          }
        }

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
    for (i=0; i<nflux_bc; i++) {
      CupBcxIn_1d[idx] = Cup[i+npitf_in][j];
      CdnBcxIn_1d[idx] = Cdn[i+npitf_in][j];
      idx++;
    }
  }

  // Solve A^-1 * B
  ierr = ComputeAinvB(nflux_in, AInxIn_1d, npcen+npitf_bc, BInxCBC_1d, AinvB_1d);

  // Solve C * (A^-1 * B) for internal, upwind, and downwind fluxes
  ierr = ComputeCtimesAinvB(nflux_in, npcen+npitf_bc, nflux_in, CupInxIn_1d, AinvB_1d, CupInxCBCtimesAinvB_1d);
  ierr = ComputeCtimesAinvB(nflux_in, npcen+npitf_bc, nflux_bc, CupBcxIn_1d, AinvB_1d, CupBCxCBCtimesAinvB_1d);
  ierr = ComputeCtimesAinvB(nflux_in, npcen+npitf_bc, nflux_bc, CdnBcxIn_1d, AinvB_1d, CdnBCxCBCtimesAinvB_1d);

  // Save transmissiblity matrix for internal fluxes including contribution from unknown P @ cell centers
  // and known P @ boundaries
  idx = 0;
  for (j=0;j<npcen+npitf_bc;j++) {
    for (i=0;i<nflux_in;i++) {
      if (j<npcen) {
        tdy->Trans[vertex_id][i][j] = CupInxCBCtimesAinvB_1d[idx] - Fup[i][j];
      } else {
        tdy->Trans[vertex_id][i][j] = CupInxCBCtimesAinvB_1d[idx] + Cup[i][j-npcen+npitf_in];
      }
      idx++;
    }
  }

  // Save transmissiblity matrix for boundary fluxes (first upwind and then downwind) including
  // contribution from unknown P @ cell centers and known P @ boundaries
  idx = 0;
  for (j=0;j<npcen+npitf_bc;j++) {
    for (i=0;i<nflux_bc;i++) {
      if (j<npcen) {
        tdy->Trans[vertex_id][i+nflux_in         ][j] = CupBCxCBCtimesAinvB_1d[idx] - Fup[i+nflux_in][j];
        tdy->Trans[vertex_id][i+nflux_in+nflux_bc][j] = CdnBCxCBCtimesAinvB_1d[idx] - Fdn[i+nflux_in][j];
      } else {
        tdy->Trans[vertex_id][i+nflux_in         ][j] = CupBCxCBCtimesAinvB_1d[idx] + Cup[i+npitf_in][j-npcen+npitf_in];
        tdy->Trans[vertex_id][i+nflux_in+nflux_bc][j] = CdnBCxCBCtimesAinvB_1d[idx] + Cdn[i+npitf_in][j-npcen+npitf_in];
      }
      idx++;
    }
  }

  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup, nflux); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup, nflux); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn, nflux); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn, nflux); CHKERRQ(ierr);

  free(AinvB_1d);
  free(CupInxCBCtimesAinvB_1d);
  free(CupBCxCBCtimesAinvB_1d);
  free(CdnBCxCBCtimesAinvB_1d);
  free(AInxIn_1d             );
  free(CupInxIn_1d           );
  free(CupBcxIn_1d           );
  free(CdnBcxIn_1d           );

  PetscFunctionReturn(0);

}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyComputeTransmissibilityMatrix3DMesh(TDy tdy) {

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    vertex = &vertices[ivertex];

    if (vertex->num_boundary_cells == 0) {
      ierr = ComputeTransmissibilityMatrixForInternalVertex3DMesh(tdy, vertex, cells);
      CHKERRQ(ierr);
    } else {
      if (vertex->num_internal_cells > 1) {
        ierr = ComputeTransmissibilityMatrixForBoundaryVertex3DMesh(tdy, vertex, cells);
        CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_3DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces;
  PetscInt       ivertex, icell_from, icell_to;
  PetscInt       irow, icol, row, col, vertex_id;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;
  faces    = mesh->faces;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;

    if (vertex->num_boundary_cells == 0) {
      PetscInt nflux_in = 12;
      
      for (irow=0; irow<nflux_in; irow++) {
        
        PetscInt face_id = vertex->trans_row_face_ids[irow];
        TDy_face *face = &faces[face_id];
        
        icell_from = face->cell_ids[0];
        icell_to   = face->cell_ids[1];
        
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = vertex->internal_cell_ids[icol];
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          if (col<0) col = -col - 1;
          
          value = tdy->Trans[vertex_id][irow][icol];
          
          row = cells[icell_from].global_id;
          if (cells[icell_from].is_local) {ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);}
          
          row = cells[icell_to].global_id;
          if (cells[icell_to].is_local) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
        }
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces;
  TDy_subcell    *subcell;
  PetscInt       ivertex, icell, isubcell, icell_from, icell_to;
  PetscInt       irow, icol, row, col, vertex_id;
  PetscInt       ncells, ncells_bnd;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscInt npcen, npitf_bc, nflux_bc, nflux_in;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;
  faces    = mesh->faces;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

    /*

    flux                   =              T                 *     P
            _           _     _                           _    _        _
           |  _       _  |   |  _           _     _     _  |  |  _    _  |
           | |         | |   | |             |   |       | |  | |      | |
           | | flux_in | |   | |    T_00     |   | T_01  | |  | | Pcen | |
           | |         | |   | |             |   |       | |  | |      | |
    flux = | |_       _| | = | |_           _|   |_     _| |  | |      | |
           |             |   |                             |  | |_    _| |
           |  _       _  |   |  _           _     _     _  |  |  _    _  |
           | | flux_bc | |   | |    T_10     |   | T_11  | |  | | Pbc  | |
           | |_       _| |   | |_           _|   |_     _| |  | |_    _| |
           |_           _|   |_                           _|  |_        _|

    */

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;

    ncells    = vertex->num_internal_cells;
    ncells_bnd= vertex->num_boundary_cells;

    if (ncells_bnd == 0) continue;
    if (ncells < 2)  continue;
    
    npcen    = vertex->num_internal_cells;
    npitf_bc = vertex->num_boundary_cells;
    nflux_bc = npitf_bc/2;
  
    switch (npcen) {
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

    // Vertex is on the boundary
    
    PetscScalar pBoundary[4];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertex->internal_cell_ids[irow];
      isubcell = vertex->subcell_ids[irow];

      cell = &cells[icell];
      subcell = &cell->subcells[isubcell];

      PetscInt iface;
      for (iface=0;iface<subcell->num_faces;iface++) {

        TDy_face *face = &faces[subcell->face_ids[iface]];
        icell_from = face->cell_ids[0];
        icell_to   = face->cell_ids[1];

        if (face->is_internal == 0) {
          PetscInt f;
          f = face->id + fStart;
          ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
          numBoundary++;
        }
      }
    }

    for (irow=0; irow<nflux_in; irow++){

      PetscInt face_id = vertex->trans_row_face_ids[irow];
      TDy_face *face = &faces[face_id];

      icell_from = face->cell_ids[0];
      icell_to   = face->cell_ids[1];

      if (cells[icell_from].is_local) {
        row   = cells[icell_from].global_id;

        // +T_00
        for (icol=0; icol<npcen; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          value = tdy->Trans[vertex_id][irow][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cells[icell_to].is_local) {

        row   = cells[icell_to].global_id;

        // -T_00
        for (icol=0; icol<npcen; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          value = tdy->Trans[vertex_id][irow][icol];
          ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = tdy->Trans[vertex_id][irow][icol + npcen] *
          pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<nflux_bc*2; irow++) {

      //row = cell_ids_from_to[irow][0];

      PetscInt face_id = vertex->trans_row_face_ids[irow + nflux_in];
      TDy_face *face = &faces[face_id];

      icell_from = face->cell_ids[0];
      icell_to   = face->cell_ids[1];

      if (icell_from>-1 && cells[icell_from].is_local) {

        row   = cells[icell_from].global_id;

        // +T_10
        for (icol=0; icol<npcen; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          value = tdy->Trans[vertex_id][irow+nflux_in][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        
      }
      
      if (icell_to>-1 && cells[icell_to].is_local) {
        row   = cells[icell_to].global_id;
        
        // -T_10
        for (icol=0; icol<npcen; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          value = tdy->Trans[vertex_id][irow+nflux_in][icol];
          {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
        }
        
        //  +T_11 * Pbc
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces;
  TDy_subcell    *subcell;
  PetscInt       ivertex, icell;
  PetscInt       icol, row, col, iface, isubcell;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscReal      sign;
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;
  faces    = mesh->faces;

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    
    vertex    = &vertices[ivertex];
    
    if (vertex->num_boundary_cells == 0) continue;
    if (vertex->num_internal_cells > 1)  continue;
    
    // Vertex is on the boundary
    
    PetscScalar pBoundary[3];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    
    icell    = vertex->internal_cell_ids[0];
    isubcell = vertex->subcell_ids[0];

    cell = &cells[icell];
    subcell = &cell->subcells[isubcell];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);

    numBoundary = 0;
    for (iface=0; iface<subcell->num_faces; iface++) {
      
      TDy_face *face = &faces[subcell->face_ids[iface]];

      PetscInt f;
      f = face->id + fStart;
       ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
       numBoundary++;

    }

    for (iface=0; iface<subcell->num_faces; iface++) {

      TDy_face *face = &faces[subcell->face_ids[iface]];

      row = face->cell_ids[0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;

      value = 0.0;
      for (j=0; j<dim; j++) {
        value += sign*Gmatrix[iface][j];
      }

      row   = cells[icell].global_id;
      col   = cells[icell].global_id;
      if (cells[icell].is_local) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}

    }

    // For fluxes through boundary edges, only add contribution to the vector
    for (iface=0; iface<subcell->num_faces; iface++) {

      TDy_face *face = &faces[subcell->face_ids[iface]];

      row = face->cell_ids[0];
      PetscInt cell_from = face->cell_ids[0];
      PetscInt cell_to   = face->cell_ids[1];

      if (cell_from>-1 && cells[cell_from].is_local) {
        row   = cells[cell_from].global_id;
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = Gmatrix[iface][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }

      }

      if (cell_to>-1 && cells[cell_to].is_local) {
        row   = cells[cell_to].global_id;
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = Gmatrix[iface][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
