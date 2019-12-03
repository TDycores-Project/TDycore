#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdysaturationimpl.h>
#include <private/tdypermeabilityimpl.h>

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

      //TDy_subcell *subcell;
      //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

      PetscInt ii,jj;

      for (ii=0;ii<subcells->num_faces[subcell_id];ii++) {

        PetscReal area;
        PetscReal normal[3];

        //TDy_face *face = &faces[subcell->face_ids[ii]];
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
  TDy_face *faces;
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
  PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
  PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

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

  faces = &tdy->mesh->faces;
  subcells = &tdy->mesh->subcells;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (i=0; i<ncells; i++) {
    icell    = vertices->internal_cell_ids[offsetCell + i];
    isubcell = vertices->subcell_ids[offsetSubcell + i];

    //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    PetscInt idx_interface_p0, idx_interface_p1, idx_interface_p2;
    idx_interface_p0 = subcells->face_unknown_idx[sOffsetFace + 0];
    idx_interface_p1 = subcells->face_unknown_idx[sOffsetFace + 1];
    idx_interface_p2 = subcells->face_unknown_idx[sOffsetFace + 2];

    PetscInt iface;
    for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {
      //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      PetscInt cell_1 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 0]);
      PetscInt cell_2 = TDyReturnIndexInList(&vertices->internal_cell_ids[offsetCell], ncells, faces->cell_ids[fOffsetCell + 1]);
      PetscBool upwind_entries;

      upwind_entries = (subcells->is_face_up[sOffsetFace + iface]==1);

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
      idx_flux = subcells->face_unknown_idx[sOffsetFace +iface];

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
  PetscInt nflux_in, nflux_bc, nflux;
  PetscInt npitf_bc, npitf_in, npitf;
  PetscInt npcen;
  PetscInt icell, isubcell;
  PetscReal **Gmatrix;
  PetscReal **Fup, **Cup, **Fdn, **Cdn;
  PetscReal *AInxIn_1d, *BInxCBC_1d;
  PetscReal *AinvB_1d;
  PetscReal *CupInxIn_1d, *CupBcxIn_1d, *CdnBcxIn_1d;
  PetscReal *CupInxCBCtimesAinvB_1d, *CupBCxCBCtimesAinvB_1d, *CdnBCxCBCtimesAinvB_1d;
  PetscInt idx, vertex_id;
  PetscInt ndim;
  PetscInt i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  vertices = &tdy->mesh->vertices;
  PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
  PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

  subcells = &tdy->mesh->subcells;
  vertex_id = ivertex;

  ierr = DMGetDimension(tdy->dm, &ndim); CHKERRQ(ierr);

  // Determine:
  //  (1) number of internal and boudnary fluxes,
  //  (2) number of internal unknown pressure values and known boundary pressure values
  npcen = vertices->num_internal_cells[ivertex];
  npitf_bc = vertices->num_boundary_cells[ivertex];
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
  

  for (i=0; i<npcen; i++) {
    icell    = vertices->internal_cell_ids[offsetCell + i];
    isubcell = vertices->subcell_ids[offsetSubcell + i];

    //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

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
      if (fabs(tdy->Trans[vertex_id][i][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i][j] = 0.0;
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
      if (fabs(tdy->Trans[vertex_id][i+nflux_in         ][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i+nflux_in         ][j] = 0.0;
      if (fabs(tdy->Trans[vertex_id][i+nflux_in+nflux_bc][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertex_id][i+nflux_in+nflux_bc][j] = 0.0;
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
PetscErrorCode ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(TDy tdy,
    PetscInt ivertex, TDy_cell *cells) {
  DM             dm;
  TDy_vertex     *vertices;
  TDy_subcell    *subcells;
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

  PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
  PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // Vertex is on the boundary
  
  // For boundary edges, save following information:
  //  - Dirichlet pressure value
  //  - Cell IDs connecting the boundary edge in the direction of unit normal
  
  icell    = vertices->internal_cell_ids[offsetCell + 0];
  isubcell = vertices->subcell_ids[offsetSubcell + 0];
  
  //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
  PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

  ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);
  
  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {
    
    for (j=0; j<dim; j++) {
      tdy->Trans[vertices->id[ivertex]][iface][j] = Gmatrix[iface][j];
      if (fabs(tdy->Trans[vertices->id[ivertex]][iface][j])<PETSC_MACHINE_EPSILON) tdy->Trans[vertices->id[ivertex]][iface][j] = 0.0;
    }
    tdy->Trans[vertices->id[ivertex]][iface][dim] = 0.0;
    for (j=0; j<dim; j++) tdy->Trans[vertices->id[ivertex]][iface][dim] -= (Gmatrix[iface][j]);
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

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_3DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  PetscInt       ivertex, cell_id_up, cell_id_dn;
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
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;
    PetscInt offsetFace = vertices->offset_face_ids[ivertex];
    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
  
    if (vertices->num_boundary_cells[ivertex] == 0) {
      PetscInt nflux_in = 12;
      
      for (irow=0; irow<nflux_in; irow++) {
        
        PetscInt face_id = vertices->face_ids[offsetFace + irow];
        //TDy_face *face = &faces[face_id];
        PetscInt fOffsetCell = faces->offset_cell_ids[face_id];
        
        cell_id_up = faces->cell_ids[fOffsetCell + 0];
        cell_id_dn = faces->cell_ids[fOffsetCell + 1];
        
        for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
          col   = vertices->internal_cell_ids[offsetCell + icol];
          col   = cells->global_id[vertices->internal_cell_ids[offsetCell + icol]];
          if (col<0) col = -col - 1;
          
          value = -tdy->Trans[vertex_id][irow][icol];
          
          row = cells->global_id[cell_id_up];
          if (cells->is_local[cell_id_up]) {ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);}
          
          row = cells->global_id[cell_id_dn];
          if (cells->is_local[cell_id_dn]) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
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
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
  PetscInt       ivertex, icell, isubcell, cell_id_up, cell_id_dn;
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
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

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

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells < 2)  continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];
    PetscInt offsetFace    = vertices->offset_face_ids[ivertex];

    npcen    = vertices->num_internal_cells[ivertex];
    npitf_bc = vertices->num_boundary_cells[ivertex];
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
      icell = vertices->internal_cell_ids[offsetCell + irow];
      isubcell = vertices->subcell_ids[offsetSubcell + irow];

      //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
        PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
        PetscInt fOffsetCell = faces->offset_cell_ids[face_id];
        cell_id_up = faces->cell_ids[fOffsetCell + 0];
        cell_id_dn = faces->cell_ids[fOffsetCell + 1];

        if (faces->is_internal[face_id] == 0) {
          PetscInt f;
          f = faces->id[face_id] + fStart;
          ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
          numBoundary++;
        }
      }
    }

    for (irow=0; irow<nflux_in; irow++){

      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      if (cells->is_local[cell_id_up]) {
        row   = cells->global_id[cell_id_up];

        // +T_00
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[offsetCell + icol]];
          value = -tdy->Trans[vertex_id][irow][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = -tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cells->is_local[cell_id_dn]) {

        row   = cells->global_id[cell_id_dn];

        // -T_00
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[offsetCell + icol]];
          value = -tdy->Trans[vertex_id][irow][icol];
          ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = -tdy->Trans[vertex_id][irow][icol + npcen] *
          pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<nflux_bc*2; irow++) {

      //row = cell_ids_from_to[irow][0];

      PetscInt face_id = vertices->face_ids[offsetFace + irow + nflux_in];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {

        row   = cells->global_id[cell_id_up];

        // +T_10
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[offsetCell + icol]];
          value = -tdy->Trans[vertex_id][irow+nflux_in][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = -tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }

      }
      
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) {
        row   = cells->global_id[cell_id_dn];
        
        // -T_10
        for (icol=0; icol<npcen; icol++) {
          col   = cells->global_id[vertices->internal_cell_ids[offsetCell + icol]];
          value = -tdy->Trans[vertex_id][irow+nflux_in][icol];
          {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
        }

        //  +T_11 * Pbc
        for (icol=0; icol<vertices->num_boundary_cells[ivertex]; icol++) {
          value = -tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
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
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
  PetscInt       ivertex, icell;
  PetscInt       icol, row, col, iface, isubcell;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      sign;
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    
    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

    // Vertex is on the boundary
    
    PetscScalar pBoundary[3];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    
    icell    = vertices->internal_cell_ids[offsetCell + 0];
    isubcell = vertices->subcell_ids[offsetSubcell + 0];

    //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

    numBoundary = 0;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {
      
      //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];

      PetscInt f;
      f = faces->id[face_id] + fStart;
       ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
       numBoundary++;

    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      row = faces->cell_ids[fOffsetCell + 0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;

      value = 0.0;
      for (j=0; j<dim; j++) {
        value += sign*(-tdy->Trans[vertices->id[ivertex]][iface][j]);
      }

      row   = cells->global_id[icell];
      col   = cells->global_id[icell];
      if (cells->is_local[icell]) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}

    }

    // For fluxes through boundary edges, only add contribution to the vector
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      row = faces->cell_ids[fOffsetCell + 0];
      PetscInt cell_id_up = faces->cell_ids[fOffsetCell + 0];
      PetscInt cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {
        row   = cells->global_id[cell_id_up];
        for (icol=0; icol<vertices->num_boundary_cells[ivertex]; icol++) {
          value = -tdy->Trans[vertices->id[ivertex]][iface][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }

      }

      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) {
        row   = cells->global_id[cell_id_dn];
        for (icol=0; icol<vertices->num_boundary_cells[ivertex]; icol++) {
          value = -tdy->Trans[vertices->id[ivertex]][iface][icol] * pBoundary[icol];
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

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGtimesZ(PetscReal *gravity, PetscReal *X, PetscInt dim, PetscReal *gz) {

  PetscInt d;

  PetscFunctionBegin;
  
  *gz = 0.0;
  if (dim == 3) {
    for (d=0;d<dim;d++) *gz += fabs(gravity[d])*X[d];
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyUpdateBoundaryState(TDy tdy) {

  TDy_mesh *mesh;
  TDy_face *faces;
  PetscErrorCode ierr;
  PetscReal Se,dSe_dS,dKr_dSe,n=0.5,m=0.8,alpha=1.e-4,Kr; /* FIX: generalize */
  PetscInt dim;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal Sr,S,dS_dP,d2S_dP2,P;

  PetscFunctionBegin;

  mesh = tdy->mesh;
  faces = &mesh->faces;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

    PetscInt fOffsetCell = faces->offset_cell_ids[iface];

    if (faces->cell_ids[fOffsetCell + 0] >= 0) {
      cell_id = faces->cell_ids[fOffsetCell + 0];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
    } else {
      cell_id = faces->cell_ids[fOffsetCell + 1];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;
    }

    switch (tdy->SatFuncType[cell_id]) {
    case SAT_FUNC_GARDNER :
      Sr = tdy->Sr[cell_id];
      P = tdy->Pref - tdy->P_BND[p_bnd_idx];

      PressureSaturation_Gardner(n,m,alpha,Sr,P,&S,&dS_dP,&d2S_dP2);
      break;
    case SAT_FUNC_VAN_GENUCHTEN :
      Sr = tdy->Sr[cell_id];
      P = tdy->Pref - tdy->P_BND[p_bnd_idx];

      PressureSaturation_VanGenuchten(m,alpha,Sr,P,&S,&dS_dP,&d2S_dP2);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown saturation function");
      break;
    }

    Se = (S - Sr)/(1.0 - Sr);
    dSe_dS = 1.0/(1.0 - Sr);

    switch (tdy->RelPermFuncType[cell_id]) {
    case REL_PERM_FUNC_IRMAY :
      RelativePermeability_Irmay(m,Se,&Kr,NULL);
      break;
    case REL_PERM_FUNC_MUALEM :
      RelativePermeability_Mualem(m,Se,&Kr,&dKr_dSe);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown relative permeability function");
      break;
    }

    tdy->S_BND[p_bnd_idx] = S;
    tdy->dS_dP_BND[p_bnd_idx] = dS_dP;
    tdy->d2S_dP2_BND[p_bnd_idx] = d2S_dP2;
    tdy->Kr_BND[p_bnd_idx] = Kr;
    tdy->dKr_dS_BND[p_bnd_idx] = dKr_dSe * dSe_dS;

    //for(j=0; j<dim2; j++) tdy->K[i*dim2+j] = tdy->K0[i*dim2+j] * Kr;
  }
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices_3DMesh(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
  PetscInt       ivertex, cell_id_up;
  PetscInt       irow, icol, vertex_id;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscScalar    *u;
  Vec            localU;
  PetscReal      vel_normal;
  PetscReal      X[3], vel[3];
  PetscReal      gz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] == 0) {

      PetscInt    nflux_in = 12;
      PetscScalar Pcomputed[vertices->num_internal_cells[ivertex]];
      PetscScalar Vcomputed[nflux_in];

      PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
      PetscInt offsetFace    = vertices->offset_face_ids[ivertex];

      // Save local pressure stencil and initialize veloctiy
      PetscInt icell, cell_id;
      for (icell=0; icell<vertices->num_internal_cells[ivertex]; icell++) {
        cell_id = vertices->internal_cell_ids[offsetCell + icell];
        ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz);
        Pcomputed[icell] = u[cell_id] + tdy->rho[cell_id]*gz;
      }

      for (irow=0; irow<nflux_in; irow++) {
        Vcomputed[irow] = 0.0;
      }

      // F = T*P
      for (irow=0; irow<nflux_in; irow++) {

        PetscInt face_id = vertices->face_ids[offsetFace + irow];
        //TDy_face *face = &faces[face_id];
        PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

        if (!faces->is_local[face_id]) continue;

        cell_id_up = faces->cell_ids[fOffsetCell + 0];

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

        for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {

          Vcomputed[irow] += -tdy->Trans[vertex_id][irow][icol]*Pcomputed[icol];
          
        }
        tdy->vel[face_id] += Vcomputed[irow];
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (Vcomputed[icell] - vel_normal), 2.0);
          (*count)++;
        }

      }
    }
  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
  PetscInt       ivertex, icell, cell_id_up, cell_id_dn;
  PetscInt       irow, icol, vertex_id;
  PetscInt       ncells, ncells_bnd;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  Vec            localU;
  PetscScalar    *u;
  PetscInt npcen, npitf_bc, nflux_bc, nflux_in;
  PetscReal       vel_normal;
  PetscReal       X[3], vel[3];
  PetscReal       gz=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

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

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells < 2)  continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];
    PetscInt offsetFace    = vertices->offset_face_ids[ivertex];

    npcen    = vertices->num_internal_cells[ivertex];
    npitf_bc = vertices->num_boundary_cells[ivertex];
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
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertices->internal_cell_ids[offsetCell + irow];
      PetscInt isubcell = vertices->subcell_ids[offsetSubcell + irow];

      //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
        PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
        PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

        cell_id_up = faces->cell_ids[fOffsetCell + 0];
        cell_id_dn = faces->cell_ids[fOffsetCell + 1];

        if (faces->is_internal[face_id] == 0) {
          PetscInt f;
          f = faces->id[face_id] + fStart;
          if (tdy->ops->computedirichletvalue) {
            ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
          } else {
            ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[icell].X,dim,&gz); CHKERRQ(ierr);
            pBoundary[numBoundary] = u[icell] + tdy->rho[icell]*gz;
          }
          numBoundary++;
        }
      }
    }

    for (irow=0; irow<nflux_in; irow++){

      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      if (!faces->is_local[face_id]) continue;

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      icell = vertices->internal_cell_ids[offsetCell + irow];

      if (cells->is_local[cell_id_up]) {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

        // +T_00 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[offsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[offsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[offsetCell + icol]]*gz;

          value += -tdy->Trans[vertex_id][irow][icol]*Pcomputed;
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      } else {
      
        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_dn, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

        // -T_00 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[offsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[offsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[offsetCell + icol]]*gz;

          value += -tdy->Trans[vertex_id][irow][icol]*Pcomputed;
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;

        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }
      }
    }
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<nflux_bc*2; irow++) {

      PetscInt face_id = vertices->face_ids[offsetFace + irow + nflux_in];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      if (!faces->is_local[face_id]) continue;

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];

      if (cell_id_up>-1 && cells->is_local[cell_id_up]) {

        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells, cell_id_up, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

        // +T_10 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[offsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[offsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[offsetCell + icol]]*gz;
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol]*Pcomputed;
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;
        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      } else {
      
        PetscInt iface=-1;
        PetscInt subcell_id;

        ierr = TDyFindSubcellOfACellThatIncludesAVertex(cells,cell_id_dn, vertices, vertex_id, subcells, &subcell_id); CHKERRQ(ierr);
        ierr = TDySubCell_GetFaceIndexForAFace(subcells, subcell_id, face_id, &iface); CHKERRQ(ierr);
        PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

        // -T_10 * Pcen
        value = 0.0;
        for (icol=0; icol<npcen; icol++) {
          ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[vertices->internal_cell_ids[offsetCell + icol]].X,dim,&gz); CHKERRQ(ierr);
          PetscReal Pcomputed = u[cells->id[vertices->internal_cell_ids[offsetCell + icol]]] + tdy->rho[vertices->internal_cell_ids[offsetCell + icol]]*gz;
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol]*Pcomputed;
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }

        //  +T_11 * Pbc
        for (icol=0; icol<vertices->num_boundary_cells[ivertex]; icol++) {
          value += -tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        tdy->vel[face_id] += value;
        tdy->vel_count[face_id]++;
        ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
        if (tdy->ops->computedirichletflux) {
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
          vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

          *vel_error += PetscPowReal( (value - vel_normal), 2.0);
          (*count)++;
        }

      }
    }
  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy tdy, Vec U, PetscReal *vel_error, PetscInt *count) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;
  TDy_face       *faces;
  TDy_subcell    *subcells;
  PetscInt       ivertex, icell;
  PetscInt       row, iface, isubcell;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscReal      sign;
  PetscInt       j;
  PetscScalar    *u;
  Vec            localU;
  PetscReal      vel_normal;
  PetscReal      X[3], vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = &mesh->vertices;
  faces    = &mesh->faces;
  subcells = &mesh->subcells;

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    
    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;
    
    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

    // Vertex is on the boundary
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    
    icell    = vertices->internal_cell_ids[offsetCell + 0];
    isubcell = vertices->subcell_ids[offsetSubcell + 0];

    //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
    PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

    PetscScalar pBoundary[subcells->num_faces[subcell_id]];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {
      
      //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];

      PetscInt f;
      f = face_id + fStart;
      if (tdy->ops->computedirichletvalue) {
        ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[iface], tdy->dirichletvaluectx);CHKERRQ(ierr);
      } else {
        pBoundary[iface] = u[icell];
      }

    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      if (!faces->is_local[face_id]) continue;

      row = faces->cell_ids[fOffsetCell + 0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;

      value = 0.0;
      for (j=0; j<dim; j++) {
        value -= sign*Gmatrix[iface][j]*(pBoundary[j] - u[icell])/subcells->face_area[sOffsetFace + iface];
      }

      //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
      // Should it be '-value' or 'value'?
      value = sign*value;
      tdy->vel[face_id] += value;
      tdy->vel_count[face_id]++;
      ierr = TDySubCell_GetIthFaceCentroid(subcells, subcell_id, iface, dim, X); CHKERRQ(ierr);
      if (tdy->ops->computedirichletflux) {
        ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
        vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim)*subcells->face_area[sOffsetFace + iface];

        *vel_error += PetscPowReal( (value - vel_normal), 2.0);
        (*count)++;
      }
    }

  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAORecoverVelocity_3DMesh(TDy tdy, Vec U) {

  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscReal vel_error = 0.0;
  PetscInt count = 0;

  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, U, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, U, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, U, &vel_error, &count); CHKERRQ(ierr);

  PetscReal vel_error_sum;
  PetscInt  count_sum;

  ierr = MPI_Allreduce(&vel_error,&vel_error_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&count,&count_sum,1,MPI_INT,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);

  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank==0) printf("%15.14f ",PetscPowReal(vel_error_sum/count_sum,0.5));

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_InternalVertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex, vertex_id;
  PetscInt dim;
  PetscInt irow, icol;
  PetscInt cell_id_up, cell_id_dn, cell_id, icell;
  PetscReal gz,den,fluxm,ukvr;
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

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] != 0) continue;
    PetscInt offsetFace = vertices->offset_face_ids[ivertex];
    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];

    PetscInt nflux_in = 12;
    PetscScalar P[vertices->num_internal_cells[ivertex]];
    PetscScalar TtimesP[nflux_in];

    // Build the P vector
    for (icell=0; icell<vertices->num_internal_cells[ivertex]; icell++) {
      cell_id = vertices->internal_cell_ids[offsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Compute = T*P
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //face = &faces[face_id];

      if (!faces->is_local[face_id]) continue;
      
      TtimesP[irow] = 0.0;

      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    //
    // fluxm_ij = rho_ij * (kr/mu)_{ij,upwind} * [ T ] *  [ P+rho*g*z ]^T
    // where
    //      rho_ij = 0.5*(rho_i + rho_j)
    //      (kr/mu)_{ij,upwind} = (kr/mu)_{i} if velocity is from i to j
    //                          = (kr/mu)_{j} otherwise
    //      T includes product of K and A_{ij}
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];
      
      if (TtimesP[irow] < 0.0) ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
      else                     ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
      
      den = 0.5*(tdy->rho[cell_id_up] + tdy->rho[cell_id_dn]);
      fluxm = den*ukvr*(-TtimesP[irow]);
      
      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cells->is_local[cell_id_up]) r[cell_id_up] += fluxm;
      if (cells->is_local[cell_id_dn]) r[cell_id_dn] -= fluxm;
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex, vertex_id;
  PetscInt dim;
  PetscInt ncells, ncells_bnd;
  PetscInt npcen, npitf_bc, nflux_bc, nflux_in;
  TDy_subcell    *subcells;
  PetscInt irow, icol;
  PetscInt cell_id_up, cell_id_dn, cell_id, icell;
  PetscReal gz,den,fluxm,ukvr;
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

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells     <  2) continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];
    PetscInt offsetFace    = vertices->offset_face_ids[ivertex];

    npcen    = vertices->num_internal_cells[ivertex];
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

    PetscScalar pBoundary[4];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertices->internal_cell_ids[offsetCell + irow];
      PetscInt isubcell = vertices->subcell_ids[offsetSubcell + irow];

      //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        //face = &faces[subcell->face_ids[iface]];
        PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
        PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

        if (faces->is_internal[face_id] == 0) {

          // Extract pressure value at the boundary
          PetscInt p_bnd_idx;
          if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
          else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

          pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];

          numBoundary++;
        }
      }
    }
    
    PetscReal P[npcen + npitf_bc];

    // Save intenral pressure values
    for (icell=0;icell<npcen;icell++) {
      cell_id = vertices->internal_cell_ids[offsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) {
      P[icell+npcen] = pBoundary[icell];
    }

    // Compute T*P
    PetscScalar TtimesP[nflux_in + 2*nflux_bc];
    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<npcen + npitf_bc; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

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

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      fluxm = den*ukvr*(-TtimesP[irow]);
      
      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up>-1 && cells->is_local[cell_id_up]) r[cell_id_up] += fluxm;
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) r[cell_id_dn] -= fluxm;

    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Vec Ul, Vec R, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscReal *p,*r;
  PetscInt ivertex, vertex_id;
  PetscInt dim;
  TDy_subcell    *subcells;
  PetscInt irow, icol;
  PetscInt isubcell, iface;
  PetscInt cell_id_up, cell_id_dn, cell_id, icell;
  PetscReal gz,den,fluxm,ukvr;
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

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

    // Vertex is on the boundary
    PetscScalar pBoundary[3];
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    cell_id  = vertices->internal_cell_ids[offsetCell + 0];
    isubcell = vertices->subcell_ids[offsetSubcell + 0];

    //subcell = &subcells[cell_id*cells->num_subcells[cell_id]+isubcell];
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

    numBoundary = 0;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //face = &faces[subcell->face_ids[iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      // Extract pressure value at the boundary
      PetscInt p_bnd_idx;
      if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
      else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

      pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];
      numBoundary++;

    }

    PetscReal P[numBoundary+1];

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) P[icell] = pBoundary[icell];

    // Save internal pressure value
    gz = 0.0;
    ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
    P[numBoundary] = p[cell_id] + tdy->rho[cell_id]*gz;

    // Compute T*P
    PetscScalar TtimesP[numBoundary];
    for (irow=0; irow<numBoundary; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<numBoundary+1; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //face = &faces[subcell->face_ids[iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      cell_id_up = faces->cell_ids[fOffsetCell + 0];
      cell_id_dn = faces->cell_ids[fOffsetCell + 1];


      if (TtimesP[irow] < 0.0) { // up ---> dn
        if (cell_id_up>=0) ukvr = tdy->Kr[cell_id_up]/tdy->vis[cell_id_up];
        else               ukvr = tdy->Kr_BND[-cell_id_up-1]/tdy->vis_BND[-cell_id_up-1];
      } else {
        if (cell_id_dn>=0) ukvr = tdy->Kr[cell_id_dn]/tdy->vis[cell_id_dn];
        else               ukvr = tdy->Kr_BND[-cell_id_dn-1]/tdy->vis_BND[-cell_id_dn-1];
      }

      den = 0.0;
      if (cell_id_up>=0) den += tdy->rho[cell_id_up];
      else               den += tdy->rho_BND[-cell_id_up-1];
      if (cell_id_dn>=0) den += tdy->rho[cell_id_dn];
      else               den += tdy->rho_BND[-cell_id_dn-1];
      den *= 0.5;

      fluxm = den*ukvr*(-TtimesP[irow]);

      // fluxm > 0 implies flow is from 'up' to 'dn'
      if (cell_id_up>-1 && cells->is_local[cell_id_up]) r[cell_id_up] += fluxm;
      if (cell_id_dn>-1 && cells->is_local[cell_id_dn]) r[cell_id_dn] -= fluxm;
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDy tdy, Vec Ul) {

  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  PetscErrorCode ierr;
  PetscInt dim;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal *p, gz;

  PetscFunctionBegin;

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);

  mesh = tdy->mesh;
  cells = &mesh->cells;
  faces = &mesh->faces;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    //face = &faces[iface];

    if (faces->is_internal[iface]) continue;

      PetscInt fOffsetCell = faces->offset_cell_ids[iface];

    if (faces->cell_ids[fOffsetCell + 0] >= 0) {
      cell_id = faces->cell_ids[fOffsetCell + 0];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
    } else {
      cell_id = faces->cell_ids[fOffsetCell + 1];
      p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;
    }

    if (tdy->ops->computedirichletvalue) {
      ierr = (*tdy->ops->computedirichletvalue)(tdy, (faces->centroid[iface].X), &(tdy->P_BND[p_bnd_idx]), tdy->dirichletvaluectx);CHKERRQ(ierr);
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      tdy->P_BND[p_bnd_idx] += tdy->rho[cell_id]*gz;
    } else {
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      tdy->P_BND[p_bnd_idx] = p[cell_id] + tdy->rho[cell_id]*gz;
    }
  }

  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {
  
  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*dp_dt,*r;
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
  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);

  PetscReal vel_error = 0.0;
  PetscInt count = 0;
  PetscInt iface;

  for (iface=0;iface<mesh->num_faces;iface++) tdy->vel[iface] = 0.0;
  
  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);

  ierr = TDyMPFAOIFunction_InternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscReal dporosity_dP = 0.0;
  PetscReal dmass_dP;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dP * dP/dt * Vol
    dmass_dP = tdy->rho[icell]     * dporosity_dP         * tdy->S[icell] +
               tdy->drho_dP[icell] * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dP[icell];
    r[icell] += dmass_dP * dp_dt[icell] * cells->volume[icell];
    r[icell] -= tdy->source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_InternalVertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt ivertex, vertex_id;
  PetscInt icell, cell_id, cell_id_up, cell_id_dn;
  PetscInt irow, icol;
  PetscInt dim;
  PetscReal gz;
  PetscReal *p;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal T;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  faces    = &mesh->faces;
  vertices = &mesh->vertices;
  dm       = tdy->dm;

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] != 0) continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetFace    = vertices->offset_face_ids[ivertex];

    PetscInt nflux_in = 12;
    PetscScalar P[vertices->num_internal_cells[ivertex]];
    PetscScalar TtimesP[nflux_in];

    // Build the P vector
    for (icell=0; icell<vertices->num_internal_cells[ivertex]; icell++) {
      cell_id = vertices->internal_cell_ids[offsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);

      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Compute = T*P
    for (irow=0; irow<nflux_in; irow++) {
      
      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      
      if (!faces->is_local[face_id]) continue;
      
      TtimesP[irow] = 0.0;

      for (icol=0; icol<vertices->num_internal_cells[ivertex]; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
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
      
      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

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
        cell_id = vertices->internal_cell_ids[offsetCell + icol];
        
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

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt ncells, ncells_bnd;
  PetscInt npcen, npitf_bc, nflux_bc, nflux_in;
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

  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    ncells    = vertices->num_internal_cells[ivertex];
    ncells_bnd= vertices->num_boundary_cells[ivertex];

    if (ncells_bnd == 0) continue;
    if (ncells     <  2) continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];
    PetscInt offsetFace    = vertices->offset_face_ids[ivertex];

    npcen    = vertices->num_internal_cells[ivertex];
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

    PetscScalar pBoundary[4];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertices->internal_cell_ids[offsetCell + irow];
      PetscInt isubcell = vertices->subcell_ids[offsetSubcell + irow];

      //subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];
      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;
      PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

      PetscInt iface;
      for (iface=0;iface<subcells->num_faces[subcell_id];iface++) {

        //TDy_face *face = &faces[subcells->face_ids[sOffsetFace + iface]];
        PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
        PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

        if (faces->is_internal[face_id] == 0) {

          // Extract pressure value at the boundary
          PetscInt p_bnd_idx;
          if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
          else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

          pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];

          numBoundary++;
        }
      }
    }
    
    PetscReal P[npcen + npitf_bc];

    // Save intenral pressure values
    for (icell=0;icell<npcen;icell++) {
      cell_id = vertices->internal_cell_ids[offsetCell + icell];
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
      P[icell] = p[cell_id] + tdy->rho[cell_id]*gz;
    }

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) {
      P[icell+npcen] = pBoundary[icell];
    }

    // Compute T*P
    PetscScalar TtimesP[nflux_in + 2*nflux_bc];
    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<npcen + npitf_bc; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (irow=0; irow<nflux_in + 2*nflux_bc; irow++) {

      PetscInt face_id = vertices->face_ids[offsetFace + irow];
      //TDy_face *face = &faces[face_id];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

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
        cell_id = vertices->internal_cell_ids[offsetCell + icol];
        
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

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Vec Ul, Mat A, void *ctx) {

  TDy tdy = (TDy)ctx;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;
  TDy_vertex *vertices;
  DM dm;
  PetscInt fStart, fEnd;
  TDy_subcell    *subcells;
  PetscInt dim;
  PetscInt icell, ivertex;
  PetscInt isubcell, iface;
  PetscInt cell_id, cell_id_up, cell_id_dn, vertex_id;
  PetscInt irow, icol;
  PetscReal T;
  PetscReal ukvr, den;
  PetscReal dukvr_dPup, dukvr_dPdn, Jac;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal *p;
  PetscReal gz;
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

  ierr = DMPlexGetDepthStratum( dm, 2, &fStart, &fEnd); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex_id = ivertex;

    if (vertices->num_boundary_cells[ivertex] == 0) continue;
    if (vertices->num_internal_cells[ivertex] > 1)  continue;

    PetscInt offsetCell    = vertices->offset_internal_cell_ids[ivertex];
    PetscInt offsetSubcell = vertices->offset_subcell_ids[ivertex];

    // Vertex is on the boundary
    PetscScalar pBoundary[3];
    PetscInt numBoundary;

    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal

    cell_id  = vertices->internal_cell_ids[offsetCell + 0];
    isubcell = vertices->subcell_ids[offsetSubcell + 0];

    //subcell = &subcells[cell_id*cells->num_subcells[cell_id]+isubcell];
    PetscInt subcell_id = cell_id*cells->num_subcells[cell_id]+isubcell;
    PetscInt sOffsetFace = subcells->offset_faces[subcell_id];

    numBoundary = 0;
    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //face = &faces[subcell->face_ids[iface]];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

      // Extract pressure value at the boundary
      PetscInt p_bnd_idx;
      if (faces->cell_ids[fOffsetCell + 0] >= 0) p_bnd_idx = -faces->cell_ids[fOffsetCell + 1] - 1;
      else                        p_bnd_idx = -faces->cell_ids[fOffsetCell + 0] - 1;

      pBoundary[numBoundary] = tdy->P_BND[p_bnd_idx];
      numBoundary++;

    }

    PetscReal P[numBoundary+1];

    // Save boundary pressure values
    for (icell=0;icell<numBoundary;icell++) P[icell] = pBoundary[icell];

    // Save internal pressure value
    gz = 0.0;
    ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[cell_id].X,dim,&gz); CHKERRQ(ierr);
    P[numBoundary] = p[cell_id] + tdy->rho[cell_id]*gz;

    // Compute T*P
    PetscScalar TtimesP[numBoundary];
    for (irow=0; irow<numBoundary; irow++) {

      TtimesP[irow] = 0.0;

      for (icol=0; icol<numBoundary+1; icol++) {
        TtimesP[irow] += tdy->Trans[vertex_id][irow][icol]*P[icol];
      }
      if (fabs(TtimesP[irow])<PETSC_MACHINE_EPSILON) TtimesP[irow] = 0.0;
    }

    for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

      //face = &faces[subcell->face_ids[iface]];
      PetscInt sOffsetFace = subcells->offset_faces[subcell_id];
      PetscInt face_id = subcells->face_ids[sOffsetFace + iface];
      PetscInt fOffsetCell = faces->offset_cell_ids[face_id];

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

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian_Accumulation_3DMesh(Vec Ul,Vec Udotl,PetscReal shift,Mat A,void *ctx) {

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
PetscErrorCode TDyMPFAOIJacobian_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

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

  ierr = TDyMPFAOIJacobian_InternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh(Ul, A, ctx);
  //ierr = TDyMPFAOIJacobian_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_Accumulation_3DMesh(Ul, Udotl, shift, A, ctx);

  if (A !=B ) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESAccumulation(TDy tdy, PetscInt icell, PetscReal *accum) {

  PetscFunctionBegin;

  TDy_cell *cells = &tdy->mesh->cells;

  *accum = tdy->rho[icell] * tdy->porosity[icell] * tdy->S[icell] * cells->volume[icell] / tdy->dtime;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESPreSolve_3DMesh(TDy tdy) {

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  PetscReal *p, *accum_prev;
  PetscInt icell;
  PetscErrorCode ierr;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;

  // Update the auxillary variables
  ierr = VecGetArray(tdy->soln_prev,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_prev,&p); CHKERRQ(ierr);

  ierr = VecGetArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dt * Vol
    //  = [(rho*phi*s)^{t+1} - (rho*phi*s)^t]/dt * Vol
    //  = accum_current - accum_prev
    ierr = TDyMPFAOSNESAccumulation(tdy,icell,&accum_prev[icell]); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESFunction_3DMesh(SNES snes,Vec U,Vec R,void *ctx) {
  
  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*r;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;

  //ierr = SNESGetDM(snes,&dm); CHKERRQ(ierr);
  dm = tdy->dm;

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

  PetscReal vel_error = 0.0;
  PetscInt count = 0;
  PetscInt iface;
  PetscReal *accum_prev;

  for (iface=0;iface<mesh->num_faces;iface++) tdy->vel[iface] = 0.0;
  
  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);

  ierr = TDyMPFAOIFunction_InternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);

  PetscReal accum_current;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dt * Vol
    //  = [(rho*phi*s)^{t+1} - (rho*phi*s)^t]/dt * Vol
    //  = accum_current - accum_prev
    ierr = TDyMPFAOSNESAccumulation(tdy,icell,&accum_current); CHKERRQ(ierr);

    r[icell] += accum_current - accum_prev[icell];
    r[icell] -= tdy->source_sink[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESJacobian_3DMesh(SNES snes,Vec U,Mat A,Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  Vec Ul, Udotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;

  dm = tdy->dm;

  ierr = MatZeroEntries(A); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Udotl); CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);

  ierr = TDyMPFAOIJacobian_InternalVertices_3DMesh(Ul, A, ctx);
  ierr = TDyMPFAOIJacobian_BoundaryVertices_SharedWithInternalVertices_3DMesh(Ul, A, ctx);

  PetscReal dtInv = 1.0/tdy->dtime;

  PetscReal dporosity_dP = 0.0;
  PetscReal dmass_dP, Jac;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d/dP ( d(rho*phi*s)/dt * Vol )
    //  = d/dP [(rho*phi*s)^{t+1} - (rho*phi*s)^t]/dt * Vol
    //  = d/dP [(rho*phi*s)^{t+1}]
    dmass_dP = tdy->rho[icell]     * dporosity_dP         * tdy->S[icell] +
               tdy->drho_dP[icell] * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell]     * tdy->porosity[icell] * tdy->dS_dP[icell];
    Jac = dmass_dP * cells->volume[icell] * dtInv;

    ierr = MatSetValuesLocal(A,1,&icell,1,&icell,&Jac,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A !=B ) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_DAE_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {
  
  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM       dm;
  Vec      Ul,P,M,R_P,R_M;
  PetscReal *p,*u_t,*r,*r_p,*m;
  PetscInt m_idx, p_idx;
  PetscInt icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Get sub-vectors
  ierr = ExtractSubVectors(Ul,0,&P);
  ierr = ExtractSubVectors(Ul,1,&M);
  ierr = ExtractSubVectors(R,0,&R_P);
  ierr = ExtractSubVectors(R,1,&R_M);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(P,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(P,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,P); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);

  PetscReal vel_error = 0.0;
  PetscInt count = 0;
  PetscInt iface;

  for (iface=0;iface<mesh->num_faces;iface++) tdy->vel[iface] = 0.0;
  
  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, P, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, P, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, P, &vel_error, &count); CHKERRQ(ierr);

  ierr = TDyMPFAOIFunction_InternalVertices_3DMesh(P,R_P,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_SharedWithInternalVertices_3DMesh(P,R_P,ctx); CHKERRQ(ierr);
  ierr = TDyMPFAOIFunction_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(P,R_P,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(M,&m); CHKERRQ(ierr);
  ierr = VecGetArray(U_t,&u_t); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(R_P,&r_p); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    //
    // u   = [P   M  ]^T
    // u_t = [P_t M_t]^T
    //
    // d(M)/dt = - Del(rho * q)
    // M       = rho * phi * sat

    p_idx = icell*2;
    m_idx = p_idx + 1;

    r[p_idx]  = u_t[m_idx] * cells->volume[icell] - tdy->source_sink[icell] * cells->volume[icell]+ r_p[icell];
    r[m_idx]  = m  [icell] - tdy->rho[icell] * tdy->porosity[icell] * tdy->S[icell]* cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(M,&m); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_t,&u_t); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(R_P,&r_p); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
