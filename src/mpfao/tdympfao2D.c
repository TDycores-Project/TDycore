#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeEntryOfGMatrix2D(PetscReal edge_len, PetscReal n[3],
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
  (*g) *= 1.0/(T)*edge_len;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyComputeGMatrixFor2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_subcell    *subcells, *subcell;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges, *edge_up, *edge_dn;
  PetscInt       num_subcells;
  PetscInt       icell, isubcell;
  PetscInt       ii,jj;
  PetscInt       dim;
  PetscInt       e_idx_up, e_idx_dn;
  PetscReal      n_up[3], n_dn[3];
  PetscReal      e_cen_up[3], e_cen_dn[3], v_c[3];
  PetscReal      e_len_dn, e_len_up;
  PetscReal      K[3][3], nu_up[3], nu_dn[3];
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;
  subcells = mesh->subcells;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // extract permeability tensor
    for (ii=0; ii<dim; ii++) {
      for (jj=0; jj<dim; jj++) {
        K[ii][jj] = tdy->K[icell*dim*dim + ii*dim + jj];
      }
    }

    num_subcells = cells->num_subcells[icell];

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      PetscInt vStart = cells->offsets_for_vertex_ids[icell];
      vertex  = &vertices[cells->vertex_ids[vStart+isubcell]];
      subcell = &subcells[icell*cells->num_subcells[icell]+isubcell];

      // determine ids of up & down edges
      PetscInt eStart = cells->offsets_for_edge_ids[icell];
      e_idx_up = cells->edge_ids[eStart + isubcell];
      if (isubcell == 0) e_idx_dn = cells->edge_ids[eStart + num_subcells-1];
      else               e_idx_dn = cells->edge_ids[eStart + isubcell    -1];

      // set points to up/down edges
      edge_up = &edges[e_idx_up];
      edge_dn = &edges[e_idx_dn];

      // extract nu-vectors
      ierr = TDySubCell_GetIthNuVector(subcell, 0, dim, &nu_up[0]); CHKERRQ(ierr);
      ierr = TDySubCell_GetIthNuVector(subcell, 1, dim, &nu_dn[0]); CHKERRQ(ierr);

      // extract centroid of edges
      ierr = TDyEdge_GetCentroid(edge_dn, dim, &e_cen_dn[0]); CHKERRQ(ierr);
      ierr = TDyEdge_GetCentroid(edge_up, dim, &e_cen_up[0]); CHKERRQ(ierr);

      // extract normal to edges
      ierr = TDyEdge_GetNormal(edge_dn, dim, &n_dn[0]); CHKERRQ(ierr);
      ierr = TDyEdge_GetNormal(edge_up, dim, &n_up[0]); CHKERRQ(ierr);

      ierr = TDyVertex_GetCoordinate(vertex, dim, &v_c[0]); CHKERRQ(ierr);

      //
      ierr = TDyComputeLength(v_c, e_cen_dn, dim, &e_len_dn);
      ierr = TDyComputeLength(v_c, e_cen_up, dim, &e_len_up);

      //                               _         _   _           _
      //                              |           | |             |
      //                              | L_up*n_up | | K_xx   K_xy |  _             _
      // Gmatrix =        -1          |           | |             | |               |
      //             -----------      |           | |             | | nu_up   nu_dn |
      //              2*A_{subcell}   | L_dn*n_dn | | K_yx   K_yy | |_             _|
      //                              |           | |             |
      //                              |_         _| |_           _|
      //
      ComputeEntryOfGMatrix2D(e_len_up, n_up, K, nu_up, subcell->T, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][0][0]));
      ComputeEntryOfGMatrix2D(e_len_up, n_up, K, nu_dn, subcell->T, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][0][1]));
      ComputeEntryOfGMatrix2D(e_len_dn, n_dn, K, nu_up, subcell->T, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][1][0]));
      ComputeEntryOfGMatrix2D(e_len_dn, n_dn, K, nu_dn, subcell->T, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][1][1]));
    }
  }

  PetscFunctionReturn(0);

}


/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForInternalVertex2DMesh(TDy tdy,
    TDy_vertex *vertex, TDy_cell *cells) {

  PetscInt       ncells, icell, isubcell;

  PetscReal **Fup, **Fdn;
  PetscReal **Cup, **Cdn;
  PetscReal **A, **B, **AinvB;
  PetscReal *A1d, *B1d, *Cup1d, *AinvB1d, *CuptimesAinvB1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscBLASInt info, *pivots;
  PetscInt i, j, n, m, ndim;
  PetscScalar zero = 0.0, one = 1.0;

  PetscFunctionBegin;

  ndim      = 2;
  ncells    = vertex->num_internal_cells;
  vertex_id = vertex->id;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim, ndim  ); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Fup, ncells, ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Fdn, ncells, ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Cup, ncells, ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&Cdn, ncells, ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&A, ncells, ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&B, ncells, ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_2D(&AinvB, ncells, ncells); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(&A1d, ncells*ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&B1d, ncells*ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&Cup1d, ncells*ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&AinvB1d, ncells*ncells); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_1D(&CuptimesAinvB1d, ncells*ncells); CHKERRQ(ierr);

  for (i=0; i<ncells; i++) {
    icell    = vertex->internal_cell_ids[i];
    isubcell = vertex->subcell_ids[i];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    Fup[i][i] = Gmatrix[0][0] + Gmatrix[0][1];
    Cup[i][i] = Gmatrix[0][0];

    if (i<ncells-1) {
      Cup[i  ][i+1     ] = Gmatrix[0][1];
      Fdn[i+1][i       ] = Gmatrix[1][0] + Gmatrix[1][1];
      Cdn[i+1][i       ] = Gmatrix[1][0];
      Cdn[i+1][i+1     ] = Gmatrix[1][1];
    } else {
      Cup[i  ][0       ] = Gmatrix[0][1];
      Fdn[0  ][ncells-1] = Gmatrix[1][0] + Gmatrix[1][1];
      Cdn[0  ][ncells-1] = Gmatrix[1][0];
      Cdn[0  ][0       ] = Gmatrix[1][1];
    }
  }

  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<ncells; i++) {
      A[i][j] = -Cup[i][j] + Cdn[i][j];
      B[i][j] = -Fup[i][j] + Fdn[i][j];
      A1d[idx]= -Cup[i][j] + Cdn[i][j];
      B1d[idx]= -Fup[i][j] + Fdn[i][j];
      Cup1d[idx] = Cup[i][j];
      idx++;
    }
  }

  n = ncells; m = ncells;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);

  LAPACKgetrf_(&m, &n, A1d, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  ierr = PetscMemcpy(AinvB1d,B1d,sizeof(PetscScalar)*(n*m));
  CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1 * B) by back-substitution
  LAPACKgetrs_("N", &m, &n, A1d, &m, pivots, AinvB1d, &m, &info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (C * AinvB)
  BLASgemm_("N","N", &m, &m, &n, &one, Cup1d, &m, AinvB1d, &m, &zero,
            CuptimesAinvB1d, &m);

  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<ncells; i++) {
      AinvB[i][j] = AinvB1d[idx];
      tdy->Trans[vertex_id][i][j] = CuptimesAinvB1d[idx] - Fup[i][j];
      idx++;
    }
  }

  // Free up the memory
  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim   ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fup, ncells ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Fdn, ncells ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup, ncells ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn, ncells ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(A, ncells ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(B, ncells ); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(AinvB, ncells ); CHKERRQ(ierr);
  ierr = PetscFree(pivots                         ); CHKERRQ(ierr);

  free(A1d             );
  free(B1d             );
  free(Cup1d           );
  free(AinvB1d         );
  free(CuptimesAinvB1d );

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForBoundaryVertex2DMesh(TDy tdy,
    TDy_vertex *vertex, TDy_cell *cells) {

  PetscInt       ncells_in, ncells_bc, icell, isubcell;

  PetscReal **Fup, **Cup, **Fdn, **Cdn;
  PetscReal **FupInxIn,
            **FdnInxIn; // InxIn: Internal flux with contribution from unknown internal pressure values
  PetscReal **FupBcxIn,
            **FdnBcxIn; // BcxIn: Boundary flux with contribution from unknown internal pressure values
  PetscReal **CupInxIn,
            **CdnInxIn; // InxIn: Internal flux with contribution from unknown internal pressure values
  PetscReal **CupInxBc,
            **CdnInxBc; // Inxbc: Internal flux with contribution from known boundary pressure values
  PetscReal **CupBcxIn,
            **CdnBcxIn; // BcxIn: Boundary flux with contribution from unknown internal pressure values
  PetscReal **CupBcxBc,
            **CdnBcxBc; // BcxIn: Boundary flux with contribution from known boundary pressure values

  PetscReal *AInxIninv_1d;
  PetscReal **AInxIn, **BInxIn, **AInxIninvBInxIn  ;
  PetscReal *AInxIn_1d, *BInxIn_1d, *DInxBc_1d, *AInxIninvBInxIn_1d,
            *AInxIninvDInxBc_1d;

  PetscReal *CupInxIn_1d, *CupInxIntimesAInxIninvBInxIn_1d,
            *CupInxIntimesAInxIninvDInxBc_1d;
  PetscReal *CupBcxIn_1d, *CdnBcxIn_1d;

  PetscReal *CupBcxIntimesAInxIninvBInxIn_1d, *CdnBcxIntimesAInxIninvBInxIn_1d;
  PetscReal *CupBcxIntimesAInxIninvDInxBc_1d, *CdnBcxIntimesAInxIninvDInxBc_1d;

  PetscReal *lapack_mem_1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscBLASInt info, *pivots;
  PetscInt n,m,k,ndim;
  PetscScalar zero = 0.0, one = 1.0;
  PetscInt i,j;

  PetscFunctionBegin;

  ndim      = 2;
  ncells_in = vertex->num_internal_cells;
  ncells_bc = vertex->num_boundary_cells;
  vertex_id = vertex->id;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim, ndim);

  ierr = TDyAllocate_RealArray_2D(&Fup, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = TDyAllocate_RealArray_2D(&Cup, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = TDyAllocate_RealArray_2D(&Fdn, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = TDyAllocate_RealArray_2D(&Cdn, ncells_in+ncells_bc, ncells_in+ncells_bc);

  ierr = TDyAllocate_RealArray_2D(&FupInxIn, ncells_in-1, ncells_in);
  ierr = TDyAllocate_RealArray_2D(&FupBcxIn, 1, ncells_in);
  ierr = TDyAllocate_RealArray_2D(&FdnInxIn, ncells_in-1, ncells_in);
  ierr = TDyAllocate_RealArray_2D(&FdnBcxIn, 1, ncells_in);

  ierr = TDyAllocate_RealArray_2D(&CupInxIn, ncells_in-1, ncells_in-1);
  ierr = TDyAllocate_RealArray_2D(&CupInxBc, ncells_in-1, ncells_bc  );
  ierr = TDyAllocate_RealArray_2D(&CupBcxIn, 1, ncells_in-1);
  ierr = TDyAllocate_RealArray_2D(&CupBcxBc, 1, ncells_bc  );

  ierr = TDyAllocate_RealArray_2D(&CdnInxIn, ncells_in-1, ncells_in-1);
  ierr = TDyAllocate_RealArray_2D(&CdnInxBc, ncells_in-1, ncells_bc  );
  ierr = TDyAllocate_RealArray_2D(&CdnBcxIn, 1, ncells_in-1);
  ierr = TDyAllocate_RealArray_2D(&CdnBcxBc, 1, ncells_bc  );

  ierr = TDyAllocate_RealArray_2D(&AInxIn, ncells_in-1, ncells_in-1);
  ierr = TDyAllocate_RealArray_2D(&BInxIn, ncells_in-1, ncells_in  );

  ierr = TDyAllocate_RealArray_2D(&AInxIninvBInxIn, ncells_in-1, ncells_in);

  ierr = TDyAllocate_RealArray_1D(&AInxIn_1d, (ncells_in-1)*(ncells_in-1));
  ierr = TDyAllocate_RealArray_1D(&lapack_mem_1d, (ncells_in-1)*(ncells_in-1));
  ierr = TDyAllocate_RealArray_1D(&BInxIn_1d, (ncells_in-1)* ncells_in   );
  ierr = TDyAllocate_RealArray_1D(&AInxIninv_1d, (ncells_in-1)*(ncells_in-1));
  ierr = TDyAllocate_RealArray_1D(&AInxIninvBInxIn_1d, (ncells_in-1)* ncells_in   );
  ierr = TDyAllocate_RealArray_1D(&CupInxIn_1d, (ncells_in-1)*(ncells_in-1));
  ierr = TDyAllocate_RealArray_1D(&CupInxIntimesAInxIninvBInxIn_1d,
                               (ncells_in-1)*(ncells_in)  );
  ierr = TDyAllocate_RealArray_1D(&CupBcxIn_1d, (1          )*(ncells_in-1));
  ierr = TDyAllocate_RealArray_1D(&CdnBcxIn_1d, (1          )*(ncells_in-1));
  ierr = TDyAllocate_RealArray_1D(&CupBcxIntimesAInxIninvBInxIn_1d,
                               (1          )*(ncells_in)  );
  ierr = TDyAllocate_RealArray_1D(&CdnBcxIntimesAInxIninvBInxIn_1d,
                               (1          )*(ncells_in)  );

  ierr = TDyAllocate_RealArray_1D(&DInxBc_1d, (ncells_in-1)* ncells_bc   );
  ierr = TDyAllocate_RealArray_1D(&AInxIninvDInxBc_1d, (ncells_in-1)* ncells_bc   );
  ierr = TDyAllocate_RealArray_1D(&CupInxIntimesAInxIninvDInxBc_1d,
                               (ncells_in-1)*(ncells_bc)  );
  ierr = TDyAllocate_RealArray_1D(&CupBcxIntimesAInxIninvDInxBc_1d,
                               (1          )*(ncells_bc)  );
  ierr = TDyAllocate_RealArray_1D(&CdnBcxIntimesAInxIninvDInxBc_1d,
                               (1          )*(ncells_bc)  );

  for (i=0; i<ncells_in; i++) {
    icell    = vertex->internal_cell_ids[i];
    isubcell = vertex->subcell_ids[i];
    ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);

    Fup[i][i  ] = Gmatrix[0][0] + Gmatrix[0][1];
    Cup[i][i  ] = Gmatrix[0][0];
    Cup[i][i+1] = Gmatrix[0][1];

    Fdn[i+1][i  ] = Gmatrix[1][0] + Gmatrix[1][1];
    Cdn[i+1][i  ] = Gmatrix[1][0];
    Cdn[i+1][i+1] = Gmatrix[1][1];
  }

  i = 0        ;
  for (j=0; j<ncells_in; j++) FupBcxIn[0][j] = Fup[i][j];
  i = ncells_in;
  for (j=0; j<ncells_in; j++) FdnBcxIn[0][j] = Fdn[i][j];

  CupBcxBc[0][0] = Cup[0][0];
  CupBcxBc[0][1] = Cup[0][ncells_in];
  CdnBcxBc[0][0] = Cdn[ncells_in][0];
  CdnBcxBc[0][1] = Cdn[ncells_in][ncells_in];

  for (i=0; i<ncells_in-1; i++) {
    CupInxBc[i][0] = Cup[i+1][0        ];
    CupInxBc[i][1] = Cup[i+1][ncells_in];
    CdnInxBc[i][0] = Cdn[i+1][0        ];
    CdnInxBc[i][1] = Cdn[i+1][ncells_in];

    for (j=0; j<ncells_in-1; j++) {
      CupInxIn[i][j] = Cup[i+1][j+1];
      CdnInxIn[i][j] = Cdn[i+1][j+1];
    }

    for (j=0; j<ncells_in; j++) {
      FupInxIn[i][j] = Fup[i+1][j];
      FdnInxIn[i][j] = Fdn[i+1][j];
    }

  }

  idx = 0;
  for (j=0; j<ncells_in-1; j++) {
    for (i=0; i<ncells_in-1; i++) {
      AInxIn[i][j]     = -CupInxIn[i][j] + CdnInxIn[i][j];
      AInxIn_1d[idx]   = -CupInxIn[i][j] + CdnInxIn[i][j];
      CupInxIn_1d[idx] = CupInxIn[i][j];
      idx++;
    }
  }

  idx = 0;
  i   = 0;
  for (j=0; j<ncells_in-1; j++) {
    CupBcxIn_1d[idx] = Cup[i][j+1];
    idx++;
  }

  idx = 0;
  i   = ncells_in;
  for (j=0; j<ncells_in-1; j++) {
    CdnBcxIn_1d[idx] = Cdn[i][j+1];
    idx++;
  }

  idx = 0;
  for (j=0; j<ncells_bc; j++) {
    for (i=0; i<ncells_in-1; i++) {
      //DInxBc[i][j]     = CupInxBc[i][j] - CdnInxBc[i][j];
      DInxBc_1d[idx]   = CupInxBc[i][j] - CdnInxBc[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells_in; j++) {
    for (i=0; i<ncells_in-1; i++) {
      BInxIn[i][j]     = -FupInxIn[i][j] + FdnInxIn[i][j];
      BInxIn_1d[idx]   = -FupInxIn[i][j] + FdnInxIn[i][j];
      idx++;
    }
  }

  n = ncells_in-1; m = ncells_in-1;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);

  LAPACKgetrf_(&m, &n, AInxIn_1d, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  ierr = PetscMemcpy(AInxIninvBInxIn_1d, BInxIn_1d,sizeof(PetscScalar)*(n*(m+1)));
  CHKERRQ(ierr); // AinvB in col major
  ierr = PetscMemcpy(AInxIninv_1d, AInxIn_1d,sizeof(PetscScalar)*(n*m    ));
  CHKERRQ(ierr); // AinvB in col major

  ierr = PetscMemcpy(AInxIninvDInxBc_1d, DInxBc_1d,
                     sizeof(PetscScalar)*(ncells_bc*(ncells_in-1)));
  CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1) by back-substitution
  PetscInt nn = n*n;
  LAPACKgetri_(&n, AInxIninv_1d, &n, pivots, lapack_mem_1d, &nn, &info);

  // Compute (Ainv*B)
  m = ncells_in-1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, AInxIninv_1d, &m, BInxIn_1d, &k, &zero,
            AInxIninvBInxIn_1d, &m);

  // Compute (Ainv*D)
  m = ncells_in-1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, AInxIninv_1d, &m, DInxBc_1d, &k, &zero,
            AInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*B)
  m = ncells_in-1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupInxIn_1d, &m, AInxIninvBInxIn_1d, &k,
            &zero, CupInxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*B) for up boundary flux
  m = 1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupBcxIn_1d, &m, AInxIninvBInxIn_1d, &k,
            &zero, CupBcxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*B) for down boundary flux
  m = 1; n = ncells_in; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CdnBcxIn_1d, &m, AInxIninvBInxIn_1d, &k,
            &zero, CdnBcxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*D)
  m = ncells_in-1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupInxIn_1d, &m, AInxIninvDInxBc_1d, &k,
            &zero, CupInxIntimesAInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*D) for up boundary flux
  m = 1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CupBcxIn_1d, &m, AInxIninvDInxBc_1d, &k,
            &zero, CupBcxIntimesAInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*D) for down boundary flux
  m = 1; n = ncells_bc; k = ncells_in-1;
  BLASgemm_("N","N", &m, &n, &k, &one, CdnBcxIn_1d, &m, AInxIninvDInxBc_1d, &k,
            &zero, CdnBcxIntimesAInxIninvDInxBc_1d, &m);

  idx = 0;
  for (j=0; j<ncells_in; j++) {
    for (i=0; i<ncells_in-1; i++) {
      tdy->Trans[vertex_id][i][j] = CupInxIntimesAInxIninvBInxIn_1d[idx] -
                                    FupInxIn[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells_bc; j++) {
    for (i=0; i<ncells_in-1; i++) {
      tdy->Trans[vertex_id][i][j+ncells_in] = CupInxIntimesAInxIninvDInxBc_1d[idx] +
                                              CupInxBc[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<ncells_in; j++) {
    i = ncells_in-1;
    tdy->Trans[vertex_id][i][j] = CupBcxIntimesAInxIninvBInxIn_1d[idx] -
                                  FupBcxIn[0][j];
    i = ncells_in  ;
    tdy->Trans[vertex_id][i][j] = CdnBcxIntimesAInxIninvBInxIn_1d[idx] -
                                  FdnBcxIn[0][j];
    idx++;
  }

  idx = 0;
  for (j=0; j<ncells_bc; j++) {
    i = ncells_in-1;
    tdy->Trans[vertex_id][i][j+ncells_in] = CupBcxIntimesAInxIninvDInxBc_1d[idx] +
                                            CupBcxBc[0][j];
    i = ncells_in  ;
    tdy->Trans[vertex_id][i][j+ncells_in] = CdnBcxIntimesAInxIninvDInxBc_1d[idx] +
                                            CdnBcxBc[0][j];
    idx++;
  }


  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim);

  ierr = TDyDeallocate_RealArray_2D(Fup, ncells_in+ncells_bc);
  ierr = TDyDeallocate_RealArray_2D(Cup, ncells_in+ncells_bc);
  ierr = TDyDeallocate_RealArray_2D(Fdn, ncells_in+ncells_bc);
  ierr = TDyDeallocate_RealArray_2D(Cdn, ncells_in+ncells_bc);

  ierr = TDyDeallocate_RealArray_2D(FupInxIn, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(FupBcxIn, 1          );
  ierr = TDyDeallocate_RealArray_2D(FdnInxIn, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(FdnBcxIn, 1          );

  ierr = TDyDeallocate_RealArray_2D(CupInxIn, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(CupInxBc, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(CupBcxIn, 1          );
  ierr = TDyDeallocate_RealArray_2D(CupBcxBc, 1          );

  ierr = TDyDeallocate_RealArray_2D(CdnInxIn, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(CdnInxBc, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(CdnBcxIn, 1          );
  ierr = TDyDeallocate_RealArray_2D(CdnBcxBc, 1          );

  ierr = TDyDeallocate_RealArray_2D(AInxIn, ncells_in-1);
  ierr = TDyDeallocate_RealArray_2D(BInxIn, ncells_in-1);

  ierr = TDyDeallocate_RealArray_2D(AInxIninvBInxIn, ncells_in-1);

  free(AInxIn_1d                      );
  free(lapack_mem_1d                  );
  free(BInxIn_1d                      );
  free(AInxIninv_1d                   );
  free(AInxIninvBInxIn_1d             );
  free(CupInxIn_1d                    );
  free(CupInxIntimesAInxIninvBInxIn_1d);
  free(CupBcxIn_1d                    );
  free(CdnBcxIn_1d                    );
  free(CupBcxIntimesAInxIninvBInxIn_1d);
  free(CdnBcxIntimesAInxIninvBInxIn_1d);
  free(DInxBc_1d                      );
  free(AInxIninvDInxBc_1d             );
  free(CupInxIntimesAInxIninvDInxBc_1d);
  free(CupBcxIntimesAInxIninvDInxBc_1d);
  free(CdnBcxIntimesAInxIninvDInxBc_1d);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyComputeTransmissibilityMatrix2DMesh(TDy tdy) {

  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = mesh->vertices;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    vertex = &vertices[ivertex];
    if (vertex->num_boundary_cells == 0) {

      ierr = ComputeTransmissibilityMatrixForInternalVertex2DMesh(tdy, vertex, cells);
      CHKERRQ(ierr);
    } else {
      if (vertex->num_internal_cells > 1) {
        ierr = ComputeTransmissibilityMatrixForBoundaryVertex2DMesh(tdy, vertex, cells);
        CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices_2DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges;
  PetscInt       ivertex, icell, icell_from, icell_to;
  PetscInt       icol, row, col, vertex_id;
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
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;

    if (vertex->num_boundary_cells == 0) {
      for (icell=0; icell<vertex->num_internal_cells; icell++) {

        TDy_edge *edge;

        if (icell==0) edge = &edges[vertex->edge_ids[vertex->num_internal_cells-1]];
        else          edge = &edges[vertex->edge_ids[icell-1]];

        icell_from = edge->cell_ids[0];
        icell_to   = edge->cell_ids[1];


        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = vertex->internal_cell_ids[icol];
          col   = cells->global_id[vertex->internal_cell_ids[icol]];
          if (col<0) col = -col - 1;
          value = tdy->Trans[vertex_id][icell][icol];
          row = cells->global_id[icell_from];
          if (cells->is_local[icell_from]) {ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);}
          row = cells->global_id[icell_to];
          if (cells->is_local[icell_to]) {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
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
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_2DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges;
  PetscInt       ivertex, icell, icell_from, icell_to;
  PetscInt       icol, row, col, vertex_id, iedge;
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
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    
    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;
    
    if (vertex->num_boundary_cells == 0) continue;
    if (vertex->num_internal_cells < 1)  continue;
    
    // Vertex is on the boundary
    
    PetscScalar pBoundary[4];
    PetscInt cell_ids_from_to[4][2];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    
    numBoundary = 0;
    for (iedge=0; iedge<vertex->num_edges; iedge++) {
      
      TDy_edge *edge;
      if (iedge==0) edge = &edges[vertex->edge_ids[vertex->num_edges-1]];
      else          edge = &edges[vertex->edge_ids[iedge-1]];
      
      if (edge->is_internal == 0) {
        
        PetscInt f;
        f = edge->id + fStart;
        ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
        cell_ids_from_to[numBoundary][0] = edge->cell_ids[0];
        cell_ids_from_to[numBoundary][1] = edge->cell_ids[1];
        numBoundary++;
      }
    }
    
    for (icell=0; icell<vertex->num_internal_cells-1; icell++) {
      iedge = vertex->edge_ids[icell];
      
      TDy_edge *edge;
      edge = &edges[vertex->edge_ids[icell]];
      
      icell_from = edge->cell_ids[0];
      icell_to   = edge->cell_ids[1];

      if (cells->is_local[icell_from]) {
        row   = cells->global_id[icell_from];
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells->global_id[vertex->internal_cell_ids[icol]];
          value = tdy->Trans[vertex_id][icell][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell][icol + vertex->num_internal_cells] *
          pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cells->is_local[icell_to]) {
        row   = cells->global_id[icell_to];
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells->global_id[vertex->internal_cell_ids[icol]];
          value = tdy->Trans[vertex_id][icell][icol];
          ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell][icol + vertex->num_internal_cells] *
          pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (icell=0; icell<vertex->num_boundary_cells; icell++) {
      row = cell_ids_from_to[icell][0];
      
      if (cell_ids_from_to[icell][0]>-1 && cells->is_local[cell_ids_from_to[icell][0]]) {
        row   = cells->global_id[cell_ids_from_to[icell][0]];
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol +
                                                                            vertex->num_internal_cells] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells->global_id[vertex->internal_cell_ids[icol]];
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cell_ids_from_to[icell][1]>-1 && cells->is_local[cell_ids_from_to[icell][1]]) {
        row   = cells->global_id[cell_ids_from_to[icell][1]];
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol +
                                                                            vertex->num_internal_cells] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells->global_id[vertex->internal_cell_ids[icol]];
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol];
          {ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);}
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
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_2DMesh(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges;
  PetscInt       ivertex, icell, isubcell;
  PetscInt       icol, row, col, iedge;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscReal      sign;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = &mesh->cells;
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    
    vertex    = &vertices[ivertex];
    
    if (vertex->num_boundary_cells == 0) continue;
    if (vertex->num_internal_cells > 1)  continue;
    
    // Vertex is on the boundary
    
    PetscScalar pBoundary[4];
    PetscInt cell_ids_from_to[4][2];
    PetscInt numBoundary;
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    
    numBoundary = 0;
    for (iedge=0; iedge<vertex->num_edges; iedge++) {
      
      TDy_edge *edge;
      if (iedge==0) edge = &edges[vertex->edge_ids[vertex->num_edges-1]];
      else          edge = &edges[vertex->edge_ids[iedge-1]];
      
      if (edge->is_internal == 0) {
        
        PetscInt f;
        f = edge->id + fStart;
        ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
        cell_ids_from_to[numBoundary][0] = edge->cell_ids[0];
        cell_ids_from_to[numBoundary][1] = edge->cell_ids[1];
        numBoundary++;
      }
    }
    
    icell    = vertex->internal_cell_ids[0];
    isubcell = vertex->subcell_ids[0];
    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);
    value = 0.0;
    for (i=0; i<dim; i++) {
      row = cell_ids_from_to[i][0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;
      for (j=0; j<dim; j++) {
        value += sign*Gmatrix[i][j];
      }
    }
    row   = cells->global_id[vertex->internal_cell_ids[0]];
    col   = cells->global_id[vertex->internal_cell_ids[0]];
    if (cells->is_local[icell]) {ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);}
    
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (icell=0; icell<vertex->num_boundary_cells; icell++) {
      
      if (cell_ids_from_to[icell][0]>-1 && cells->is_local[cell_ids_from_to[icell][0]]) {
        row   = cells->global_id[cell_ids_from_to[icell][0]];
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = Gmatrix[icell][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cell_ids_from_to[icell][1]>-1 && cells->is_local[cell_ids_from_to[icell][1]]) {
        row   = cells->global_id[cell_ids_from_to[icell][1]];
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = Gmatrix[icell][icol] * pBoundary[icol];
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

PetscErrorCode TDyMPFAORecoverVelocity_2DMesh(TDy tdy, Vec U) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges, *edge;
  PetscInt       ivertex, icell, isubcell;
  PetscInt       icol, row, col, vertex_id, iedge;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscInt       edge_id;
  PetscScalar    *u;
  Vec            localU;
  PetscReal      vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  vertices = mesh->vertices;
  edges    = mesh->edges;
  row = -1;

  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, dim, dim);

  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  PetscReal vel_error;
  PetscReal X[2];
  PetscReal vel_normal;

  vel_error = 0.0;
  PetscInt count = 0;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;

    if (vertex->num_boundary_cells == 0) {

      // Internal vertex

      PetscScalar Pcomputed[vertex->num_internal_cells];
      PetscScalar Vcomputed[vertex->num_internal_cells];

      // Save local pressure stencil
      for (icell=0; icell<vertex->num_internal_cells; icell++) {
        Pcomputed[icell] = u[vertex->internal_cell_ids[icell]];
        Vcomputed[icell] = 0.0;
      }

      // F = T*P
      for (icell=0; icell<vertex->num_internal_cells; icell++) {
        if (icell==0) edge_id = vertex->edge_ids[vertex->num_internal_cells-1];
        else          edge_id = vertex->edge_ids[icell-1];

        edge = &(edges[edge_id]);

        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          Vcomputed[icell] += tdy->Trans[vertex_id][icell][icol] * Pcomputed[icol] *2.0/edge->length/2.0;
        }

        tdy->vel[edge_id] += Vcomputed[icell];

        if (edges[edge_id].is_local){
          X[0] = (tdy->X[(edge_id + fStart)*dim]     + vertex->coordinate.X[0])/2.0;
          X[1] = (tdy->X[(edge_id + fStart)*dim + 1] + vertex->coordinate.X[1])/2.0;
          
          ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);

          vel_normal = (vel[0]*edge->normal.V[0] + vel[1]*edge->normal.V[1])/2.0;

          vel_error += PetscPowReal( (Vcomputed[icell] - vel_normal), 2.0);
          count++;
        }
      }

    } else {

      // Boudnary vertex

      PetscScalar Pcomputed[vertex->num_internal_cells];
      PetscScalar Pboundary[vertex->num_boundary_cells];
      PetscScalar Vcomputed[vertex->num_internal_cells + vertex->num_boundary_cells];

      for (icell=0; icell<vertex->num_internal_cells; icell++) {
        Pcomputed[icell] = u[vertex->internal_cell_ids[icell]];
        Vcomputed[icell] = 0.0;
      }

      PetscInt numBoundary = 0;
      for (iedge=0; iedge<vertex->num_edges; iedge++) {

        if (iedge==0) edge = &edges[vertex->edge_ids[vertex->num_edges-1]];
        else          edge = &edges[vertex->edge_ids[iedge-1]];

        if (edge->is_internal == 0) {
          PetscInt f = edge->id + fStart;
          ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &Pboundary[numBoundary], tdy->dirichletvaluectx);CHKERRQ(ierr);
          Vcomputed[vertex->num_internal_cells + numBoundary] = 0.0;
          numBoundary++;
        }
      }

      if (vertex->num_internal_cells > 1) {

        // F = T_in * P_in + T_bc * P_bc

        // Flux through internal edges
        for (icell=0; icell<vertex->num_internal_cells-1; icell++) {
          edge_id = vertex->edge_ids[icell];
          edge = &edges[edge_id];

          // T_in * P_in
          for (icol=0; icol<vertex->num_internal_cells; icol++) {
            row = icell;
            col = icol;
            Vcomputed[row] += tdy->Trans[vertex_id][row][col] * Pcomputed[icol] *2.0/edge->length/2.0;
          }

          // T_bc * P_bc
          for (icol=0; icol<vertex->num_boundary_cells; icol++) {
            row = icell;
            col = icol+vertex->num_internal_cells;
            Vcomputed[row] += tdy->Trans[vertex_id][row][col] * Pboundary[icol] *2.0/edge->length/2.0;
          }

          tdy->vel[edge_id] += Vcomputed[row];

          if (edges[edge_id].is_local){
            X[0] = (tdy->X[(edge_id + fStart)*dim]     + vertex->coordinate.X[0])/2.0;
            X[1] = (tdy->X[(edge_id + fStart)*dim + 1] + vertex->coordinate.X[1])/2.0;
  
            ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
            vel_normal = (vel[0]*edge->normal.V[0] + vel[1]*edge->normal.V[1])/2.0;
  
            vel_error += PetscPowReal( (Vcomputed[row] - vel_normal), 2.0);
            count++;
          }
        }

        // Flux through boundary edges
        for (icell=0; icell<vertex->num_boundary_cells; icell++) {

          if (icell == 0) edge_id = vertex->edge_ids[icell + vertex->num_internal_cells  ];
          else            edge_id = vertex->edge_ids[icell + vertex->num_internal_cells-2];
          edge = &edges[edge_id];

          // T_in * P_in
          for (icol=0; icol<vertex->num_internal_cells; icol++) {
            row = icell+vertex->num_internal_cells-1;
            col = icol;
            Vcomputed[row] += tdy->Trans[vertex_id][row][col] * Pcomputed[icol] *2.0/edge->length/2.0;
          }

          // T_bc * P_bc

          for (icol=0; icol<vertex->num_boundary_cells; icol++) {
            row = icell+vertex->num_internal_cells-1;
            col = icol+vertex->num_internal_cells;
            Vcomputed[row] += tdy->Trans[vertex_id][row][col] * Pboundary[icol] *2.0/edge->length/2.0;
          }

          tdy->vel[edge_id] += Vcomputed[row];

          if (edges[edge_id].is_local){
            X[0] = (tdy->X[(edge_id + fStart)*dim]     + vertex->coordinate.X[0])/2.0;
            X[1] = (tdy->X[(edge_id + fStart)*dim + 1] + vertex->coordinate.X[1])/2.0;

            ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);
            vel_normal = (vel[0]*edge->normal.V[0] + vel[1]*edge->normal.V[1])/2.0;

            vel_error += PetscPowReal( (Vcomputed[row] - vel_normal), 2.0);
            count++;
          }

        }
      } else {

        // Boundary vertex is at a corner
        icell    = vertex->internal_cell_ids[0];
        isubcell = vertex->subcell_ids[0];
        ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);

        for (iedge=0; iedge<vertex->num_edges; iedge++) {

          if (iedge==0) edge = &edges[vertex->edge_ids[vertex->num_edges-1]];
          else          edge = &edges[vertex->edge_ids[iedge-1]];

          edge_id = edge->id;
          Vcomputed[0] = 0.0;
          for (icol=0; icol<vertex->num_boundary_cells; icol++) {
            Vcomputed[0]      += -(Gmatrix[iedge][icol]*Pcomputed[0] - Gmatrix[iedge][icol]*Pboundary[icol])*2.0/edge->length/2.0;
          }

          tdy->vel[edge_id] += Vcomputed[0];

          if (edges[edge_id].is_local){
            X[0] = (tdy->X[(edge_id + fStart)*dim]     + vertex->coordinate.X[0])/2.0;
            X[1] = (tdy->X[(edge_id + fStart)*dim + 1] + vertex->coordinate.X[1])/2.0;

            ierr = (*tdy->ops->computedirichletflux)(tdy,X,vel,tdy->dirichletfluxctx);CHKERRQ(ierr);

            vel_normal = (vel[0]*edge->normal.V[0] + vel[1]*edge->normal.V[1])/2.0;

            vel_error += PetscPowReal( (Vcomputed[0] - vel_normal), 2.0);
            count++;
          }

        }

      }

    } // if (vertex->num_boundary_cells == 0)

  } // for-loop

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

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
PetscReal TDyMPFAOVelocityNorm_2DMesh(TDy tdy) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_edge       *edges, *edge;
  TDy_cell       *cells;
  PetscInt       dim;
  PetscInt       icell, iedge, edge_id;
  PetscInt       fStart, fEnd;
  PetscReal      norm, norm_sum, vel_normal;
  PetscReal      vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = &mesh->cells;
  edges = mesh->edges;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  norm_sum = 0.0;
  norm     = 0.0;

  for (icell=0; icell<mesh->num_cells; icell++) {

    if (!cells->is_local[icell]) continue;

    for (iedge=0; iedge<cells->num_edges[icell]; iedge++) {
      PetscInt eStart = cells->offsets_for_edge_ids[icell];
      edge_id = cells->edge_ids[eStart+iedge];
      edge    = &(edges[edge_id]);

      ierr = (*tdy->ops->computedirichletflux)(tdy, &(tdy->X[(edge_id + fStart)*dim]), vel, tdy->dirichletfluxctx);CHKERRQ(ierr);
      vel_normal = vel[0]*edge->normal.V[0] + vel[1]*edge->normal.V[1];

      norm += PetscSqr((vel_normal - tdy->vel[edge_id]))*tdy->V[icell];
    }
  }

  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);

  PetscFunctionReturn(norm_sum);
}
