#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeEntryOfGMatrix(PetscReal edge_len, PetscReal n[3],
                                     PetscReal K[3][3], PetscReal v[3],
                                     PetscReal area, PetscInt dim, PetscReal *g) {

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
  (*g) *= 1.0/(2.0*area)*edge_len;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeEntryOfGMatrix3D(PetscReal area, PetscReal n[3],
                                       PetscReal K[3][3], PetscReal v[3],
                                       PetscReal vol, PetscInt dim, PetscReal *g) {

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
  (*g) *= -1.0/(vol)*area;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGMatrixFor2DMesh(TDy tdy) {

  PetscFunctionBegin;

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_subcell    *subcell;
  TDy_vertex     *vertices, *vertex;
  TDy_edge       *edges, *edge_up, *edge_dn;
  PetscInt       num_subcells;
  PetscInt       icell, isubcell;
  PetscInt       ii,jj;
  PetscInt       dim, d;
  PetscInt       e_idx_up, e_idx_dn;
  PetscReal      n_up[3], n_dn[3];
  PetscReal      e_cen_up[3], e_cen_dn[3], v_c[3];
  PetscReal      e_len_dn, e_len_up;
  PetscReal      K[3][3], nu_up[3], nu_dn[3];
  PetscErrorCode ierr;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;
  edges    = mesh->edges;
  vertices = mesh->vertices;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // set pointer to cell
    cell = &cells[icell];

    // extract permeability tensor
    for (ii=0; ii<dim; ii++) {
      for (jj=0; jj<dim; jj++) {
        K[ii][jj] = tdy->K[icell*dim*dim + ii*dim + jj];
      }
    }

    num_subcells = cell->num_subcells;

    for (isubcell=0; isubcell<num_subcells; isubcell++) {

      vertex  = &vertices[cell->vertex_ids[isubcell]];
      subcell = &cell->subcells[isubcell];

      // determine ids of up & down edges
      e_idx_up = cells[icell].edge_ids[isubcell];
      if (isubcell == 0) e_idx_dn = cells[icell].edge_ids[num_subcells-1];
      else               e_idx_dn = cells[icell].edge_ids[isubcell    -1];

      // set points to up/down edges
      edge_up = &edges[e_idx_up];
      edge_dn = &edges[e_idx_dn];

      for (d=0; d<dim; d++) {

        // extract nu-vectors
        nu_up[d]    = subcell->nu_vector[0].V[d];
        nu_dn[d]    = subcell->nu_vector[1].V[d];

        // extract face centroid of edges
        e_cen_dn[d] = edge_dn->centroid.X[d];
        e_cen_up[d] = edge_up->centroid.X[d];

        // extract normal to edges
        n_dn[d] = edge_dn->normal.V[d];
        n_up[d] = edge_up->normal.V[d];

        // extract coordinate of the vertex
        v_c[d] = vertex->coordinate.X[d];
      }

      //
      ierr = ComputeLength(v_c, e_cen_dn, dim, &e_len_dn);
      ierr = ComputeLength(v_c, e_cen_up, dim, &e_len_up);

      //                               _         _   _           _
      //                              |           | |             |
      //                              | L_up*n_up | | K_xx   K_xy |  _             _
      // Gmatrix =        -1          |           | |             | |               |
      //             -----------      |           | |             | | nu_up   nu_dn |
      //              2*A_{subcell}   | L_dn*n_dn | | K_yx   K_yy | |_             _|
      //                              |           | |             |
      //                              |_         _| |_           _|
      //
      ComputeEntryOfGMatrix(e_len_up, n_up, K, nu_up, subcell->volume, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][0][0]));
      ComputeEntryOfGMatrix(e_len_up, n_up, K, nu_dn, subcell->volume, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][0][1]));
      ComputeEntryOfGMatrix(e_len_dn, n_dn, K, nu_up, subcell->volume, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][1][0]));
      ComputeEntryOfGMatrix(e_len_dn, n_dn, K, nu_dn, subcell->volume, dim,
                            &(tdy->subc_Gmatrix[icell][isubcell][1][1]));
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGMatrixFor3DMesh(TDy tdy) {

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

        PetscInt d;
        PetscReal area;
        PetscReal normal[3];

        TDy_face *face = &faces[subcell->face_ids[ii]];
        
        area = subcell->face_area[ii];
        for (d=0; d<dim; d++) normal[d] = face->normal.V[d];

        for (jj=0;jj<subcell->num_faces;jj++) {
          PetscReal nu[3];

          for (d=0; d<dim; d++) nu[d] = subcell->nu_vector[jj].V[d];
          ierr = ComputeEntryOfGMatrix3D(area, normal, K, nu, subcell->volume, dim,
                                         &(tdy->subc_Gmatrix[icell][isubcell][ii][jj]));
          CHKERRQ(ierr);
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGMatrix(TDy tdy) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  if (tdy->ops->computepermeability) {

    PetscReal *localK;
    PetscInt icell, ii, jj;
    TDy_mesh *mesh;

    mesh = tdy->mesh;

    // If peremeability function is set, use it instead.
    // Will need to consolidate this code with code in tdypermeability.c
    ierr = PetscMalloc(9*sizeof(PetscReal),&localK); CHKERRQ(ierr);
    for (icell=0; icell<mesh->num_cells; icell++) {
      ierr = (*tdy->ops->computepermeability)(tdy, &(tdy->X[icell*dim]), localK, tdy->permeabilityctx);CHKERRQ(ierr);

      PetscInt count = 0;
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          tdy->K[icell*dim*dim + ii*dim + jj] = localK[count];
          count++;
        }
      }
    }
    ierr = PetscFree(localK); CHKERRQ(ierr);
  }

  switch (dim) {
  case 2:
    ierr = ComputeGMatrixFor2DMesh(tdy); CHKERRQ(ierr);
    break;
  case 3:
    ierr = ComputeGMatrixFor3DMesh(tdy); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

PetscErrorCode ExtractSubGmatrix(TDy tdy, PetscInt cell_id,
                                 PetscInt sub_cell_id, PetscInt dim, PetscReal **Gmatrix) {

  PetscInt i, j;

  PetscFunctionBegin;

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      Gmatrix[i][j] = tdy->subc_Gmatrix[cell_id][sub_cell_id][i][j];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForInternalVertex(TDy tdy,
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

  ierr = Allocate_RealArray_2D(&Gmatrix, ndim, ndim  ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Fup, ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Fdn, ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Cup, ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Cdn, ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&A, ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&B, ncells, ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&AinvB, ncells, ncells); CHKERRQ(ierr);

  ierr = Allocate_RealArray_1D(&A1d, ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&B1d, ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&Cup1d, ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&AinvB1d, ncells*ncells); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&CuptimesAinvB1d, ncells*ncells); CHKERRQ(ierr);

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
  ierr = Deallocate_RealArray_2D(Gmatrix, ndim   ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fup, ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fdn, ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cup, ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cdn, ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(A, ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(B, ncells ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(AinvB, ncells ); CHKERRQ(ierr);
  ierr = PetscFree(pivots                         ); CHKERRQ(ierr);

  free(A1d             );
  free(B1d             );
  free(Cup1d           );
  free(AinvB1d         );
  free(CuptimesAinvB1d );

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrixForBoundaryVertex(TDy tdy,
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

  ierr = Allocate_RealArray_2D(&Gmatrix, ndim, ndim);

  ierr = Allocate_RealArray_2D(&Fup, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = Allocate_RealArray_2D(&Cup, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = Allocate_RealArray_2D(&Fdn, ncells_in+ncells_bc, ncells_in+ncells_bc);
  ierr = Allocate_RealArray_2D(&Cdn, ncells_in+ncells_bc, ncells_in+ncells_bc);

  ierr = Allocate_RealArray_2D(&FupInxIn, ncells_in-1, ncells_in);
  ierr = Allocate_RealArray_2D(&FupBcxIn, 1, ncells_in);
  ierr = Allocate_RealArray_2D(&FdnInxIn, ncells_in-1, ncells_in);
  ierr = Allocate_RealArray_2D(&FdnBcxIn, 1, ncells_in);

  ierr = Allocate_RealArray_2D(&CupInxIn, ncells_in-1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&CupInxBc, ncells_in-1, ncells_bc  );
  ierr = Allocate_RealArray_2D(&CupBcxIn, 1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&CupBcxBc, 1, ncells_bc  );

  ierr = Allocate_RealArray_2D(&CdnInxIn, ncells_in-1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&CdnInxBc, ncells_in-1, ncells_bc  );
  ierr = Allocate_RealArray_2D(&CdnBcxIn, 1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&CdnBcxBc, 1, ncells_bc  );

  ierr = Allocate_RealArray_2D(&AInxIn, ncells_in-1, ncells_in-1);
  ierr = Allocate_RealArray_2D(&BInxIn, ncells_in-1, ncells_in  );

  ierr = Allocate_RealArray_2D(&AInxIninvBInxIn, ncells_in-1, ncells_in);

  ierr = Allocate_RealArray_1D(&AInxIn_1d, (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&lapack_mem_1d, (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&BInxIn_1d, (ncells_in-1)* ncells_in   );
  ierr = Allocate_RealArray_1D(&AInxIninv_1d, (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&AInxIninvBInxIn_1d, (ncells_in-1)* ncells_in   );
  ierr = Allocate_RealArray_1D(&CupInxIn_1d, (ncells_in-1)*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&CupInxIntimesAInxIninvBInxIn_1d,
                               (ncells_in-1)*(ncells_in)  );
  ierr = Allocate_RealArray_1D(&CupBcxIn_1d, (1          )*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&CdnBcxIn_1d, (1          )*(ncells_in-1));
  ierr = Allocate_RealArray_1D(&CupBcxIntimesAInxIninvBInxIn_1d,
                               (1          )*(ncells_in)  );
  ierr = Allocate_RealArray_1D(&CdnBcxIntimesAInxIninvBInxIn_1d,
                               (1          )*(ncells_in)  );

  ierr = Allocate_RealArray_1D(&DInxBc_1d, (ncells_in-1)* ncells_bc   );
  ierr = Allocate_RealArray_1D(&AInxIninvDInxBc_1d, (ncells_in-1)* ncells_bc   );
  ierr = Allocate_RealArray_1D(&CupInxIntimesAInxIninvDInxBc_1d,
                               (ncells_in-1)*(ncells_bc)  );
  ierr = Allocate_RealArray_1D(&CupBcxIntimesAInxIninvDInxBc_1d,
                               (1          )*(ncells_bc)  );
  ierr = Allocate_RealArray_1D(&CdnBcxIntimesAInxIninvDInxBc_1d,
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


  ierr = Deallocate_RealArray_2D(Gmatrix, ndim);

  ierr = Deallocate_RealArray_2D(Fup, ncells_in+ncells_bc);
  ierr = Deallocate_RealArray_2D(Cup, ncells_in+ncells_bc);
  ierr = Deallocate_RealArray_2D(Fdn, ncells_in+ncells_bc);
  ierr = Deallocate_RealArray_2D(Cdn, ncells_in+ncells_bc);

  ierr = Deallocate_RealArray_2D(FupInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(FupBcxIn, 1          );
  ierr = Deallocate_RealArray_2D(FdnInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(FdnBcxIn, 1          );

  ierr = Deallocate_RealArray_2D(CupInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CupInxBc, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CupBcxIn, 1          );
  ierr = Deallocate_RealArray_2D(CupBcxBc, 1          );

  ierr = Deallocate_RealArray_2D(CdnInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CdnInxBc, ncells_in-1);
  ierr = Deallocate_RealArray_2D(CdnBcxIn, 1          );
  ierr = Deallocate_RealArray_2D(CdnBcxBc, 1          );

  ierr = Deallocate_RealArray_2D(AInxIn, ncells_in-1);
  ierr = Deallocate_RealArray_2D(BInxIn, ncells_in-1);

  ierr = Deallocate_RealArray_2D(AInxIninvBInxIn, ncells_in-1);

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
  PetscReal *A1d, *B1d, *Cup1d, *AinvB1d, *CuptimesAinvB1d;
  PetscReal **Gmatrix;
  PetscInt idx, vertex_id;
  PetscErrorCode ierr;
  PetscBLASInt info, *pivots;
  PetscInt i, j, k, n, m, ndim, nfluxes;
  PetscScalar zero = 0.0, one = 1.0;

  PetscFunctionBegin;

  ndim      = 3;
  ncells    = vertex->num_internal_cells;
  vertex_id = vertex->id;
  nfluxes   = ncells * 3/2;

  ierr = Allocate_RealArray_2D(&Gmatrix, ndim   , ndim   ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Fup    , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Fdn    , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Cup    , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&Cdn    , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&A      , nfluxes, nfluxes); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&B      , nfluxes, ncells ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_2D(&AinvB  , nfluxes, ncells ); CHKERRQ(ierr);

  ierr = Allocate_RealArray_1D(&A1d            , nfluxes*nfluxes); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&B1d            , nfluxes*ncells ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&Cup1d          , nfluxes*nfluxes); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&AinvB1d        , nfluxes*ncells ); CHKERRQ(ierr);
  ierr = Allocate_RealArray_1D(&CuptimesAinvB1d, nfluxes*ncells ); CHKERRQ(ierr);

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
      
      PetscInt cell_1 = ReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[0]);
      PetscInt cell_2 = ReturnIndexInList(vertex->internal_cell_ids, ncells, face->cell_ids[1]);
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
      B1d[idx]= -Fup[i][j] + Fdn[i][j];
      idx++;
    }
  }

  m = nfluxes; n = nfluxes;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);

  LAPACKgetrf_(&m, &n, A1d, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                        "Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT,
                        "Bad LU factorization");

  ierr = PetscMemcpy(AinvB1d,B1d,sizeof(PetscScalar)*(nfluxes*ncells));
  CHKERRQ(ierr); // AinvB in col major

  // Solve AinvB = (A^-1 * B) by back-substitution
  m = nfluxes; n = ncells;
  LAPACKgetrs_("N", &m, &n, A1d, &m, pivots, AinvB1d, &m, &info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (C * AinvB)
  m = nfluxes; n = ncells; k = nfluxes;
  BLASgemm_("N","N", &m, &n, &k, &one, Cup1d, &m, AinvB1d, &m, &zero,
            CuptimesAinvB1d, &m);

  idx = 0;
  for (j=0; j<ncells; j++) {
    for (i=0; i<nfluxes; i++) {
      AinvB[i][j] = AinvB1d[idx];
      tdy->Trans[vertex_id][i][j] = CuptimesAinvB1d[idx] - Fup[i][j];
      idx++;
    }
  }

  // Free up the memory
  ierr = Deallocate_RealArray_2D(Gmatrix, ndim   ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fup    , nfluxes ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fdn    , nfluxes ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cup    , nfluxes ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cdn    , nfluxes ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(A      , nfluxes ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(B      , nfluxes ); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(AinvB  , nfluxes ); CHKERRQ(ierr);
  ierr = PetscFree(pivots                         ); CHKERRQ(ierr);

  free(A1d             );
  free(B1d             );
  free(Cup1d           );
  free(AinvB1d         );
  free(CuptimesAinvB1d );

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
  PetscReal **FupInxIn, **FdnInxIn; // InxIn: Internal flux with contribution from unknown internal pressure values
  PetscReal **FupBcxIn, **FdnBcxIn; // BcxIn: Boundary flux with contribution from unknown internal pressure values
  PetscReal **CupInxIn, **CdnInxIn; // InxIn: Internal flux with contribution from unknown internal pressure values
  PetscReal **CupInxBc, **CdnInxBc; // Inxbc: Internal flux with contribution from known boundary pressure values
  PetscReal **CupBcxIn, **CdnBcxIn; // BcxIn: Boundary flux with contribution from unknown internal pressure values
  PetscReal **CupBcxBc, **CdnBcxBc; // BcxIn: Boundary flux with contribution from known boundary pressure values

  PetscReal *AInxIninv_1d;
  PetscReal **AInxIn, **BInxIn, **AInxIninvBInxIn  ;
  PetscReal *AInxIn_1d, *BInxIn_1d, *DInxBc_1d, *AInxIninvBInxIn_1d, *AInxIninvDInxBc_1d;
  PetscReal *CupInxIn_1d, *CupInxIntimesAInxIninvBInxIn_1d, *CupInxIntimesAInxIninvDInxBc_1d;
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

  ierr = Allocate_RealArray_2D(&Gmatrix, ndim, ndim);

  ierr = Allocate_RealArray_2D(&Fup, nflux, npcen);
  ierr = Allocate_RealArray_2D(&Cup, nflux, npitf);
  ierr = Allocate_RealArray_2D(&Fdn, nflux, npcen);
  ierr = Allocate_RealArray_2D(&Cdn, nflux, npitf);

  ierr = Allocate_RealArray_2D(&FupInxIn, nflux_in, npcen);
  ierr = Allocate_RealArray_2D(&FupBcxIn, nflux_bc, npcen);
  ierr = Allocate_RealArray_2D(&FdnInxIn, nflux_in, npcen);
  ierr = Allocate_RealArray_2D(&FdnBcxIn, nflux_bc, npcen);

  ierr = Allocate_RealArray_2D(&CupInxIn, nflux_in, npitf_in);
  ierr = Allocate_RealArray_2D(&CupInxBc, nflux_in, npitf_bc);
  ierr = Allocate_RealArray_2D(&CupBcxIn, nflux_bc, npitf_in);
  ierr = Allocate_RealArray_2D(&CupBcxBc, nflux_bc, npitf_bc);

  ierr = Allocate_RealArray_2D(&CdnInxIn, nflux_in, npitf_in);
  ierr = Allocate_RealArray_2D(&CdnInxBc, nflux_in, npitf_bc);
  ierr = Allocate_RealArray_2D(&CdnBcxIn, nflux_bc, npitf_in);
  ierr = Allocate_RealArray_2D(&CdnBcxBc, nflux_bc, npitf_bc);

  ierr = Allocate_RealArray_2D(&AInxIn, nflux_in, npitf_in);
  ierr = Allocate_RealArray_2D(&BInxIn, nflux_in, npcen);
  ierr = Allocate_RealArray_2D(&AInxIninvBInxIn, nflux_in, npcen);

  ierr = Allocate_RealArray_1D(&AInxIn_1d    , nflux_in*npitf_in);
  ierr = Allocate_RealArray_1D(&lapack_mem_1d, nflux_in*npitf_in);
  ierr = Allocate_RealArray_1D(&BInxIn_1d    , nflux_in*npcen   );

  ierr = Allocate_RealArray_1D(&AInxIninv_1d      , nflux_in*npitf_in);
  ierr = Allocate_RealArray_1D(&AInxIninvBInxIn_1d, nflux_in*npcen   );

  ierr = Allocate_RealArray_1D(&CupInxIn_1d                    , nflux_in*npitf_in);
  ierr = Allocate_RealArray_1D(&CupInxIntimesAInxIninvBInxIn_1d, nflux_in*npcen);

  ierr = Allocate_RealArray_1D(&CupBcxIn_1d, nflux_bc*npitf_in);
  ierr = Allocate_RealArray_1D(&CdnBcxIn_1d, nflux_bc*npitf_in);

  ierr = Allocate_RealArray_1D(&CupBcxIntimesAInxIninvBInxIn_1d, nflux_bc*npcen);
  ierr = Allocate_RealArray_1D(&CdnBcxIntimesAInxIninvBInxIn_1d, nflux_bc*npcen);

  ierr = Allocate_RealArray_1D(&DInxBc_1d, nflux_in*npitf_bc);

  ierr = Allocate_RealArray_1D(&AInxIninvDInxBc_1d, nflux_in*npitf_bc);
  ierr = Allocate_RealArray_1D(&CupInxIntimesAInxIninvDInxBc_1d, nflux_in*npitf_bc);
  ierr = Allocate_RealArray_1D(&CupBcxIntimesAInxIninvDInxBc_1d, nflux_bc*npitf_bc);
  ierr = Allocate_RealArray_1D(&CdnBcxIntimesAInxIninvDInxBc_1d, nflux_bc*npitf_bc);

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
    if (npcen == 2) {
      if (idx_interface_p0>=npcen) idx_interface_p0--;
      if (idx_interface_p1>=npcen) idx_interface_p1--;
      if (idx_interface_p2>=npcen) idx_interface_p2--;
    }

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
      Flux     =         F            *     Pcenter      +                         C                         *      P_interface
    _       _     _                  _   _           _       _                                             _    _               _
   |         |   |                    | |             |     |                                               |  |  P_(nptif_in,1) |
   | Flux_in |   | F_(nflux_in,npcen) | |             |     | C_(nflux_in,npitf_in)   C_(nflux_in,npitf_bc) |  |                 |
   |         | = |                    | | P_(npcen,1) |  +  |                                               |  |                 |
   | Flux_bc |   | F_(nflux_bc,npcen) | |             |     | C_(nflux_bc,npitf_in)   C_(nflux_in,npitf_bc) |  |  P_(nptif_bc,1) |
   |_       _|   |_                  _| |_           _|     |_                                             _|  |_               _|

  */

  for (j=0; j<npcen; j++) {
    for (i=0; i<nflux_in; i++) {
      FupInxIn[i][j] = Fup[i][j];
      FdnInxIn[i][j] = Fdn[i][j];
    }
    for (i=0; i<nflux_bc; i++) {
      FupBcxIn[i][j] = Fup[i+nflux_in][j];
      FdnBcxIn[i][j] = Fdn[i+nflux_in][j];
    }
  }

  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_in; i++) {
      CupInxIn[i][j] = Cup[i][j];
      CdnInxIn[i][j] = Cdn[i][j];
    }
    for (i=0; i<nflux_bc; i++) {
      CupBcxIn[i][j] = Cup[i+npitf_in][j];
      CdnBcxIn[i][j] = Cdn[i+npitf_in][j];
    }
  }

  idx = 0;
  for (j=0; j<nflux_bc*2; j++) {
    for (i=0; i<nflux_in; i++) {
      CupInxBc[i][j] = Cup[i][j+npitf_in];
      CdnInxBc[i][j] = Cdn[i][j+npitf_in];
      DInxBc_1d[idx]   = CupInxBc[i][j] - CdnInxBc[i][j];
      idx++;
    }
    for (i=0; i<nflux_bc; i++) {
      CupBcxBc[i][j] = Cup[i+npitf_in][j+npitf_in];
      CdnBcxBc[i][j] = Cdn[i+npitf_in][j+npitf_in];
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_in; i++) {
      AInxIn[i][j]     = -CupInxIn[i][j] + CdnInxIn[i][j];
      AInxIn_1d[idx]   = -CupInxIn[i][j] + CdnInxIn[i][j];
      CupInxIn_1d[idx] = CupInxIn[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_in; j++) {
    for (i=0; i<nflux_bc; i++) {
      CupBcxIn_1d[idx] = CupBcxIn[i][j];
      CdnBcxIn_1d[idx] = CdnBcxIn[i][j];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npcen; j++) {
    for (i=0; i<nflux_in; i++) {
      BInxIn[i][j]   = -FupInxIn[i][j] + FdnInxIn[i][j];
      BInxIn_1d[idx] = -FupInxIn[i][j] + FdnInxIn[i][j];
      idx++;
    }
  }

  n = nflux_in; m = npitf_in;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt), &pivots); CHKERRQ(ierr);

  LAPACKgetrf_(&m, &n, AInxIn_1d, &m, pivots, &info);
  if (info<0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  ierr = PetscMemcpy(AInxIninvBInxIn_1d, BInxIn_1d,sizeof(PetscScalar)*(nflux_in*npcen   )); CHKERRQ(ierr);
  ierr = PetscMemcpy(AInxIninv_1d      , AInxIn_1d,sizeof(PetscScalar)*(nflux_in*nflux_in)); CHKERRQ(ierr);
  ierr = PetscMemcpy(AInxIninvDInxBc_1d, DInxBc_1d,sizeof(PetscScalar)*(nflux_in*npitf_bc)); CHKERRQ(ierr);

  // Solve AinvB = (A^-1) by back-substitution
  n = nflux_in;
  PetscInt nn = n*n;
  LAPACKgetri_(&n, AInxIninv_1d, &n, pivots, lapack_mem_1d, &nn, &info);

  // Compute (Ainv*B)
  m = npitf_in; n = npcen; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, AInxIninv_1d, &m, BInxIn_1d, &k, &zero, AInxIninvBInxIn_1d, &m);

  // Compute (Ainv*D)
  m = npitf_in; n = npitf_bc; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, AInxIninv_1d, &m, DInxBc_1d, &k, &zero, AInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*B) for internal flux
  m = npitf_in; n = npcen; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, CupInxIn_1d, &m, AInxIninvBInxIn_1d, &k, &zero, CupInxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*B) for up boundary flux
  m = nflux_bc; n = npcen; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, CupBcxIn_1d, &m, AInxIninvBInxIn_1d, &k, &zero, CupBcxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*B) for down boundary flux
  m = nflux_bc; n = npcen; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, CdnBcxIn_1d, &m, AInxIninvBInxIn_1d, &k, &zero, CdnBcxIntimesAInxIninvBInxIn_1d, &m);

  // Compute C*(Ainv*D) for internal flux
  m = nflux_in; n = npitf_bc; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, CupInxIn_1d, &m, AInxIninvDInxBc_1d, &k, &zero, CupInxIntimesAInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*D) for up boundary flux
  m = nflux_bc; n = npitf_bc; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, CupBcxIn_1d, &m, AInxIninvDInxBc_1d, &k, &zero, CupBcxIntimesAInxIninvDInxBc_1d, &m);

  // Compute C*(Ainv*D) for down boundary flux
  m = nflux_bc; n = npitf_bc; k = nflux_in;
  BLASgemm_("N","N", &m, &n, &k, &one, CdnBcxIn_1d, &m, AInxIninvDInxBc_1d, &k, &zero, CdnBcxIntimesAInxIninvDInxBc_1d, &m);

  /*
        _          _     _                                                                        _   _              _
       |            |   |                                                                          | |                |
       | Flux_in    |   | (-Fu^(IxI) + Cu^(IxI) x Inv(A) x B)   (Cu^(IxB) + Cu^(IxI) x Inv(A) x D) | |                |
       |            |   |                                                                          | | P_(npcen,1)    |
   T = | Flux_bc_up | = | (-Fu^(BxI) + Cu^(BxI) x Inv(A) x B)   (Cu^(BxB) + Cu^(BxI) x Inv(A) x D) | |                |
       |            |   |                                                                          | | P_(nptif_bc,1) |
       | Flux_bc_dn |   | (-Fd^(BxI) + Cu^(BxI) x Inv(A) x B)   (Cd^(BxB) + Cu^(BxI) x Inv(A) x D) | |                |
       |_          _|   |_                                                                        _| |_              _|

  */

  idx = 0;
  for (j=0; j<npcen; j++) {
    for (i=0; i<nflux_in; i++) {
      tdy->Trans[vertex_id][i][j] = -FupInxIn[i][j] + CupInxIntimesAInxIninvBInxIn_1d[idx];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_bc; j++) {
    for (i=0; i<nflux_in; i++) {
      tdy->Trans[vertex_id][i][j+npcen] = CupInxBc[i][j] + CupInxIntimesAInxIninvDInxBc_1d[idx];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npcen; j++) {
    for (i=0; i<nflux_bc; i++) {
      tdy->Trans[vertex_id][i+nflux_in         ][j] = -FupBcxIn[i][j] + CupBcxIntimesAInxIninvBInxIn_1d[idx];
      tdy->Trans[vertex_id][i+nflux_in+nflux_bc][j] = -FdnBcxIn[i][j] + CdnBcxIntimesAInxIninvBInxIn_1d[idx];
      idx++;
    }
  }

  idx = 0;
  for (j=0; j<npitf_bc; j++) {
    for (i=0; i<nflux_bc; i++) {
      tdy->Trans[vertex_id][i+nflux_in         ][j+npcen] = CupBcxBc[i][j] + CupBcxIntimesAInxIninvDInxBc_1d[idx];
      tdy->Trans[vertex_id][i+nflux_in+nflux_bc][j+npcen] = CdnBcxBc[i][j] + CdnBcxIntimesAInxIninvDInxBc_1d[idx];
      idx++;
    }
  }

  ierr = Deallocate_RealArray_2D(Gmatrix, ndim); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fup, nflux); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cup, nflux); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Fdn, nflux); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(Cdn, nflux); CHKERRQ(ierr);

  ierr = Deallocate_RealArray_2D(FupInxIn, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(FupBcxIn, nflux_bc); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(FdnInxIn, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(FdnBcxIn, nflux_bc); CHKERRQ(ierr);

  ierr = Deallocate_RealArray_2D(CupInxIn, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(CupInxBc, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(CupBcxIn, nflux_bc); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(CupBcxBc, nflux_bc); CHKERRQ(ierr);

  ierr = Deallocate_RealArray_2D(CdnInxIn, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(CdnInxBc, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(CdnBcxIn, nflux_bc); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(CdnBcxBc, nflux_bc); CHKERRQ(ierr);

  ierr = Deallocate_RealArray_2D(AInxIn, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(BInxIn, nflux_in); CHKERRQ(ierr);
  ierr = Deallocate_RealArray_2D(AInxIninvBInxIn, nflux_in); CHKERRQ(ierr);

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
PetscErrorCode ComputeTransmissibilityMatrix3DMesh(TDy tdy) {

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
PetscErrorCode ComputeTransmissibilityMatrix2DMesh(TDy tdy) {

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

      ierr = ComputeTransmissibilityMatrixForInternalVertex(tdy, vertex, cells);
      CHKERRQ(ierr);
    } else {
      if (vertex->num_internal_cells > 1) {
        ierr = ComputeTransmissibilityMatrixForBoundaryVertex(tdy, vertex, cells);
        CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix(TDy tdy) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = ComputeTransmissibilityMatrix2DMesh(tdy); CHKERRQ(ierr);
    break;
  case 3:
    ierr = ComputeTransmissibilityMatrix3DMesh(tdy); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalCells(TDy tdy) {

  PetscErrorCode ierr;
  DM             dm;
  Vec            junkVec;
  PetscInt       junkInt;
  PetscInt       gref;
  PetscInt       cStart, cEnd, c;
  TDy_cell       *cells;

  PetscFunctionBegin;

  dm = tdy->dm;
  cells = tdy->mesh->cells;

  // Once needs to atleast haved called a DMCreateXYZ() before using DMPlexGetPointGlobal()
  ierr = DMCreateGlobalVector(dm, &junkVec); CHKERRQ(ierr);
  ierr = VecDestroy(&junkVec); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      cells[c].is_local = PETSC_TRUE;
      cells[c].global_id = gref;
    } else {
      cells[c].is_local = PETSC_FALSE;
      cells[c].global_id = -gref-1;
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalVertices(TDy tdy) {

  PetscInt       ivertex, icell, c;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  vertices = mesh->vertices;

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    for (c=0; c<vertices[ivertex].num_internal_cells; c++) {
      icell = vertices[ivertex].internal_cell_ids[c];
      if (cells[icell].is_local) vertices[ivertex].is_local = PETSC_TRUE;
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalEdges(TDy tdy) {

  PetscInt iedge, icell_1, icell_2;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_edge *edges;

  PetscFunctionBegin;

  mesh  = tdy->mesh;
  cells = mesh->cells;
  edges = mesh->edges;

  for (iedge=0; iedge<mesh->num_edges; iedge++) {

    if (!edges[iedge].is_internal) { // Is it a boundary edge?

      // Determine the cell ID for the boundary edge
      if (edges[iedge].cell_ids[0] != -1) icell_1 = edges[iedge].cell_ids[0];
      else                                icell_1 = edges[iedge].cell_ids[1];

      // Is the cell locally owned?
      if (cells[icell_1].is_local) edges[iedge].is_local = PETSC_TRUE;

    } else { // An internal edge

      // Save the two cell ID
      icell_1 = edges[iedge].cell_ids[0];
      icell_2 = edges[iedge].cell_ids[1];

      if (cells[icell_1].is_local && cells[icell_2].is_local) { // Are both cells locally owned?

        edges[iedge].is_local = PETSC_TRUE;

      } else if (cells[icell_1].is_local && !cells[icell_2].is_local) { // Is icell_1 locally owned?

        // Is the global ID of icell_1 lower than global ID of icell_2?
        if (cells[icell_1].global_id < cells[icell_2].global_id) edges[iedge].is_local = PETSC_TRUE;

      } else if (!cells[icell_1].is_local && cells[icell_2].is_local) { // Is icell_2 locally owned

        // Is the global ID of icell_2 lower than global ID of icell_1?
        if (cells[icell_2].global_id < cells[icell_1].global_id) edges[iedge].is_local = PETSC_TRUE;

      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode IdentifyLocalFaces(TDy tdy) {

  PetscInt iface, icell_1, icell_2;
  TDy_mesh *mesh;
  TDy_cell *cells;
  TDy_face *faces;

  PetscFunctionBegin;

  mesh  = tdy->mesh;
  cells = mesh->cells;
  faces = mesh->faces;

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (!faces[iface].is_internal) { // Is it a boundary face?

      // Determine the cell ID for the boundary edge
      if (faces[iface].cell_ids[0] != -1) icell_1 = faces[iface].cell_ids[0];
      else                                icell_1 = faces[iface].cell_ids[1];

      // Is the cell locally owned?
      if (cells[icell_1].is_local) faces[iface].is_local = PETSC_TRUE;

    } else { // An internal face

      // Save the two cell ID
      icell_1 = faces[iface].cell_ids[0];
      icell_2 = faces[iface].cell_ids[1];

      if (cells[icell_1].is_local && cells[icell_2].is_local) { // Are both cells locally owned?

        faces[iface].is_local = PETSC_TRUE;

      } else if (cells[icell_1].is_local && !cells[icell_2].is_local) { // Is icell_1 locally owned?

        // Is the global ID of icell_1 lower than global ID of icell_2?
        if (cells[icell_1].global_id < cells[icell_2].global_id) faces[iface].is_local = PETSC_TRUE;

      } else if (!cells[icell_1].is_local && cells[icell_2].is_local) { // Is icell_2 locally owned

        // Is the global ID of icell_2 lower than global ID of icell_1?
        if (cells[icell_2].global_id < cells[icell_1].global_id) faces[iface].is_local = PETSC_TRUE;

      }
    }
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOInitialize(TDy tdy) {

  PetscErrorCode ierr;
  MPI_Comm       comm;
  DM             dm;
  PetscInt       dim;

  PetscFunctionBegin;

  dm = tdy->dm;

  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  tdy->mesh = (TDy_mesh *) malloc(sizeof(TDy_mesh));

  ierr = AllocateMemoryForMesh(dm, tdy->mesh); CHKERRQ(ierr);

  ierr = Allocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells,
                               tdy->mesh->num_vertices, 3, 3); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = Allocate_RealArray_3D(&tdy->Trans, tdy->mesh->num_vertices, 5, 5);
    CHKERRQ(ierr);
    ierr = PetscMalloc(tdy->mesh->num_edges*sizeof(PetscReal),
                     &(tdy->vel )); CHKERRQ(ierr);
    ierr = Initialize_RealArray_1D(tdy->vel, tdy->mesh->num_edges, 0.0); CHKERRQ(ierr);

    break;
  case 3:
    ierr = Allocate_RealArray_3D(&tdy->Trans, tdy->mesh->num_vertices, 12, 12);
    CHKERRQ(ierr);
    ierr = PetscMalloc(tdy->mesh->num_faces*sizeof(PetscReal),
                     &(tdy->vel )); CHKERRQ(ierr);
    ierr = Initialize_RealArray_1D(tdy->vel, tdy->mesh->num_faces, 0.0); CHKERRQ(ierr);

    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in TDyMPFAOInitialize");
    break;
  }

  ierr = BuildMesh(tdy); CHKERRQ(ierr);

  ierr = ComputeGMatrix(tdy); CHKERRQ(ierr);

  ierr = ComputeTransmissibilityMatrix(tdy); CHKERRQ(ierr);

  /* Setup the section, 1 dof per cell */
  PetscSection sec;
  PetscInt p, pStart, pEnd;
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, 1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0, 1); CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  ierr = IdentifyLocalCells(tdy); CHKERRQ(ierr);
  ierr = IdentifyLocalVertices(tdy); CHKERRQ(ierr);
  ierr = IdentifyLocalEdges(tdy); CHKERRQ(ierr);
  if (dim == 3) {
    ierr = IdentifyLocalFaces(tdy); CHKERRQ(ierr);
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
  cells    = mesh->cells;
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          if (col<0) col = -col - 1;
          value = tdy->Trans[vertex_id][icell][icol];
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = TDyMPFAOComputeSystem_InternalVertices_2DMesh(tdy,K,F); CHKERRQ(ierr);
    break;
  case 3:
    ierr = TDyMPFAOComputeSystem_InternalVertices_3DMesh(tdy,K,F); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

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
  cells    = mesh->cells;
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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
      
      if (cells[icell_from].is_local) {
        row   = cells[icell_from].global_id;
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          value = tdy->Trans[vertex_id][icell][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell][icol + vertex->num_internal_cells] *
          pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cells[icell_to].is_local) {
        row   = cells[icell_to].global_id;
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
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
      
      if (cell_ids_from_to[icell][0]>-1 && cells[cell_ids_from_to[icell][0]].is_local) {
        row   = cells[cell_ids_from_to[icell][0]].global_id;
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol +
                                                                            vertex->num_internal_cells] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol];
          ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cell_ids_from_to[icell][1]>-1 && cells[cell_ids_from_to[icell][1]].is_local) {
        row   = cells[cell_ids_from_to[icell][1]].global_id;
        
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][icell+vertex->num_internal_cells-1][icol +
                                                                            vertex->num_internal_cells] * pBoundary[icol];
          ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        for (icol=0; icol<vertex->num_internal_cells; icol++) {
          col   = cells[vertex->internal_cell_ids[icol]].global_id;
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_2DMesh(tdy,K,F); CHKERRQ(ierr);
    break;
  case 3:
    ierr = TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy,K,F); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

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
  cells    = mesh->cells;
  vertices = mesh->vertices;
  edges    = mesh->edges;

  ierr = DMPlexGetDepthStratum (dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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
    row   = cells[vertex->internal_cell_ids[0]].global_id;
    col   = cells[vertex->internal_cell_ids[0]].global_id;
    if (cells[icell].is_local) {ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);}
    
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (icell=0; icell<vertex->num_boundary_cells; icell++) {
      
      if (cell_ids_from_to[icell][0]>-1 && cells[cell_ids_from_to[icell][0]].is_local) {
        row   = cells[cell_ids_from_to[icell][0]].global_id;
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = Gmatrix[icell][icol] * pBoundary[icol];
          ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
      }
      
      if (cell_ids_from_to[icell][1]>-1 && cells[cell_ids_from_to[icell][1]].is_local) {
        row   = cells[cell_ids_from_to[icell][1]].global_id;
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_2DMesh(tdy,K,F); CHKERRQ(ierr);
    break;
  case 3:
    ierr = TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy,K,F); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem(TDy tdy,Mat K,Vec F) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  PetscInt       icell;
  PetscInt       row;
  PetscReal      value;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm       = tdy->dm;
  mesh     = tdy->mesh;
  cells    = mesh->cells;

  ierr = MatZeroEntries(K);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = TDyMPFAOComputeSystem_InternalVertices(tdy,K,F); CHKERRQ(ierr);
  ierr = TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(tdy,K,F); CHKERRQ(ierr);
  ierr = TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(tdy,K,F); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscReal f;
  if (tdy->ops->computeforcing) {
    for (icell=0; icell<tdy->mesh->num_cells; icell++) {
      if (cells[icell].is_local) {
        ierr = (*tdy->ops->computeforcing)(tdy, &(tdy->X[icell*dim]), &f, tdy->forcingctx);CHKERRQ(ierr);
        value = f * cells[icell].volume;
        row = cells[icell].global_id;
        ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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

PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices_3DMesh(TDy tdy, Vec U) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces;
  TDy_subcell    *subcell;
  PetscInt       ivertex, icell_from, icell_to;
  PetscInt       irow, icol, vertex_id;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscScalar    *u;
  Vec            localU;
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    vertex    = &vertices[ivertex];
    vertex_id = vertex->id;

    if (vertex->num_boundary_cells == 0) {

      PetscInt    nflux_in = vertex->num_internal_cells;
      PetscScalar Pcomputed[vertex->num_internal_cells];
      PetscScalar Vcomputed[vertex->num_internal_cells];

      // Save local pressure stencil and initialize veloctiy
      PetscInt icell;
      for (icell=0; icell<nflux_in; icell++) {
        Pcomputed[icell] = u[vertex->internal_cell_ids[icell]];
        Vcomputed[icell] = 0.0;
      }

      // F = T*P
      for (irow=0; irow<nflux_in; irow++) {

        PetscInt face_id = vertex->trans_row_face_ids[irow];
        TDy_face *face = &faces[face_id];

        if (!face->is_local) continue;

        icell_from = face->cell_ids[0];
        icell_to   = face->cell_ids[1];

        TDy_cell *cell = &cells[icell_from];


        PetscInt i, iface=-1, isubcell = -1;

        for (i=0; i<vertex->num_internal_cells;i++){
          if (vertex->internal_cell_ids[i] == icell_from) {
            isubcell = vertex->subcell_ids[i];
            break;
          }
        }

        subcell = &cell->subcells[isubcell];

        for (i=0; i<subcell->num_faces;i++) {
          if (subcell->face_ids[i] == face_id) {
            iface = i;
            break;
          }
        }

        for (icol=0; icol<vertex->num_internal_cells; icol++) {

          Vcomputed[irow] = tdy->Trans[vertex_id][irow][icol] *
                            Pcomputed[icol]                   *
                            subcell->face_area[iface];
          
        }
        tdy->vel[face_id] += Vcomputed[irow];
      }
    }
  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(TDy tdy, Vec U) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces;
  TDy_subcell    *subcell;
  PetscInt       ivertex, icell, icell_from, icell_to;
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

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
    numBoundary = 0;

    for (irow=0; irow<ncells; irow++){
      icell = vertex->internal_cell_ids[irow];
      PetscInt isubcell = vertex->subcell_ids[irow];

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

      if (!face->is_local) continue;

      icell_from = face->cell_ids[0];
      icell_to   = face->cell_ids[1];
      icell = vertex->internal_cell_ids[irow];

      if (cells[icell_from].is_local) {

        cell = &cells[icell_from];
        PetscInt i, iface=-1, isubcell = -1;

        for (i=0; i<vertex->num_internal_cells;i++){
          if (vertex->internal_cell_ids[i] == icell_from) {
            isubcell = vertex->subcell_ids[i];
            break;
          }
        }

        subcell = &cell->subcells[isubcell];

        for (i=0; i<subcell->num_faces;i++) {
          if (subcell->face_ids[i] == face_id) {
            iface = i;
            break;
          }
        }

        // +T_00
        for (icol=0; icol<npcen; icol++) {
          value = tdy->Trans[vertex_id][irow][icol] *
                  u[cells[vertex->internal_cell_ids[icol]].id] *
                  subcell->face_area[iface];
          tdy->vel[face_id] += value;
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
        }
        
        // -T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol]*subcell->face_area[iface];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += -value;
        }
      }
      
      if (cells[icell_to].is_local) {

        cell = &cells[icell_to];
        PetscInt i, iface=-1, isubcell = -1;

        for (i=0; i<vertex->num_internal_cells;i++){
          if (vertex->internal_cell_ids[i] == icell_to) {
            isubcell = vertex->subcell_ids[i];
            break;
          }
        }

        subcell = &cell->subcells[isubcell];

        for (i=0; i<subcell->num_faces;i++) {
          if (subcell->face_ids[i] == face_id) {
            iface = i;
            break;
          }
        }

        // -T_00
        for (icol=0; icol<npcen; icol++) {
          value = tdy->Trans[vertex_id][irow][icol] *
                  u[cells[vertex->internal_cell_ids[icol]].id] *
                  subcell->face_area[iface];
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += -value;
        }
        
        // +T_01 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = tdy->Trans[vertex_id][irow][icol + npcen] * pBoundary[icol]*subcell->face_area[iface];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += value;
        }
      }
    }
    
    // For fluxes through boundary edges, only add contribution to the vector
    for (irow=0; irow<nflux_bc*2; irow++) {

      PetscInt face_id = vertex->trans_row_face_ids[irow + nflux_in];
      TDy_face *face = &faces[face_id];

      if (!face->is_local) continue;

      icell_from = face->cell_ids[0];
      icell_to   = face->cell_ids[1];

      if (icell_from>-1 && cells[icell_from].is_local) {
        cell = &cells[icell_from];

        PetscInt i, iface=-1, isubcell = -1;

        for (i=0; i<vertex->num_internal_cells;i++){
          if (vertex->internal_cell_ids[i] == icell_from) {
            isubcell = vertex->subcell_ids[i];
            break;
          }
        }
        
        subcell = &cell->subcells[isubcell];

        for (i=0; i<subcell->num_faces;i++) {
          if (subcell->face_ids[i] == face_id) {
            iface = i;
            break;
          }
        }

        // +T_10
        for (icol=0; icol<npcen; icol++) {
          value = tdy->Trans[vertex_id][irow+nflux_in][icol]*
                  u[cells[vertex->internal_cell_ids[icol]].id] *
                  subcell->face_area[iface];
          //ierr = MatSetValue(K, row, col, value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += value;
        }

        //  -T_11 * Pbc
        for (icol=0; icol<npitf_bc; icol++) {
          value = tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol]*subcell->face_area[iface];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += -value;
        }
        
      }
      
      if (icell_to>-1 && cells[icell_to].is_local) {
        cell = &cells[icell_to];
        
        PetscInt i, iface=-1, isubcell = -1;

        for (i=0; i<vertex->num_internal_cells;i++){
          if (vertex->internal_cell_ids[i] == icell_to) {
            isubcell = vertex->subcell_ids[i];
            break;
          }
        }
        
        subcell = &cell->subcells[isubcell];

        for (i=0; i<subcell->num_faces;i++) {
          if (subcell->face_ids[i] == face_id) {
            iface = i;
            break;
          }
        }

        // -T_10
        for (icol=0; icol<npcen; icol++) {
          value = tdy->Trans[vertex_id][irow+nflux_in][icol]*
                  u[cells[vertex->internal_cell_ids[icol]].id] *
                  subcell->face_area[iface];
          //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += -value;
        }
        
        //  +T_11 * Pbc
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value = tdy->Trans[vertex_id][irow+nflux_in][icol+npcen] * pBoundary[icol]*subcell->face_area[iface];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
          tdy->vel[face_id] += value;
        }
      }
    }
  }

  ierr = VecRestoreArray(localU,&u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(TDy tdy, Vec U) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_vertex     *vertices, *vertex;
  TDy_face       *faces;
  TDy_subcell    *subcell;
  PetscInt       ivertex, icell;
  PetscInt       icol, row, iface, isubcell;
  PetscReal      value;
  PetscInt       vStart, vEnd;
  PetscInt       fStart, fEnd;
  PetscInt       dim;
  PetscReal      **Gmatrix;
  PetscReal      sign;
  PetscInt       j;
  PetscScalar    *u;
  Vec            localU;
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

  ierr = Allocate_RealArray_2D(&Gmatrix, dim, dim);

  // TODO: Save localU
  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {
    
    vertex    = &vertices[ivertex];
    
    if (vertex->num_boundary_cells == 0) continue;
    if (vertex->num_internal_cells > 1)  continue;
    
    // Vertex is on the boundary
    
    // For boundary edges, save following information:
    //  - Dirichlet pressure value
    //  - Cell IDs connecting the boundary edge in the direction of unit normal
    
    icell    = vertex->internal_cell_ids[0];
    isubcell = vertex->subcell_ids[0];

    cell = &cells[icell];
    subcell = &cell->subcells[isubcell];

    PetscScalar pBoundary[subcell->num_faces];

    ierr = ExtractSubGmatrix(tdy, icell, isubcell, dim, Gmatrix);

    for (iface=0; iface<subcell->num_faces; iface++) {
      
      TDy_face *face = &faces[subcell->face_ids[iface]];

      PetscInt f;
      f = face->id + fStart;
      ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]), &pBoundary[iface], tdy->dirichletvaluectx);CHKERRQ(ierr);

    }

    for (iface=0; iface<subcell->num_faces; iface++) {

      TDy_face *face = &faces[subcell->face_ids[iface]];

      if (!face->is_local) continue;

      row = face->cell_ids[0];
      if (row>-1) sign = -1.0;
      else        sign = +1.0;

      value = 0.0;
      for (j=0; j<dim; j++) {
        value += sign*Gmatrix[iface][j]*(pBoundary[j] - u[icell])*subcell->face_area[iface];
      }

      //ierr = MatSetValue(K, row, col, -value, ADD_VALUES); CHKERRQ(ierr);
      // Should it be '-value' or 'value'?
      tdy->vel[face->id] += -value;

    }

    // For fluxes through boundary edges, only add contribution to the vector
    for (iface=0; iface<subcell->num_faces; iface++) {

      TDy_face *face = &faces[subcell->face_ids[iface]];

      if (!face->is_local) continue;

      row = face->cell_ids[0];
      PetscInt cell_from = face->cell_ids[0];
      PetscInt cell_to   = face->cell_ids[1];

      if (cell_from>-1 && cells[cell_from].is_local) {
        row   = cells[cell_from].global_id;
        value = 0.0;
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value += Gmatrix[iface][icol] * (pBoundary[icol] - u[icell])*subcell->face_area[iface];
          //ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
        }
        // Should it be '-value' or 'value'?
        tdy->vel[face->id] += value;

      }

      if (cell_to>-1 && cells[cell_to].is_local) {
        row   = cells[cell_to].global_id;
        value = 0.0;
        for (icol=0; icol<vertex->num_boundary_cells; icol++) {
          value += Gmatrix[iface][icol] * (pBoundary[icol] - u[icell])*subcell->face_area[iface];
          //ierr = VecSetValue(F, row, -value, ADD_VALUES); CHKERRQ(ierr);
        }
        // Should it be '-value' or 'value'?
        tdy->vel[face->id] += -value;
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

  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, U); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, U); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, U); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAORecoverVelocity(TDy tdy, Vec U) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = TDyMPFAORecoverVelocity_2DMesh(tdy,U); CHKERRQ(ierr);
    break;
  case 3:
    ierr = TDyMPFAORecoverVelocity_3DMesh(tdy,U); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscReal TDyMPFAOPressureNorm(TDy tdy, Vec U) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  PetscScalar    *u;
  Vec            localU;
  PetscInt       dim;
  PetscInt       icell;
  PetscReal      norm, norm_sum, pressure;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = mesh->cells;

  if (! tdy->ops->computedirichletvalue) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the dirichlet function with TDySetDirichletValueFunction");
  }

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  norm_sum = 0.0;
  norm     = 0.0;

  for (icell=0; icell<mesh->num_cells; icell++) {

    cell = &(cells[icell]);
    if (!cell->is_local) continue;

    ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[icell*dim]), &pressure, tdy->dirichletvaluectx);CHKERRQ(ierr);
    norm += (PetscSqr(pressure - u[icell])) * cell->volume;
  }

  ierr = VecRestoreArray(localU, &u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);

  norm_sum = PetscSqrtReal(norm_sum);

  PetscFunctionReturn(norm_sum);
}

/* -------------------------------------------------------------------------- */
PetscReal TDyMPFAOVelocityNorm_2DMesh(TDy tdy) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_edge       *edges, *edge;
  TDy_cell       *cells, *cell;
  PetscInt       dim;
  PetscInt       icell, iedge, edge_id;
  PetscInt       fStart, fEnd;
  PetscReal      norm, norm_sum, vel_normal;
  PetscReal      vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = mesh->cells;
  edges = mesh->edges;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  norm_sum = 0.0;
  norm     = 0.0;

  for (icell=0; icell<mesh->num_cells; icell++) {

    cell = &(cells[icell]);

    if (!cell->is_local) continue;

    for (iedge=0; iedge<cell->num_edges; iedge++) {
      edge_id = cell->edge_ids[iedge];
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

/* -------------------------------------------------------------------------- */
PetscReal TDyMPFAOVelocityNorm(TDy tdy) {

  PetscFunctionBegin;

  PetscInt dim;
  PetscErrorCode ierr;
  PetscReal norm_sum = 0.0;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    norm_sum = TDyMPFAOVelocityNorm_2DMesh(tdy);
    break;
  case 3:
    //norm_sum = TDyMPFAOVelocityNorm_3DMesh(tdy);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(norm_sum);
}
