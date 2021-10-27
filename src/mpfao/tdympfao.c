#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdydiscretization.h>
#include <petscblaslapack.h>

static PetscErrorCode ComputeGMatrix_MPFAO(TDyMPFAO* mpfao, DM dm,
                                           MaterialProp* matprop) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt dim,icell;
  PetscErrorCode ierr;

  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDySubcell *subcells = &mesh->subcells;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // extract permeability tensor
    PetscReal K[3][3];

    for (PetscInt ii=0; ii<dim; ii++) {
      for (PetscInt jj=0; jj<dim; jj++) {
        K[ii][jj] = matprop->K0[icell*dim*dim + ii*dim + jj];
      }
    }

    // extract thermal conductivity tensor
    PetscReal Kappa[3][3];
    if (mpfao->Temp_subc_Gmatrix) { // TH
      for (PetscInt ii=0; ii<dim; ii++) {
        for (PetscInt jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = matprop->Kappa0[icell*dim*dim + ii*dim + jj];
        }
      }
    } // TH

    for (PetscInt isubcell=0; isubcell<cells->num_subcells[icell]; isubcell++) {

      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

      PetscInt *subcell_face_ids, subcell_num_faces;
      PetscReal *subcell_face_areas;
      ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);
      ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &subcell_face_areas, &subcell_num_faces); CHKERRQ(ierr);

      for (PetscInt ii=0;ii<subcell_num_faces;ii++) {

        PetscInt face_id = subcell_face_ids[ii];
        PetscReal area = subcell_face_areas[ii];

        PetscReal normal[3];
        ierr = TDyFace_GetNormal(faces, face_id, dim, &normal[0]); CHKERRQ(ierr);

        for (PetscInt jj=0;jj<subcell_num_faces;jj++) {
          PetscReal nu[dim];

          ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

          ierr = TDyComputeEntryOfGMatrix(area, normal, K, nu, subcells->T[subcell_id], dim,
                                          &(tdy->subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);

          if (mpfao->Temp_subc_Gmatrix) { // TH
            ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

            ierr = TDyComputeEntryOfGMatrix(area, normal, Kappa,
              nu, subcells->T[subcell_id], dim,
              &(tdy->Temp_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
          } // TH
        } // jj-subcell-faces
      } // ii-isubcell faces
    } // isubcell
  } // icell

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeGMatrix_TPF(TDyMPFAO *mpfao, DM dm,
                                  MaterialProp *matprop) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt dim,icell;
  PetscErrorCode ierr;

  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  TDySubcell *subcells = &mesh->subcells;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (icell=0; icell<mesh->num_cells; icell++) {

    // extract permeability tensor
    PetscInt ii,jj;
    PetscReal K[3][3];

    for (ii=0; ii<dim; ii++) {
      for (jj=0; jj<dim; jj++) {
        K[ii][jj] = matprop->K0[icell*dim*dim + ii*dim + jj];
      }
    }

    // extract thermal conductivity tensor
    PetscReal Kappa[3][3];
    if (mpfao->Temp_subc_Gmatrix) { // TH
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = matprop->Kappa0[icell*dim*dim + ii*dim + jj];
        }
      }
    } // TH

    PetscInt isubcell;

    for (isubcell=0; isubcell<cells->num_subcells[icell]; isubcell++) {

      PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

      PetscInt *subcell_face_ids, subcell_num_faces;
      PetscReal *subcell_face_areas;
      ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);
      ierr = TDyMeshGetSubcellFaceAreas(mesh, subcell_id, &subcell_face_areas, &subcell_num_faces); CHKERRQ(ierr);

      for (PetscInt ii=0;ii<subcell_num_faces;ii++) {

        PetscReal area;
        PetscReal normal[3];

        PetscInt face_id = subcell_face_ids[ii];

        area = subcell_face_areas[ii];

        ierr = TDyFace_GetNormal(faces, face_id, dim, &normal[0]); CHKERRQ(ierr);

        for (PetscInt jj=0;jj<subcell_num_faces;jj++) {
          PetscReal nu[dim];

          if (ii != jj) {

            tdy->subc_Gmatrix[icell][isubcell][ii][jj] = 0.0;

          } else {

            PetscInt neighbor_cell_id;
            PetscReal neighbor_cell_cen[dim],cell_cen[dim], dist;

            PetscInt faceCellOffset = faces->cell_offset[face_id];

            ierr = TDyCell_GetCentroid2(cells, icell, dim, &cell_cen[0]); CHKERRQ(ierr);

            if (faces->cell_ids[faceCellOffset]==icell) {
              neighbor_cell_id = faces->cell_ids[faceCellOffset+1];
            } else {
              neighbor_cell_id = faces->cell_ids[faceCellOffset];
            }

            PetscReal K_neighbor[3][3];
            PetscInt kk,mm;

            if (neighbor_cell_id >=0) {
              ierr = TDyCell_GetCentroid2(cells, neighbor_cell_id, dim, &neighbor_cell_cen[0]); CHKERRQ(ierr);
              ierr = TDyComputeLength(neighbor_cell_cen, cell_cen, dim, &dist); CHKERRQ(ierr);
              dist *= 0.50;

              for (kk=0; kk<dim; kk++) {
                for (mm=0; mm<dim; mm++) {
                  K_neighbor[kk][mm] = matprop->K0[neighbor_cell_id*dim*dim + kk*dim + mm];
                }
              }
            } else {
              ierr = TDyFace_GetCentroid(faces, face_id, dim, &neighbor_cell_cen[0]); CHKERRQ(ierr);
              ierr = TDyComputeLength(neighbor_cell_cen, cell_cen, dim, &dist); CHKERRQ(ierr);
              for (kk=0; kk<dim; kk++) {
                for (mm=0; mm<dim; mm++) {
                  K_neighbor[kk][mm] = matprop->K0[icell*dim*dim + kk*dim + mm];
                }
              }
            }
            PetscReal K_neighbor_value = 0.0, K_value = 0.0, K_aveg;

            PetscReal dot_prod, normal_up2dn[dim];
            ierr = TDyUnitNormalVectorJoiningTwoVertices(neighbor_cell_cen, cell_cen, normal_up2dn); CHKERRQ(ierr);
            ierr = TDyDotProduct(normal,normal_up2dn,&dot_prod); CHKERRQ(ierr);

            for (kk=0;kk<dim;kk++) {
              K_neighbor_value += pow(normal_up2dn[kk],2.0)/K_neighbor[kk][kk];
              K_value          += pow(normal_up2dn[kk],2.0)/K[kk][kk];
            }

            K_neighbor_value = 1.0/K_neighbor_value;
            K_value          = 1.0/K_value;
            K_aveg = 0.5*K_value + 0.5*K_neighbor_value;

            tdy->subc_Gmatrix[icell][isubcell][ii][jj] = area * (dot_prod) * K_aveg/(dist);
          }

          if (mpfao->Temp_subc_Gmatrix) { // TH
            if (ii == jj) {
              ierr = TDySubCell_GetIthNuStarVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

              ierr = TDyComputeEntryOfGMatrix(area, normal, Kappa, nu, subcells->T[subcell_id], dim,
                                              &(tdy->Temp_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
            } else {
              tdy->Temp_subc_Gmatrix[icell][isubcell][ii][jj] = 0.0;
            }
          } // TH

        } // jj-subcell-faces
      } // ii-isubcell faces
    } // isubcell
  } // icell

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeGMatrix(TDyMPFAO* mpfao) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  switch (tdy->options.gmatrix_method) {
    case MPFAO_GMATRIX_DEFAULT:
      ierr = ComputeGMatrix_MPFAO(tdy); CHKERRQ(ierr);
      break;

    case MPFAO_GMATRIX_TPF:
      ierr = ComputeGMatrix_TPF(tdy); CHKERRQ(ierr);
      break;
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_AllocateMemoryForBoundaryValues(TDy tdy) {

  TDyMesh *mesh = tdy->mesh;
  PetscInt nbnd_faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  nbnd_faces = mesh->num_boundary_faces;

  ierr = CharacteristicCurveCreate(nbnd_faces, &tdy->cc_bnd); CHKERRQ(ierr);

  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->P_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->rho_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->vis_BND)); CHKERRQ(ierr);

  PetscInt i;
  PetscReal dden_dP, d2den_dP2, dmu_dP, d2mu_dP2;
  for (i=0;i<nbnd_faces;i++) {
    ierr = ComputeWaterDensity(tdy->Pref, tdy->options.rho_type, &(tdy->rho_BND[i]), &dden_dP, &d2den_dP2); CHKERRQ(ierr);
    ierr = ComputeWaterViscosity(tdy->Pref, tdy->options.mu_type, &(tdy->vis_BND[i]), &dmu_dP, &d2mu_dP2); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_AllocateMemoryForEnergyBoundaryValues(TDy tdy) {

  TDyMesh *mesh = tdy->mesh;
  PetscInt nbnd_faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  nbnd_faces = mesh->num_boundary_faces;

  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->T_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->h_BND)); CHKERRQ(ierr);

  PetscInt i;
  PetscReal dh_dP, dh_dT;
  for (i=0;i<nbnd_faces;i++) {
    ierr = ComputeWaterEnthalpy(tdy->Tref, tdy->Pref,tdy->options.enthalpy_type,
                                &(tdy->h_BND[i]), &dh_dP, &dh_dT); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_AllocateMemoryForSourceSinkValues(TDy tdy) {

  TDyMesh *mesh = tdy->mesh;
  PetscInt ncells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ncells = mesh->num_cells;

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(tdy->source_sink)); CHKERRQ(ierr);

  PetscInt i;
  for (i=0;i<ncells;i++) tdy->source_sink[i] = 0.0;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_AllocateMemoryForEnergySourceSinkValues(TDy tdy) {

  TDyMesh *mesh = tdy->mesh;
  PetscInt ncells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ncells = mesh->num_cells;

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(tdy->energy_source_sink)); CHKERRQ(ierr);

  PetscInt i;
  for (i=0;i<ncells;i++) tdy->energy_source_sink[i] = 0.0;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyMPFAOSetGmatrixMethod(TDyMPFAO* mpfao,TDyMPFAOGmatrixMethod method) {
  PetscFunctionBegin;

  PetscValidPointer(tdy,1);
  mpfao->gmatrix_method = method;

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetMPFAOBoundaryConditionType(TDyMPFAO* mpfao,TDyMPFAOBoundaryConditionType bctype) {
  PetscFunctionBegin;

  PetscValidPointer(tdy,1);
  mpfao->bc_type = bctype;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreate_MPFAO(void **context) {
  PetscFunctionBegin;

  // Allocate a new context for the MPFA-O method.
  TDyMPFAO* mpfao;
  ierr = PetscMalloc(sizeof(TDyMPFAO), &mpfao);
  *context = mpfao;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy_MPFAO(void *context) {
  PetscFunctionBegin;
  TDyMPFAO* mpfao = context;

  if (mpfao->vel   ) { ierr = PetscFree(mpfao->vel); CHKERRQ(ierr); }

  ierr = PetscFree(wy->V); CHKERRQ(ierr);
  ierr = PetscFree(wy->X); CHKERRQ(ierr);
  ierr = PetscFree(wy->N); CHKERRQ(ierr);

  // if (tdy->subc_Gmatrix) { ierr = TDyDeallocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells,
  //                                   nsubcells, nrow, ncol); CHKERRQ(ierr); }
  // if (tdy->Trans       ) { ierr = TDyDeallocate_RealArray_3D(&tdy->Trans,
  //                                   tdy->mesh->num_vertices, 12, 12); CHKERRQ(ierr); }
  // if (tdy->Trans_mat   ) { ierr = MatDestroy(&tdy->Trans_mat  ); CHKERRQ(ierr); }
  if (tdy->P_vec       ) { ierr = VecDestroy(&tdy->P_vec      ); CHKERRQ(ierr); }
  if (tdy->TtimesP_vec ) { ierr = VecDestroy(&tdy->TtimesP_vec); CHKERRQ(ierr); }
  // if (tdy->Temp_subc_Gmatrix) { ierr = TDyDeallocate_RealArray_4D(&tdy->Temp_subc_Gmatrix,
  //                                        tdy->mesh->num_cells,
  //                                        nsubcells, nrow, ncol); CHKERRQ(ierr); }
  // if (tdy->Temp_Trans       ) { ierr = TDyDeallocate_RealArray_3D(&tdy->Temp_Trans,
  //                                        tdy->mesh->num_vertices, 12, 12); CHKERRQ(ierr); }
  if (tdy->Temp_Trans_mat   ) { ierr = MatDestroy(&tdy->Temp_Trans_mat  ); CHKERRQ(ierr); }
  if (tdy->Temp_P_vec       ) { ierr = VecDestroy(&tdy->Temp_P_vec      ); CHKERRQ(ierr); }
  if (tdy->Temp_TtimesP_vec ) { ierr = VecDestroy(&tdy->Temp_TtimesP_vec); CHKERRQ(ierr); }

  // TODO: Need to destroy the mesh.

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetFromOptions_MPFAO(void *context) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  TDyMPFAO* mpfao = context;

  // Set MPFA-O options.
  ierr = PetscOptionsEnum("-tdy_mpfao_gmatrix_method","MPFA-O gmatrix method",
    "TDySetMPFAOGmatrixMethod",TDyMPFAOGmatrixMethods,
    (PetscEnum)mpfao->gmatrix_method,(PetscEnum *)&mpfao->gmatrix_method,NULL);
    CHKERRQ(ierr);
  TDyMPFAOBoundaryConditionType bctype = MPFAO_DIRICHLET_BC;
  ierr = PetscOptionsEnum("-tdy_mpfao_boundary_condition_type","MPFA-O boundary condition type","TDySetMPFAOBoundaryConditionType",TDyMPFAOBoundaryConditionTypes,(PetscEnum)bctype,(PetscEnum *)&bctype, &flag); CHKERRQ(ierr);
  if (flag && (bctype != mpfao->bc_type)) {
    ierr = TDySetMPFAOBoundaryConditionType(mpfao, bctype); CHKERRQ(ierr);
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

//-----------------
// Setup functions
//-----------------

// There's a lot of duplicated code here for the moment. We'll factor out the
// repeated stuff when we move the data out of the TDy struct and into
// discretization-specific types.

// Setup function for Richards + MPFA_O
PetscErrorCode TDySetup_Richards_MPFAO(void *context,
                                       DM dm,
                                       MaterialProp *matprop,
                                       TDyConditions* conditions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  TDyMPFAO *mpfao = context;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  if (dim == 2) {
    SETERRQ(comm,PETSC_ERR_USER,"MPFA-O method supports only 3D calculations.");
  }

  // Allocate mesh data.
  mpfao->mesh = (TDyMesh *) malloc(sizeof(TDyMesh));
  ierr = TDyAllocateMemoryForMesh(mpfao, dm); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_3D(&mpfao->Trans, tdy->mesh->num_vertices, tdy->nfv, tdy->nfv + tdy->ncv); CHKERRQ(ierr);
  ierr = PetscMalloc(mpfao->mesh->num_faces*sizeof(PetscReal),
                     &(tdy->vel )); CHKERRQ(ierr);
  ierr = TDyInitialize_RealArray_1D(tdy->vel, tdy->mesh->num_faces, 0.0); CHKERRQ(ierr);
  ierr = PetscMalloc(tdy->mesh->num_faces*sizeof(PetscInt),
                     &(tdy->vel_count)); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_1D(tdy->vel_count, tdy->mesh->num_faces, 0); CHKERRQ(ierr);
  PetscInt nsubcells = 8;
  PetscInt nrow = 3;
  PetscInt ncol = 3;
  ierr = TDyAllocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells,
                                  nsubcells, nrow, ncol); CHKERRQ(ierr);

  // Set up the section, 1 dof per cell
  PetscSection sec;
  PetscInt p, pStart, pEnd;
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, 1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);

  // Build the mesh.
  ierr = TDyBuildMesh(mpfao, dm); CHKERRQ(ierr);
  {
    PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
    PetscInt nrow, ncol, nz;

    nFaces = mpfao->mesh->num_faces;
    nLocalCells = TDyMeshGetNumberOfLocalCells(mpfao->mesh);
    nNonLocalFaces = TDyMeshGetNumberOfNonLocalFacess(mpfao->mesh);
    nNonInternalFaces = TDyMeshGetNumberOfNonInternalFacess(mpfao->mesh);

    nrow = 4*nFaces;
    ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
    nz   = mpfao->nfv;
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Trans_mat); CHKERRQ(ierr);
    ierr = MatSetOption(mpfao->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->P_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesP_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->GravDisVec);
    ierr = VecZeroEntries(mpfao->GravDisVec);
  }

  // Set up data structures for the discretization.
  ierr = TDyComputeGMatrix(mpfao); CHKERRQ(ierr);
  ierr = TDyComputeTransmissibilityMatrix(mpfao); CHKERRQ(ierr);
  ierr = TDyComputeGravityDiscretization(mpfao); CHKERRQ(ierr);

  ierr = TDyMPFAO_AllocateMemoryForBoundaryValues(mpfao); CHKERRQ(ierr);
  ierr = TDyMPFAO_AllocateMemoryForSourceSinkValues(mpfao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup function for Richards + MPFA_O_DAE
PetscErrorCode TDy_Setup_Richards_MPFAO_DAE(void *context, DM dm,
                                            MaterialProp *matprop,
                                            TDyConditions* conditions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);

  TDyMPFAO *mpfao = context;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  if (dim == 2) {
    SETERRQ(comm,PETSC_ERR_USER,"MPFA-O method supports only 3D calculations.");
  }

  // Allocate mesh data.
  mpfao->mesh = (TDyMesh *) malloc(sizeof(TDyMesh));
  ierr = TDyAllocateMemoryForMesh(tdy); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_3D(&tdy->Trans, tdy->mesh->num_vertices, tdy->nfv, tdy->nfv + tdy->ncv); CHKERRQ(ierr);
  ierr = PetscMalloc(tdy->mesh->num_faces*sizeof(PetscReal),
                     &(tdy->vel )); CHKERRQ(ierr);
  ierr = TDyInitialize_RealArray_1D(tdy->vel, tdy->mesh->num_faces, 0.0); CHKERRQ(ierr);
  ierr = PetscMalloc(tdy->mesh->num_faces*sizeof(PetscInt),
                     &(tdy->vel_count)); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_1D(tdy->vel_count, tdy->mesh->num_faces, 0); CHKERRQ(ierr);
  PetscInt nsubcells = 8;
  PetscInt nrow = 3;
  PetscInt ncol = 3;
  ierr = TDyAllocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells,
                                  nsubcells, nrow, ncol); CHKERRQ(ierr);

  // Set up the section, 1 dof per cell
  PetscSection sec;
  PetscInt p, pStart, pEnd;
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, 2); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 1, "LiquidMass"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 1, 1); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(sec,p,1,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,2); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);

  // Build the mesh.
  ierr = TDyBuildMesh(tdy); CHKERRQ(ierr);
  {
    PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
    PetscInt nrow, ncol, nz;

    nFaces = tdy->mesh->num_faces;
    nLocalCells = TDyMeshGetNumberOfLocalCells(tdy->mesh);
    nNonLocalFaces = TDyMeshGetNumberOfNonLocalFacess(tdy->mesh);
    nNonInternalFaces = TDyMeshGetNumberOfNonInternalFacess(tdy->mesh);

    nrow = 4*nFaces;
    ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
    nz   = tdy->nfv;
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&tdy->Trans_mat); CHKERRQ(ierr);
    ierr = MatSetOption(tdy->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&tdy->P_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&tdy->TtimesP_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&tdy->GravDisVec);
    ierr = VecZeroEntries(tdy->GravDisVec);
  }

  // Set up data structures for the discretization.
  ierr = TDyComputeGMatrix(tdy); CHKERRQ(ierr);
  ierr = TDyComputeTransmissibilityMatrix(tdy); CHKERRQ(ierr);
  ierr = TDyComputeGravityDiscretization(tdy); CHKERRQ(ierr);

  ierr = TDyMPFAO_AllocateMemoryForBoundaryValues(tdy); CHKERRQ(ierr);
  ierr = TDyMPFAO_AllocateMemoryForSourceSinkValues(tdy); CHKERRQ(ierr);
}

// Setup function for Richards + MPFA_O_TRANSIENTVAR
PetscErrorCode TDySetup_Richards_MPFAO_TRANSIENTVAR(void *context, DM dm,
                                                    MaterialProp *matprop,
                                                    TDyConditions* conditions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // This is essentially the same as the MPFA-O one.
  ierr = TDyRichards_MPFAO_Setup(context, dm);
  PetscFunctionReturn(0);
}

// Setup function for TH + MPFA-O
PetscErrorCode TDySetup_TH_MPFAO(void *context, DM dm,
                                 MaterialProp *matprop,
                                 TDyConditions* conditions) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  TDyMPFAO* mpfao = context;

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  if (dim == 2) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"MPFA-O method supports only 3D calculations.");
  }

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  // Allocate mesh data.
  mpfao->mesh = (TDyMesh *) malloc(sizeof(TDyMesh));
  ierr = TDyAllocateMemoryForMesh(mpfao); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_3D(&mpfao->Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfo->nfv + mpfao->ncv); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_3D(&mpfao->Temp_Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv); CHKERRQ(ierr);
  ierr = PetscMalloc(mpfao->mesh->num_faces*sizeof(PetscReal),
                     &(mpfao->vel)); CHKERRQ(ierr);
  ierr = TDyInitialize_RealArray_1D(mpfao->vel, mpfao->mesh->num_faces, 0.0); CHKERRQ(ierr);
  ierr = PetscMalloc(mpfao->mesh->num_faces*sizeof(PetscInt),
                     &(mpfao->vel_count)); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_1D(mpfao->vel_count, mpfao->mesh->num_faces,
                                       0); CHKERRQ(ierr);

  PetscInt nsubcells = 8;
  PetscInt nrow = 3;
  PetscInt ncol = 3;

  ierr = TDyAllocate_RealArray_4D(&mpfao->subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, nrow, ncol); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_4D(&mpfao->Temp_subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, nrow, ncol); CHKERRQ(ierr);

  // Set up the section, 1 dof per cell
  PetscSection sec;
  PetscInt p, pStart, pEnd;
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, 2); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 1, "LiquidTemperature"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 1, 1); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(sec,p,1,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,2); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);

  // Build the mesh.
  ierr = TDyBuildMesh(mpfao, dm); CHKERRQ(ierr);
  {
    PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
    PetscInt nrow, ncol, nz;

    nFaces = mpfao->mesh->num_faces;
    nLocalCells = TDyMeshGetNumberOfLocalCells(mpfao->mesh);
    nNonLocalFaces = TDyMeshGetNumberOfNonLocalFacess(mpfao->mesh);
    nNonInternalFaces = TDyMeshGetNumberOfNonInternalFacess(mpfao->mesh);

    nrow = 4*nFaces;
    ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
    nz   = mpfao->nfv;
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Trans_mat); CHKERRQ(ierr);
    ierr = MatSetOption(mpfao->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->P_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesP_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->GravDisVec);
    ierr = VecZeroEntries(mpfao->GravDisVec);

    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Temp_Trans_mat); CHKERRQ(ierr);
    ierr = MatSetOption(mpfao->Temp_Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->Temp_P_vec);
    ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->Temp_TtimesP_vec);
  }

  // Compute matrices for our discretization.
  ierr = TDyComputeGMatrix(mpfao); CHKERRQ(ierr);
  ierr = TDyComputeTransmissibilityMatrix(mpfao); CHKERRQ(ierr);
  ierr = TDyComputeGravityDiscretization(mpfao); CHKERRQ(ierr);

  ierr = TDyMPFAO_AllocateMemoryForBoundaryValues(mpfao); CHKERRQ(ierr);
  ierr = TDyMPFAO_AllocateMemoryForSourceSinkValues(mpfao); CHKERRQ(ierr);
  ierr = TDyMPFAO_AllocateMemoryForEnergyBoundaryValues(mpfao); CHKERRQ(ierr);
  ierr = TDyMPFAO_AllocateMemoryForEnergySourceSinkValues(mpfao); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

//-----------------------
// UpdateState functions
//-----------------------
PetscErrorCode TDyUpdateState_Richards_MPFAO(void *context, DM dm,
                                             TDyEOS *eos,
                                             MaterialProp *matprop,
                                             CharacteristicCurve *cc) {
  PetscFunctionBegin;
  TDyMPFAO *mpfao = context;

  PetscReal Se,dSe_dS,dKr_dSe,Kr;
  PetscReal *P, *temp;
  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  PetscInt dim2 = dim*dim;
  PetscInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&P);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&temp);CHKERRQ(ierr);

  for (PetscInt c=0;c<cEnd-cStart;c++) P[c] = U[c];

  for(PetscInt c=cStart; c<cEnd; c++) {
    PetscInt i = c-cStart;

    PetscReal n = cc->gardner_n[c];
    PetscReal alpha = cc->vg_alpha[c];

    switch (cc->SatFuncType[i]) {
    case SAT_FUNC_GARDNER :
      PressureSaturation_Gardner(n,cc->gardner_m[c],alpha,cc->sr[i],mpfao->Pref-P[i],&(cc->S[i]),&(cc->dS_dP[i]),&(cc->d2S_dP2[i]));
      break;
    case SAT_FUNC_VAN_GENUCHTEN :
      PressureSaturation_VanGenuchten(cc->vg_m[c],alpha,cc->sr[i],mpfao->Pref-P[i],&(cc->S[i]),&cc->dS_dP[i],&(cc->d2S_dP2[i]));
      break;
    default:
      SETERRQ(comm,PETSC_ERR_SUP,"Unknown saturation function");
      break;
    }

    Se = (cc->S[i] - cc->sr[i])/(1.0 - cc->sr[i]);
    dSe_dS = 1.0/(1.0 - cc->sr[i]);

    switch (cc->RelPermFuncType[i]) {
    case REL_PERM_FUNC_IRMAY :
      RelativePermeability_Irmay(cc->irmay_m[c],Se,&Kr,NULL);
      break;
    case REL_PERM_FUNC_MUALEM :
      RelativePermeability_Mualem(cc->mualem_m[c],cc->mualem_poly_low[c],cc->mualem_poly_coeffs[c],Se,&Kr,&dKr_dSe);
      break;
    default:
      SETERRQ(comm,PETSC_ERR_SUP,"Unknown relative permeability function");
      break;
    }
    cc->Kr[i] = Kr;
    cc->dKr_dS[i] = dKr_dSe * dSe_dS;

    for(PetscInt j=0; j<dim2; j++) {
      matprop->K[i*dim2+j] = matprop->K0[i*dim2+j] * Kr;
    }

    ierr = TDyEOSComputeWaterDensity(eos, P[i], &(mpfao->rho[i]),
                                     &(mpfao->drho_dP[i]),
                                     &(mpfao->d2rho_dP2[i])); CHKERRQ(ierr);
    ierr = TDyEOSComputeWaterViscosity(eos, P[i], &(mpfao->vis[i]),
                                       &(mpfao->dvis_dP[i]),
                                       &(mpfao->d2vis_dP2[i])); CHKERRQ(ierr);
  }

  PetscReal *p_vec_ptr, gz;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;

  ierr = VecGetArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=cStart; c<cEnd; c++) {
    PetscInt i = c-cStart;
    ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[i].X,dim,&gz);
    p_vec_ptr[i] = P[i];
  }
  ierr = VecRestoreArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateState_TH_MPFAO(void *context, DM dm,
                                       TDyEOS *eos,
                                       MaterialProp *matprop,
                                       CharacteristicCurve *cc) {
  PetscFunctionBegin;
  TDyMPFAO *mpfao = context;

  PetscReal Se,dSe_dS,dKr_dSe,Kr;
  PetscReal *P, *temp;
  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  PetscInt dim2 = dim*dim;
  PetscInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&P);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&temp);CHKERRQ(ierr);
  for (PetscInt c=0;c<cEnd-cStart;c++) {
    P[c] = U[c*2];
    temp[c] = U[c*2+1];
  }

  for(PetscInt c=cStart; c<cEnd; c++) {
    PetscInt i = c-cStart;

    PetscReal n = cc->gardner_n[c];
    PetscReal alpha = cc->vg_alpha[c];

    switch (cc->SatFuncType[i]) {
    case SAT_FUNC_GARDNER :
      PressureSaturation_Gardner(n,cc->gardner_m[c],alpha,cc->sr[i],mpfao->Pref-P[i],&(cc->S[i]),&(cc->dS_dP[i]),&(cc->d2S_dP2[i]));
      break;
    case SAT_FUNC_VAN_GENUCHTEN :
      PressureSaturation_VanGenuchten(cc->vg_m[c],alpha,cc->sr[i],mpfao->Pref-P[i],&(cc->S[i]),&cc->dS_dP[i],&(cc->d2S_dP2[i]));
      break;
    default:
      SETERRQ(comm,PETSC_ERR_SUP,"Unknown saturation function");
      break;
    }

    Se = (cc->S[i] - cc->sr[i])/(1.0 - cc->sr[i]);
    dSe_dS = 1.0/(1.0 - cc->sr[i]);

    switch (cc->RelPermFuncType[i]) {
    case REL_PERM_FUNC_IRMAY :
      RelativePermeability_Irmay(cc->irmay_m[c],Se,&Kr,NULL);
      break;
    case REL_PERM_FUNC_MUALEM :
      RelativePermeability_Mualem(cc->mualem_m[c],cc->mualem_poly_low[c],cc->mualem_poly_coeffs[c],Se,&Kr,&dKr_dSe);
      break;
    default:
      SETERRQ(comm,PETSC_ERR_SUP,"Unknown relative permeability function");
      break;
    }
    cc->Kr[i] = Kr;
    cc->dKr_dS[i] = dKr_dSe * dSe_dS;

    for(PetscInt j=0; j<dim2; j++) {
      matprop->K[i*dim2+j] = matprop->K0[i*dim2+j] * Kr;
    }

    ierr = TDyEOSComputeWaterDensity(eos, P[i], &(mpfao->rho[i]),
                                     &(mpfao->drho_dP[i]),
                                     &(mpfao->d2rho_dP2[i])); CHKERRQ(ierr);
    ierr = TDyEOSComputeWaterViscosity(eos, P[i], &(mpfao->vis[i]),
                                       &(mpfao->dvis_dP[i]),
                                       &(mpfao->d2vis_dP2[i])); CHKERRQ(ierr);
    for(PetscInt j=0; j<dim2; j++)
      matprop->Kappa[i*dim2+j] = matprop->Kappa0[i*dim2+j]; // update this based on Kersten number, etc.
    ierr = TDyEOSComputeWaterEnthalpy(eos, temp[i], P[i], &(mpfao->h[i]),
                                      &(mpfao->dh_dP[i]),
                                      &(mpfao->dh_dT[i])); CHKERRQ(ierr);
    mpfao->u[i] = mpfao->h[i] - P[i]/mpfao->rho[i];
  }

  PetscReal *p_vec_ptr, gz;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;

  ierr = VecGetArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=cStart; c<cEnd; c++) {
    PetscInt i = c-cStart;
    ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[i].X,dim,&gz);
    p_vec_ptr[i] = P[i];
  }
  ierr = VecRestoreArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscReal *t_vec_ptr;
  ierr = VecGetArray(mpfao->Temp_P_vec, &t_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=cStart; c<cEnd; c++) {
    PetscInt i = c-cStart;
    t_vec_ptr[i] = temp[i];
  }
  ierr = VecRestoreArray(mpfao->Temp_P_vec, &t_vec_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem(TDy tdy,Mat K,Vec F) {

  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  ierr = MatZeroEntries(K);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = TDyMPFAOComputeSystem_InternalVertices(tdy,K,F); CHKERRQ(ierr);
  ierr = TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(tdy,K,F); CHKERRQ(ierr);
  ierr = TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(tdy,K,F); CHKERRQ(ierr);

  PetscInt dim = 3;

  if (tdy->ops->computeforcing) {
    for (PetscInt icell=0; icell<tdy->mesh->num_cells; icell++) {
      if (cells->is_local[icell]) {
        PetscReal f;
        ierr = (*tdy->ops->computeforcing)(tdy, &(tdy->X[icell*dim]), &f, tdy->forcingctx);CHKERRQ(ierr);
        PetscReal value = f * cells->volume[icell];
        PetscInt row = cells->global_id[icell];
        ierr = VecSetValue(F, row, value, ADD_VALUES); CHKERRQ(ierr);
      }
    }
  }

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscReal TDyMPFAOPressureNorm(TDy tdy, Vec U) {

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscScalar    *u;
  Vec            localU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  cells = &mesh->cells;

  if (! tdy->ops->compute_boundary_pressure) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the boundary pressure function with TDySetBoundaryPressureFn");
  }

  PetscInt dim = 3;

  ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U,localU); CHKERRQ(ierr);
  ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

  PetscReal norm_sum = 0.0;
  PetscReal norm = 0.0;

  for (PetscInt icell=0; icell<mesh->num_cells; icell++) {

    if (!cells->is_local[icell]) continue;

    PetscReal pressure;
    ierr = (*tdy->ops->compute_boundary_pressure)(tdy, &(tdy->X[icell*dim]), &pressure, tdy->boundary_pressure_ctx);CHKERRQ(ierr);
    norm += (PetscSqr(pressure - u[icell])) * cells->volume[icell];
  }

  ierr = VecRestoreArray(localU, &u); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);

  norm_sum = PetscSqrtReal(norm_sum);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(norm_sum);
}


/* -------------------------------------------------------------------------- */
PetscReal TDyMPFAOVelocityNorm(TDy tdy) {

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyFace       *faces = &mesh->faces;
  TDyCell       *cells = &mesh->cells;
  PetscInt       dim;
  PetscInt       icell, iface, face_id;
  PetscInt       fStart, fEnd;
  PetscReal      norm, norm_sum, vel_normal;
  PetscReal      vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  cells = &mesh->cells;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  norm_sum = 0.0;
  norm     = 0.0;

  for (icell=0; icell<mesh->num_cells; icell++) {

    if (!cells->is_local[icell]) continue;

    for (iface=0; iface<cells->num_faces[icell]; iface++) {
      PetscInt faceStart = cells->face_offset[icell];
      face_id = cells->face_ids[faceStart + iface];
      //face    = &(faces[face_id]);

      ierr = (*tdy->ops->compute_boundary_velocity)(tdy, &(tdy->X[(face_id + fStart)*dim]), vel, tdy->boundary_velocity_ctx);CHKERRQ(ierr);
      vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim);
      if (tdy->vel_count[face_id] != faces->num_vertices[face_id]) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"tdy->vel_count != faces->num_vertices[face_id]");
      }

      norm += PetscSqr((vel_normal - tdy->vel[face_id]))*cells->volume[icell];
    }
  }

  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(norm_sum);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyComputeEntryOfGMatrix(PetscReal area, PetscReal n[3],
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

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyVertex *vertices = &mesh->vertices;

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

    PetscInt *face_unknown_idx, *is_face_up, *face_flux_idx, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaceUnknownIdxs(mesh, subcell_id, &face_unknown_idx, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellIsFaceUp(mesh, subcell_id, &is_face_up, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceFluxIdxs(mesh, subcell_id, &face_flux_idx, &subcell_num_faces); CHKERRQ(ierr);

    if (varID == VAR_PRESSURE) {
      ierr = ExtractSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);
    } else if (varID == VAR_TEMPERATURE){
      ierr = ExtractTempSubGmatrix(tdy, icell, isubcell, ndim, Gmatrix);
    }

    PetscInt idx_interface_p0, idx_interface_p1, idx_interface_p2;

    idx_interface_p0 = face_unknown_idx[0];
    idx_interface_p1 = face_unknown_idx[1];
    idx_interface_p2 = face_unknown_idx[2];

    PetscInt idx_flux, iface;
    for (iface=0; iface<subcell_num_faces; iface++) {

      PetscBool upwind_entries = (is_face_up[iface] == 1);

      if (upwind_entries) {
        idx_flux = face_flux_idx[iface];

        Cup[idx_flux][idx_interface_p0] = -Gmatrix[iface][0];
        Cup[idx_flux][idx_interface_p1] = -Gmatrix[iface][1];
        Cup[idx_flux][idx_interface_p2] = -Gmatrix[iface][2];

        Fup[idx_flux][i] = -Gmatrix[iface][0] - Gmatrix[iface][1] - Gmatrix[iface][2];

      } else {
        idx_flux = face_flux_idx[iface];

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

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyVertex *vertices = &mesh->vertices;
  TDyFace *faces = &mesh->faces;

  PetscInt npcen = vertices->num_internal_cells[ivertex];
  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  PetscErrorCode ierr;

  *nflux_bc_up = 0;
  *nflux_bc_dn = 0;

  for (PetscInt i=0; i<npcen; i++) {
    PetscInt icell    = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

    PetscInt *face_ids, *is_face_up, num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellIsFaceUp(mesh, subcell_id, &is_face_up, &num_faces); CHKERRQ(ierr);

    for (PetscInt iface=0; iface<num_faces; iface++) {

      PetscInt faceID = face_ids[iface];
      if (faces->is_internal[faceID]) continue;

      if (is_face_up[iface]==1) {
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
    PetscInt ivertex, TDyCell *cells, PetscInt varID) {

  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  TDyFace *faces = &mesh->faces;
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

  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  PetscInt *face_ids, num_faces;
  PetscInt *subface_ids, num_subfaces;
  ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
  ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

  vertex_id = ivertex;

  ierr = DMGetDimension(tdy->dm, &ndim); CHKERRQ(ierr);

  PetscInt npcen        = vertices->num_internal_cells[ivertex];
  PetscInt npitf_bc_all = vertices->num_boundary_faces[ivertex];

  PetscInt nflux_all_bc_up, nflux_all_bc_dn;
  PetscInt nflux_dir_bc_up, nflux_dir_bc_dn;
  PetscInt nflux_neu_bc_up, nflux_neu_bc_dn;
  ierr = DetermineNumberOfUpAndDownBoundaryFaces(tdy, ivertex, &nflux_all_bc_up, &nflux_all_bc_dn);

  PetscInt npitf_dir_bc_all, npitf_neu_bc_all;

  if (tdy->options.mpfao_bc_type == MPFAO_DIRICHLET_BC ||
      tdy->options.mpfao_bc_type == MPFAO_SEEPAGE_BC) {
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
  ierr = ComputeAinvB(nflux_in + nflux_neu_bc_all, AINBCxINBC_1d, npcen+npitf_dir_bc_all, BINBCxCDBC_1d, AinvB_1d); CHKERRQ(ierr);

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

      PetscInt *subcell_face_ids, num_faces;
      ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &num_faces); CHKERRQ(ierr);

      for (PetscInt iface=0;iface<num_faces;iface++) {

        PetscInt face_id = subcell_face_ids[iface];

        PetscInt *face_cell_ids, num_cell_ids;
        ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);

        if (faces->is_internal[face_id] == 0) {

          // Extract pressure value at the boundary
          if (face_cell_ids[0] >= 0) idxBnd[numBnd] = -face_cell_ids[1] - 1;
          else                       idxBnd[numBnd] = -face_cell_ids[0] - 1;

          numBnd++;
        }
      }
    }
  }

  PetscInt num_subfaces_per_face = 4;
  for (i=0; i<nflux_in+nflux_dir_bc_up+nflux_dir_bc_dn; i++) {
    face_id = face_ids[i];
    subface_id = subface_ids[i];
    row = face_id*num_subfaces_per_face + subface_id;

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
    PetscInt ivertex, TDyCell *cells, PetscInt varID) {
  DM             dm = tdy->dm;
  TDyMesh *mesh = tdy->mesh;
  TDyVertex *vertices = &mesh->vertices;
  TDySubcell    *subcells = &mesh->subcells;
  TDyFace       *faces = &mesh->faces;
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
      (*Trans)[vertices->id[ivertex]][iface][j] = -Gmatrix[iface][j];
    }
    (*Trans)[vertices->id[ivertex]][iface][dim] = 0.0;
    for (j=0; j<dim; j++) (*Trans)[vertices->id[ivertex]][iface][dim] += (Gmatrix[iface][j]);
  }

  PetscInt i, face_id, subface_id;
  PetscInt row, col, ncells;
  PetscInt numBnd, idxBnd[3];

  ncells = vertices->num_internal_cells[ivertex];
  numBnd = 0;
  PetscInt face_ids_of_subcell[ncells*6];
  PetscInt count = 0;

  for (i=0; i<ncells; i++) {
    icell = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

    PetscInt *subcell_face_ids, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);

    PetscInt iface;
    for (iface=0; iface<subcell_num_faces; iface++) {

      PetscInt face_id = subcell_face_ids[iface];
      PetscInt *face_cell_ids, num_cell_ids;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);

      face_ids_of_subcell[count] = face_id;
      count++;

      if (faces->is_internal[face_id] == 0) {

        // Extract pressure value at the boundary
        if (face_cell_ids[0] >= 0) idxBnd[numBnd] = -face_cell_ids[1] - 1;
        else                       idxBnd[numBnd] = -face_cell_ids[0] - 1;

        numBnd++;
      }
    }
  }

  icell    = vertices->internal_cell_ids[vOffsetCell + 0];
  isubcell = vertices->subcell_ids[vOffsetSubcell + 0];
  subcell_id = icell*cells->num_subcells[icell]+isubcell;

  for (i=0; i<subcells->num_faces[subcell_id]; i++) {

    face_id = face_ids_of_subcell[i];

    // Determine the subface ID of 'face_id' that includes 'ivertex'
    PetscInt *vertex_ids, num_vertices;
    ierr = TDyMeshGetFaceVertices(mesh, face_id, &vertex_ids, &num_vertices); CHKERRQ(ierr);
    for (PetscInt ii=0; ii<num_vertices; ii++){
      if (ivertex == vertex_ids[ii]) {
        subface_id = ii;
        break;
      }
    }
    row = face_id * 4 + subface_id;

    for (j=0; j<numBnd; j++) {
      col = idxBnd[j] + tdy->mesh->num_cells;
      ierr = MatSetValues(*Trans_mat,1,&row,1,&col,&(*Trans)[ivertex][i][j],ADD_VALUES); CHKERRQ(ierr);
    }

    icell  = vertices->internal_cell_ids[vOffsetCell + 0];
    col = icell;
    ierr = MatSetValues(*Trans_mat,1,&row,1,&col,&(*Trans)[ivertex][i][numBnd],ADD_VALUES); CHKERRQ(ierr);
  }

  // Need to swap entries in (*Trans)[vertices->id][:][:] such that
  // Step-1. Rows correspond to faces saved in the order of vertices->face_ids[:]
  // Step-2. Columns are such that boundary cells are followed by the internal cell.
  //         The boundary cells are in the order of vertices->cell_ids[1:3]
  //         Note: vertices->cell_ids[0] correspond to internal cell
  // Step-3. Finally move the last column as the first column. So, columns
  //         are in the order of vertices->cell_ids[0:3]

  PetscReal T_1[3][4], T_2[3][4];
  // Swap rows
  PetscInt subcell_boundary_cell_id[dim];
  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

    PetscInt row = -1;

    PetscInt *face_ids, num_faces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);

    PetscInt face_id_wrt_vertex = face_ids[iface];

    PetscInt i = 0;
    PetscInt icell = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];
    PetscInt subcell_id = icell*cells->num_subcells[icell] + isubcell;

    PetscInt *subcell_face_ids, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaces(mesh, subcell_id, &subcell_face_ids, &subcell_num_faces); CHKERRQ(ierr);

    for (PetscInt mm=0; mm<dim; mm++){
      PetscInt face_id_wrt_subcell = subcell_face_ids[mm];
      PetscInt offset = faces->cell_offset[face_id_wrt_subcell];

      if (faces->cell_ids[offset] < 0) {
        subcell_boundary_cell_id[mm] = faces->cell_ids[offset];
      } else {
        subcell_boundary_cell_id[mm] = faces->cell_ids[offset + 1];
      }
      if (face_id_wrt_vertex == face_id_wrt_subcell){
        row = mm;
        break;
      }
    }

    for (j=0; j<dim; j++) {
      T_1[iface][j] = -Gmatrix[row][j];
    }
    T_1[iface][dim] = 0.0;
    for (j=0; j<dim; j++) T_1[iface][dim] += (Gmatrix[row][j]);
  }

  // Swap columns
  PetscReal Told[subcells->num_faces[subcell_id]][dim];
  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {
    for (j = 0; j < dim; j++) {
      Told[iface][j] = T_1[iface][j];
    }
  }

  PetscInt vertex_boundary_cell_id [subcells->num_faces[subcell_id]];
  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++) {

    PetscInt *face_ids, num_faces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);

    PetscInt face_id_wrt_vertex = face_ids[iface];
    PetscInt *face_cell_ids, num_cell_ids;
    ierr = TDyMeshGetFaceCells(mesh, face_id_wrt_vertex, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);

    if (face_cell_ids[0] < 0){
      vertex_boundary_cell_id[iface] = face_cell_ids[0];
    } else {
      vertex_boundary_cell_id[iface] = face_cell_ids[1];
    }
  }

  for (j = 0; j < dim; j++){
    PetscInt col = -1;
    for (PetscInt mm=0; mm<dim; mm++){
      if (vertex_boundary_cell_id[j] == subcell_boundary_cell_id[mm]) {
        col = mm;
        break;
      }
    }

    for (PetscInt row=0; row<dim; row++){
      T_2[row][col] = Told[row][j];
    }

  }
  for (PetscInt row=0; row<dim; row++) T_2[row][dim] = T_1[row][dim];

  PetscReal T_old[subcells->num_faces[subcell_id]][dim+1];
  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++){
    for (j = 0; j<dim+1; j++){
      T_old[iface][j] = T_2[iface][j];
    }
  }

  for (iface=0; iface<subcells->num_faces[subcell_id]; iface++){
    (*Trans)[vertices->id[ivertex]][iface][0] = T_old[iface][dim];
    for (j = 0; j<dim; j++){
      (*Trans)[vertices->id[ivertex]][iface][j+1] = T_old[iface][j];
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyUpdateTransmissibilityMatrix(TDy tdy) {

  TDyMesh       *mesh = tdy->mesh;
  TDyRegion     *region = &mesh->region_connected;
  PetscInt       iface, isubface;
  PetscInt       num_subfaces = 4;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  // If a face is shared by two cells that belong to different
  // regions, zero the rows in the transmissiblity matrix

  for (iface=0; iface<tdy->mesh->num_faces; iface++) {

    PetscInt *face_cell_ids, num_cell_ids;
    ierr = TDyMeshGetFaceCells(mesh, iface, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
    PetscInt cell_id_up = face_cell_ids[0];
    PetscInt cell_id_dn = face_cell_ids[1];

    if (cell_id_up>=0 && cell_id_dn>=0) {
      if (!TDyRegionAreCellsInTheSameRegion(region, cell_id_up, cell_id_dn)) {
        for (isubface=0; isubface<4; isubface++) {
          PetscInt row[1];
          row[0] = iface*num_subfaces + isubface;
          ierr = MatZeroRows(tdy->Trans_mat,1,row,0.0,0,0); CHKERRQ(ierr);
          if (tdy->options.mode == TH) {
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
PetscErrorCode TDyComputeTransmissibilityMatrix(TDy tdy) {

  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    if (vertices->num_boundary_faces[ivertex] == 0 || vertices->num_internal_cells[ivertex] > 1) {
      ierr = ComputeTransmissibilityMatrix_ForNonCornerVertex(tdy, ivertex, cells, 0); CHKERRQ(ierr);
      if (tdy->options.mode == TH) {
        ierr = ComputeTransmissibilityMatrix_ForNonCornerVertex(tdy, ivertex, cells, 1); CHKERRQ(ierr);
      }
    } else {
      // It is assumed that neumann boundary condition is a zero-flux boundary condition.
      // Thus, compute transmissiblity entries only for dirichlet boundary condition.
      if (tdy->options.mpfao_bc_type == MPFAO_DIRICHLET_BC ||
          tdy->options.mpfao_bc_type == MPFAO_SEEPAGE_BC) {
        ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(tdy, ivertex, cells, 0); CHKERRQ(ierr);
        if (tdy->options.mode == TH) {
          ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(tdy, ivertex, cells, 1); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = MatAssemblyBegin(tdy->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tdy->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (tdy->options.mode == TH) {
    ierr = MatAssemblyBegin(tdy->Temp_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tdy->Temp_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  TDyRegion *region = &mesh->region_connected;
  if (region->num_cells > 0){
    if (mpfao->gmatrix_method == MPFAO_GMATRIX_TPF ) {
      ierr = TDyUpdateTransmissibilityMatrix(tdy); CHKERRQ(ierr);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"WARNING -- Connected region option is only supported with MPFA-O TPF\n");
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
/// Computes unit vector joining upwind and downwind cells that share a face.
/// The unit vector points from upwind to downwind cell.
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] up2dn_uvec Unit vector from upwind to downwind cell
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ComputeUpDownUnitVector(TDy tdy, PetscInt face_id, PetscReal up2dn_uvec[3]) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

    PetscInt *face_cell_ids, num_cell_ids;
    ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
    PetscInt cell_id_up = face_cell_ids[0];
    PetscInt cell_id_dn = face_cell_ids[1];

  if (cell_id_up < 0 && cell_id_dn < 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Both cell IDs sharing a face are not valid");
  }

  PetscInt dim = 3;
  PetscReal coord_up[dim], coord_dn[dim], coord_face[dim];

  ierr = TDyFace_GetCentroid(faces, face_id, dim, &coord_face[0]);
  CHKERRQ(ierr);

  if (cell_id_up >= 0) {
    ierr = TDyCell_GetCentroid2(cells, cell_id_up, dim, &coord_up[0]); CHKERRQ(ierr);
  } else {
    for (PetscInt d = 0; d < 3; d++) coord_up[d] = coord_face[d];
  }

  if (cell_id_dn >= 0) {
    ierr = TDyCell_GetCentroid2(cells, cell_id_dn, dim, &coord_dn[0]); CHKERRQ(ierr);
  } else {
    for (PetscInt d = 0; d < 3; d++) coord_dn[d] = coord_face[d];
  }

  ierr = TDyUnitNormalVectorJoiningTwoVertices(coord_up, coord_dn, up2dn_uvec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Computes upwind and downwind distance of cells sharing a face. If the face is a
/// boundary face, one of the distance is zero
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] dist_up Distance between the upwind cell centroid and face centroid
/// @param [out] dist_dn Distance between the downwind cell centroid and face centroid
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ComputeUpAndDownDist(TDy tdy, PetscInt face_id, PetscReal *dist_up, PetscReal *dist_dn) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  TDyCell *cells = &mesh->cells;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscInt *face_cell_ids, num_cell_ids;
  ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
  PetscInt cell_id_up = face_cell_ids[0];
  PetscInt cell_id_dn = face_cell_ids[1];

  if (cell_id_up < 0 && cell_id_dn < 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Both cell IDs sharing a face are not valid");
  }

  PetscInt dim = 3;

  PetscReal coord_face[dim];
  ierr = TDyFace_GetCentroid(faces, face_id, dim, &coord_face[0]); CHKERRQ(ierr);

  if (cell_id_up >= 0) {
    PetscReal coord_up[dim];
    ierr = TDyCell_GetCentroid2(cells, cell_id_up, dim, &coord_up[0]); CHKERRQ(ierr);
    ierr = TDyComputeLength(coord_up, coord_face, dim, dist_up); CHKERRQ(ierr);
  } else {
    *dist_up = 0.0;
  }

  if (cell_id_dn >= 0) {
    PetscReal coord_dn[dim];
    ierr = TDyCell_GetCentroid2(cells, cell_id_dn, dim, &coord_dn[0]); CHKERRQ(ierr);
    ierr = TDyComputeLength(coord_dn, coord_face, dim, dist_dn); CHKERRQ(ierr);
  } else {
    *dist_dn = 0.0;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Computes face permeability tensor as a harmonically distance-weighted
//  permeability of upwind and downwind permeability tensors
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] Kup components of the upwind permeability tensor in a row-major order
/// @param [out] Kdn components of the downwind permeability tensor in a row-major order
PetscErrorCode ExtractUpAndDownPermeabilityTensors(TDy tdy, PetscInt face_id, PetscInt dim, PetscReal Kup[dim*dim], PetscReal Kdn[dim*dim]) {

  PetscFunctionBegin;

  TDyMesh *mesh = tdy->mesh;
  MaterialProp *matprop = tdy->matprop;
  PetscErrorCode ierr;

  PetscInt *face_cell_ids, num_cell_ids;
  ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
  PetscInt cell_id_up = face_cell_ids[0];
  PetscInt cell_id_dn = face_cell_ids[1];

  if (cell_id_up < 0 && cell_id_dn < 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Both cell IDs sharing a face are not valid");
  }

  for (PetscInt kk = 0; kk < dim; kk++)
  {
    for (PetscInt mm = 0; mm < dim; mm++)
    {
      if (cell_id_up >= 0) Kup[mm * dim + kk] = matprop->K0[cell_id_up * dim * dim + kk * dim + mm];
      else                 Kup[mm * dim + kk] = matprop->K0[cell_id_dn * dim * dim + kk * dim + mm];

      if (cell_id_dn >= 0) Kdn[mm * dim + kk] = matprop->K0[cell_id_dn * dim * dim + kk * dim + mm];
      else                 Kdn[mm * dim + kk] = matprop->K0[cell_id_up * dim * dim + kk * dim + mm];
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Computes face permeability tensor as a harmonically distance-weighted
//  permeability of upwind and downwind permeability tensors
///
/// K = (wt_1 * K_u^{-1} + (1-wt_1) * K_d^{-1})^{-1}
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] Kface components of face permeability tensor in row-major order
PetscErrorCode ComputeFacePermeabilityTensor(TDy tdy, PetscInt face_id, PetscReal Kface[9]){

  PetscFunctionBegin;

  PetscErrorCode ierr;
  PetscReal dist_up, dist_dn;

  ierr = ComputeUpAndDownDist(tdy, face_id, &dist_up, &dist_dn); CHKERRQ(ierr);

  PetscInt dim = 3;

  PetscReal Kup[dim*dim], Kdn[dim*dim];
  PetscReal KupInv[dim*dim], KdnInv[dim*dim];

  ierr = ExtractUpAndDownPermeabilityTensors(tdy, face_id, dim, Kup, Kdn); CHKERRQ(ierr);

  ierr = ComputeInverseOf3by3Matrix(Kup, KupInv); CHKERRQ(ierr);
  ierr = ComputeInverseOf3by3Matrix(Kdn, KdnInv); CHKERRQ(ierr);

  PetscReal KfaceInv[dim*dim];
  PetscReal wt_up = dist_up / (dist_up + dist_dn);

  PetscInt idx;

  // Compute (wt_1 * K_u^{-1} + (1-wt_1) * K_d^{-1})
  idx = 0;
  for (PetscInt kk = 0; kk < dim; kk++) {
    for (PetscInt mm = 0; mm < dim; mm++) {
      KfaceInv[idx] = wt_up * KupInv[idx] + (1.0 - wt_up) * KdnInv[idx];
      idx++;
    }
  }

  ierr = ComputeInverseOf3by3Matrix(KfaceInv, Kface); CHKERRQ(ierr);

  idx = 0;
  for (PetscInt kk = 0; kk < dim; kk++) {
    for (PetscInt mm = 0; mm < dim; mm++) {
      if (Kface[idx] < 0.0 || fabs(Kface[idx]) < PETSC_MACHINE_EPSILON) Kface[idx] = 0.0;
      idx++;
    }
  }

 PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Computes the two-point flux discretization of the gravity term in Richards equation
///
/// Flux associated with the gravity term is given as
///
/// flux_m = rho^2 * ukvr * A_face * dot (n_face, K_face x u_up2dn) * dot(g, u_up2dn)
///        = rho^2 * ukvr * GravDis
///
/// where
///   rho     = distance weighted density
///   ukvr    = upwind relative permeability divided by viscosity
///   A_face  = face area
///   n_face  = normal to face area
///   K_face  = permeability tensor at the face
///   u_up2dn = unit vector from upwind to downwind cell connected through a common face
///   g       = gravity vector
///   GravDis = gravity discretization term that is not dependent on the unknown variable(s)
///             such as pressure, temperautre. Thus, this term is precomputed.
///
/// Starnoni, M., Berre, I., Keilegavlen, E., & Nordbotten, J. M. (2019).
/// Consistent mpfa discretization for flow in the presence of gravity. Water
/// Resources Research, 55, 10105– 10118. https://doi.org/10.1029/2019WR025384
///
/// @param [in] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyComputeGravityDiscretization(TDy tdy) {

  PetscFunctionBegin;

  TDY_START_FUNCTION_TIMER()

  DM dm = tdy->dm;
  TDyMesh *mesh = tdy->mesh;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  TDySubcell *subcells = &mesh->subcells;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscScalar *gradDisPtr;
  ierr = VecGetArray(tdy->GravDisVec, &gradDisPtr); CHKERRQ(ierr);

  // Loop over all vertices
  for (PetscInt ivertex=0; ivertex<mesh->num_vertices; ivertex++){

    // Skip the vertex that is not locally owned
    if (!vertices->is_local[ivertex]) continue;

    PetscInt *face_ids, num_faces;
    PetscInt *subface_ids, num_subfaces;
    ierr = TDyMeshGetVertexFaces(mesh, ivertex, &face_ids, &num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetVertexSubfaces(mesh, ivertex, &subface_ids, &num_subfaces); CHKERRQ(ierr);

    PetscInt num_face = vertices->num_faces[ivertex];

    // Loop over all faces sharing ivertex
    for (PetscInt iface=0; iface<num_face; iface++){

      PetscInt face_id = face_ids[iface];

      // Skip the face that is not locally owned
      if (!faces->is_local[face_id]) continue;

      // Determine the subcell id
      PetscInt *face_cell_ids, num_cell_ids;
      ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
      PetscInt cell_id_up = face_cell_ids[0];
      PetscInt cell_id_dn = face_cell_ids[1];

      // Currently, only zero-flux neumann boundary condition is implemented.
      // If the boundary condition is neumann, then gravity discretization term is zero
      if (tdy->options.mpfao_bc_type == MPFAO_NEUMANN_BC && (cell_id_up < 0 || cell_id_dn < 0)) continue;

      PetscInt cell_id;
      if (cell_id_up < 0) {
        cell_id = cell_id_dn;
      } else {
        cell_id = cell_id_up;
      }

      PetscInt subcell_id;
      ierr = TDyGetSubcellIDGivenCellIdVertexIdFaceId(tdy, cell_id, ivertex, face_id, &subcell_id); CHKERRQ(ierr);

      // area of subface
      PetscReal area = subcells->face_area[subcell_id];

      PetscReal u_up2dn[dim];
      ierr = ComputeUpDownUnitVector(tdy, face_id, u_up2dn); CHKERRQ(ierr);

      PetscReal K_face[dim*dim];
      ierr = ComputeFacePermeabilityTensor(tdy, face_id, K_face);

      // Ku = K_face x u_up2dn
      PetscReal Ku[dim];
      for (PetscInt ii = 0; ii < dim; ii++ ){
        Ku[ii] = 0.0;
        for (PetscInt jj = 0; jj < dim; jj++ ){
          Ku[ii] += K_face[ii*dim + jj] * u_up2dn[jj];
        }
      }

      PetscReal n_face[dim];
      ierr = TDyFace_GetNormal(faces, face_id, dim, &n_face[0]); CHKERRQ(ierr);

      // dot (n_face, K_face x u_up2dn)
      PetscReal dot_prod_1;
      ierr = TDyDotProduct(n_face, Ku, &dot_prod_1); CHKERRQ(ierr);

      // dot(g, u_up2dn)
      PetscReal dot_prod_2;
      ierr = ComputeGtimesZ(tdy->gravity,u_up2dn, dim, &dot_prod_2); CHKERRQ(ierr);

      // GravDis = A_face * dot (n_face, K_face x u_up2dn) * dot(g, u_up2dn)
      PetscReal GravDis = area * dot_prod_1 * dot_prod_2;

      PetscInt isubcell = subface_ids[iface];
      PetscInt num_subcells = 4;
      PetscInt irow = face_id*num_subcells + isubcell;
      gradDisPtr[irow] = GravDis;

    }
  }

  ierr = VecRestoreArray(tdy->GravDisVec, &gradDisPtr); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
