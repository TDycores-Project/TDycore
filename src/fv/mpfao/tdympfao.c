#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyoptions.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdydiscretizationimpl.h>
#include <petscblaslapack.h>

static PetscErrorCode ComputeEntryOfGMatrix(PetscReal area, PetscReal n[3],
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
        K[ii][jj] = mpfao->K0[icell*dim*dim + ii*dim + jj];
      }
    }

    // extract thermal conductivity tensor
    PetscReal Kappa[3][3];
    if (mpfao->Temp_subc_Gmatrix) { // TH
      for (PetscInt ii=0; ii<dim; ii++) {
        for (PetscInt jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = mpfao->Kappa0[icell*dim*dim + ii*dim + jj];
        }
      }
    } else if (mpfao->Psi_subc_Gmatrix) { // SALINITY
      for (PetscInt ii=0; ii<dim; ii++) {
        for (PetscInt jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = mpfao->D_saline[icell*dim*dim + ii*dim + jj];
        }
      }
    }

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

          ierr = ComputeEntryOfGMatrix(area, normal, K, nu, subcells->T[subcell_id], dim,
                                       &(mpfao->subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);

          if (mpfao->Temp_subc_Gmatrix) { // TH
            ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

            ierr = ComputeEntryOfGMatrix(area, normal, Kappa,
              nu, subcells->T[subcell_id], dim,
              &(mpfao->Temp_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
          } else if (mpfao->Psi_subc_Gmatrix) { // SALINITY
            ierr = TDySubCell_GetIthNuVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);
 	    	    ierr = ComputeEntryOfGMatrix(area, normal, Kappa,
      	      nu, subcells->T[subcell_id], dim,
              &(mpfao->Psi_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
          }
        } // jj-subcell-faces
      } // ii-isubcell faces
    } // isubcell
  } // icell

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeGMatrix_TPF(TDyMPFAO *mpfao, DM dm,
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
        K[ii][jj] = mpfao->K0[icell*dim*dim + ii*dim + jj];
      }
    }

    // extract thermal conductivity tensor
    PetscReal Kappa[3][3];
    if (mpfao->Temp_subc_Gmatrix) { // TH
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = mpfao->Kappa0[icell*dim*dim + ii*dim + jj];
        }
      }
    } else if (mpfao->Psi_subc_Gmatrix) { // SALINITY
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          Kappa[ii][jj] = mpfao->D_saline[icell*dim*dim + ii*dim + jj];
        }
      }
    }

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

            mpfao->subc_Gmatrix[icell][isubcell][ii][jj] = 0.0;

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
                  K_neighbor[kk][mm] = mpfao->K0[neighbor_cell_id*dim*dim + kk*dim + mm];
                }
              }
            } else {
              ierr = TDyFace_GetCentroid(faces, face_id, dim, &neighbor_cell_cen[0]); CHKERRQ(ierr);
              ierr = TDyComputeLength(neighbor_cell_cen, cell_cen, dim, &dist); CHKERRQ(ierr);
              for (kk=0; kk<dim; kk++) {
                for (mm=0; mm<dim; mm++) {
                  K_neighbor[kk][mm] = mpfao->K0[icell*dim*dim + kk*dim + mm];
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

            mpfao->subc_Gmatrix[icell][isubcell][ii][jj] = area * (dot_prod) * K_aveg/(dist);
          }

          if (mpfao->Temp_subc_Gmatrix) { // TH
            if (ii == jj) {
              ierr = TDySubCell_GetIthNuStarVector(subcells, subcell_id, jj, dim, &nu[0]); CHKERRQ(ierr);

              ierr = ComputeEntryOfGMatrix(area, normal, Kappa, nu, subcells->T[subcell_id], dim,
                                           &(mpfao->Temp_subc_Gmatrix[icell][isubcell][ii][jj])); CHKERRQ(ierr);
            } else {
              mpfao->Temp_subc_Gmatrix[icell][isubcell][ii][jj] = 0.0;
            }
          } // TH

        } // jj-subcell-faces
      } // ii-isubcell faces
    } // isubcell
  } // icell

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeGMatrix(TDyMPFAO* mpfao, DM dm,
                                     MaterialProp *matprop) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  switch (mpfao->gmatrix_method) {
    case MPFAO_GMATRIX_DEFAULT:
      ierr = ComputeGMatrix_MPFAO(mpfao, dm, matprop); CHKERRQ(ierr);
      break;

    case MPFAO_GMATRIX_TPF:
      ierr = ComputeGMatrix_TPF(mpfao, dm, matprop); CHKERRQ(ierr);
      break;
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode AllocateMemoryForBoundaryValues(TDyMPFAO *mpfao,
                                                      EOS *eos) {

  TDyMesh *mesh = mpfao->mesh;
  PetscInt nbnd_faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  nbnd_faces = mesh->num_boundary_faces;

  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->Kr_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->dKr_dS_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->S_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->dS_dP_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->d2S_dP2_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->P_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->rho_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->vis_bnd)); CHKERRQ(ierr);

  if (mpfao->Psi_subc_Gmatrix) { // SALINITY
    ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->Psi_bnd)); CHKERRQ(ierr);
  }

  PetscInt i;
  PetscReal dden_dP, d2den_dP2, dmu_dP, d2mu_dP2;
  for (i=0;i<nbnd_faces;i++) {
    PetscReal m_nacl = mpfao->m_nacl[0];
    ierr = EOSComputeWaterDensity(eos,
      mpfao->Pref, mpfao->Tref, m_nacl,
      &(mpfao->rho_bnd[i]), &dden_dP, &d2den_dP2); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos,
      mpfao->Pref, mpfao->Tref, m_nacl,
      &(mpfao->vis_bnd[i]), &dmu_dP, &d2mu_dP2); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode AllocateMemoryForEnergyBoundaryValues(TDyMPFAO *mpfao,
                                                            EOS *eos) {

  TDyMesh *mesh = mpfao->mesh;
  PetscInt nbnd_faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  nbnd_faces = mesh->num_boundary_faces;

  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->T_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(mpfao->h_bnd)); CHKERRQ(ierr);

  PetscInt i;
  PetscReal dh_dP, dh_dT;
  for (i=0;i<nbnd_faces;i++) {
    ierr = EOSComputeWaterEnthalpy(eos, mpfao->Tref, mpfao->Pref,
                                   &(mpfao->h_bnd[i]), &dh_dP, &dh_dT); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode AllocateMemoryForSourceSinkValues(TDyMPFAO *mpfao) {

  TDyMesh *mesh = mpfao->mesh;
  PetscInt ncells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ncells = mesh->num_cells;

  ierr = TDyAlloc(ncells*sizeof(PetscReal),&(mpfao->source_sink)); CHKERRQ(ierr);
  ierr = TDyAlloc(ncells*sizeof(PetscReal),&(mpfao->salinity_source_sink)); CHKERRQ(ierr);

  PetscInt i;
  for (i=0;i<ncells;i++) mpfao->source_sink[i] = 0.0;
  for (i=0;i<ncells;i++) mpfao->salinity_source_sink[i] = 0.0;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode AllocateMemoryForEnergySourceSinkValues(TDyMPFAO *mpfao) {

  TDyMesh *mesh = mpfao->mesh;
  PetscInt ncells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ncells = mesh->num_cells;

  ierr = TDyAlloc(ncells*sizeof(PetscReal),&(mpfao->energy_source_sink)); CHKERRQ(ierr);

  PetscInt i;
  for (i=0;i<ncells;i++) mpfao->energy_source_sink[i] = 0.0;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyMPFAOSetGmatrixMethod(TDy tdy,
                                        TDyMPFAOGmatrixMethod method) {
  PetscFunctionBegin;

  PetscValidPointer(tdy,1);
  TDyMPFAO *mpfao = tdy->context;
  mpfao->gmatrix_method = method;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyMPFAOSetBoundaryConditionType(TDy tdy,
                                                TDyBoundaryConditionType bctype) {
  PetscFunctionBegin;

  PetscValidPointer(tdy,1);
  TDyMPFAO *mpfao = tdy->context;
  mpfao->bc_type = bctype;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreate_MPFAO(void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Allocate a new context for the MPFA-O method.
  TDyMPFAO* mpfao;
  ierr = TDyAlloc(sizeof(TDyMPFAO), &mpfao); CHKERRQ(ierr);
  *context = mpfao;

  // Initialize defaults and data.
  mpfao->gmatrix_method = MPFAO_GMATRIX_DEFAULT;
  mpfao->bc_type = DIRICHLET_BC;
  mpfao->Pref = 101325;
  mpfao->Tref = 25;
  mpfao->gravity[0] = 0; mpfao->gravity[1] = 0; mpfao->gravity[2] = 0;
  mpfao->vel = NULL;
  mpfao->vel_count = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy_MPFAO(void *context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyMPFAO* mpfao = context;

  if (mpfao->vel) { ierr = TDyDeallocate_RealArray_1D(mpfao->vel); CHKERRQ(ierr); }
  if (mpfao->vel_count) { ierr = TDyDeallocate_IntegerArray_1D(mpfao->vel_count); CHKERRQ(ierr); }

  if (mpfao->source_sink) { ierr = TDyFree(mpfao->source_sink); CHKERRQ(ierr); }
  if (mpfao->energy_source_sink) {
    ierr = TDyFree(mpfao->energy_source_sink); CHKERRQ(ierr);
  }
  if (mpfao->salinity_source_sink) {
    ierr = TDyFree(mpfao->salinity_source_sink); CHKERRQ(ierr);
  }

  ierr = TDyFree(mpfao->V); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->X); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->N); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->Kr); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dKr_dS); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->S); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dS_dP); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->d2S_dP2); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dS_dT); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->Sr); CHKERRQ(ierr);

  ierr = TDyFree(mpfao->rho); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->drho_dP); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->d2rho_dP2); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->vis); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dvis_dP); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->d2vis_dP2); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->h); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dh_dP); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dh_dT); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->drho_dT); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->u); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->du_dP); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->du_dT); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dvis_dT); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->m_nacl); CHKERRQ(ierr);

  ierr = TDyFree(mpfao->Kr_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dKr_dS_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->S_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->dS_dP_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->d2S_dP2_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->P_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->rho_bnd); CHKERRQ(ierr);
  ierr = TDyFree(mpfao->vis_bnd); CHKERRQ(ierr);

  if (mpfao->T_bnd) { ierr = TDyFree(mpfao->T_bnd); CHKERRQ(ierr); }
  if (mpfao->h_bnd) { ierr = TDyFree(mpfao->h_bnd); CHKERRQ(ierr); }
  if (mpfao->Psi_bnd) { ierr = TDyFree(mpfao->Psi_bnd); CHKERRQ(ierr); }

  // if (mpfao->subc_Gmatrix) { ierr = TDyDeallocate_RealArray_4D(&mpfao->subc_Gmatrix, mpfao->mesh->num_cells,
  //                                   nsubcells, nrow, ncol); CHKERRQ(ierr); }
  if (mpfao->Trans) {
    ierr = TDyDeallocate_RealArray_3D(mpfao->Trans, mpfao->mesh->num_vertices,
                                      mpfao->nfv);
    CHKERRQ(ierr);
  }
  if (mpfao->Trans_mat) { ierr = MatDestroy(&mpfao->Trans_mat  ); CHKERRQ(ierr); }

  if (mpfao->P_vec       ) { ierr = VecDestroy(&mpfao->P_vec      ); CHKERRQ(ierr); }
  if (mpfao->TtimesP_vec ) { ierr = VecDestroy(&mpfao->TtimesP_vec); CHKERRQ(ierr); }
  if (mpfao->GravDisVec  ) { ierr = VecDestroy(&mpfao->GravDisVec); CHKERRQ(ierr); }

  if (mpfao->subc_Gmatrix) {
    ierr = TDyDeallocate_RealArray_4D(mpfao->subc_Gmatrix,
                                      mpfao->mesh->num_cells, 8, 3);
    CHKERRQ(ierr);
  }

  // TH
  if (mpfao->Temp_subc_Gmatrix) {
    ierr = TDyDeallocate_RealArray_4D(mpfao->Temp_subc_Gmatrix,
                                      mpfao->mesh->num_cells, 8, 3);
    CHKERRQ(ierr);
  }
  if (mpfao->Temp_Trans) {
    ierr = TDyDeallocate_RealArray_3D(mpfao->Temp_Trans, mpfao->mesh->num_vertices,
                                      mpfao->nfv);
    CHKERRQ(ierr);
  }
  if (mpfao->Temp_Trans_mat   ) { ierr = MatDestroy(&mpfao->Temp_Trans_mat  ); CHKERRQ(ierr); }
  if (mpfao->Temp_P_vec       ) { ierr = VecDestroy(&mpfao->Temp_P_vec      ); CHKERRQ(ierr); }
  if (mpfao->Temp_TtimesP_vec ) { ierr = VecDestroy(&mpfao->Temp_TtimesP_vec); CHKERRQ(ierr); }

  // SALINITY
  if (mpfao->Psi_subc_Gmatrix) {
    ierr = TDyDeallocate_RealArray_4D(mpfao->Psi_subc_Gmatrix,
                                      mpfao->mesh->num_cells, 8, 3);
    CHKERRQ(ierr);
  }
  if (mpfao->Psi_Trans) {
    ierr = TDyDeallocate_RealArray_3D(mpfao->Psi_Trans, mpfao->mesh->num_vertices,
                                      mpfao->nfv);
    CHKERRQ(ierr);
  }
  if (mpfao->Psi_Trans_mat) { ierr = MatDestroy(&mpfao->Psi_Trans_mat   ); CHKERRQ(ierr); }
  if (mpfao->Psi_vec      ) { ierr = VecDestroy(&mpfao->Psi_vec      ); CHKERRQ(ierr); }
  if (mpfao->TtimesPsi_vec) { ierr = VecDestroy(&mpfao->TtimesPsi_vec); CHKERRQ(ierr); }

  if (mpfao->c_soil) {
    ierr = TDyFree(mpfao->c_soil); CHKERRQ(ierr);
  }
  if (mpfao->rho_soil) {
    ierr = TDyFree(mpfao->rho_soil); CHKERRQ(ierr);
  }
  if (mpfao->D_saline) {
    ierr = TDyFree(mpfao->D_saline); CHKERRQ(ierr);
  }
  if (mpfao->mu_saline) {
    ierr = TDyFree(mpfao->mu_saline); CHKERRQ(ierr);
  }

  ierr = TDyMeshDestroy(mpfao->mesh);

  TDyFree(mpfao);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetFromOptions_MPFAO(void *context, TDyOptions *options) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  TDyMPFAO* mpfao = context;

  // Set MPFA-O options.
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: MPFA-O options","");
  ierr = PetscOptionsEnum("-tdy_mpfao_gmatrix_method","MPFA-O gmatrix method",
    "TDySetMPFAOGmatrixMethod",TDyMPFAOGmatrixMethods,
    (PetscEnum)mpfao->gmatrix_method,(PetscEnum *)&mpfao->gmatrix_method,NULL);
    CHKERRQ(ierr);
  TDyBoundaryConditionType bctype = DIRICHLET_BC;
  PetscBool flag;
  ierr = PetscOptionsEnum("-tdy_mpfao_boundary_condition_type",
      "MPFA-O boundary condition type", "TDyMPFAOSetBoundaryConditionType",
      TDyBoundaryConditionTypes,(PetscEnum)bctype,(PetscEnum *)&bctype,
      &flag); CHKERRQ(ierr);
  if (flag && (bctype != mpfao->bc_type)) {
    mpfao->bc_type = bctype;
  }
  /* TODO: geometric attribute reading/writing is broken
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_output_geo_attributes",
      mpfao->geom_attributes_file,sizeof(mpfao->geom_attributes_file),
      &mpfao->output_geom_attributes); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_read_geo_attributes",
      mpfao->geom_attributes_file,sizeof(mpfao->geom_attributes_file),
      &mpfao->read_geom_attributes); CHKERRQ(ierr);
  if (mpfao->output_geom_attributes && mpfao->read_geom_attributes){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Only one of -tdy_output_geom_attributes and -tdy_read_geom_attributes can be specified");
  }
  */
  PetscOptionsEnd();

  // Set characteristic curve data.
  mpfao->vangenuchten_m = options->vangenuchten_m;
  mpfao->vangenuchten_alpha = options->vangenuchten_alpha;
  mpfao->mualem_poly_x0 = options->mualem_poly_x0;
  mpfao->mualem_poly_x1 = options->mualem_poly_x1;
  mpfao->mualem_poly_x2 = options->mualem_poly_x2;
  mpfao->mualem_poly_dx = options->mualem_poly_dx;

  // Copy g into place.
  mpfao->gravity[2] = options->gravity_constant;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

//-----------------
// Setup functions
//-----------------

// Initializes material properties and characteristic curve data.
static PetscErrorCode InitMaterials(TDyMPFAO *mpfao,
                                    DM dm,
                                    MaterialProp *matprop,
                                    CharacteristicCurves *cc) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  // Allocate storage for material data and characteristic curves.
  PetscInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscInt nc = cEnd-cStart;

  // Material properties
  ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(mpfao->K)); CHKERRQ(ierr);
  ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(mpfao->K0)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->porosity)); CHKERRQ(ierr);
  if (MaterialPropHasThermalConductivity(matprop)) {
    ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(mpfao->Kappa)); CHKERRQ(ierr);
    ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(mpfao->Kappa0)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilSpecificHeat(matprop)) {
    ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->c_soil)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilDensity(matprop)) {
    ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->rho_soil)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSalineDiffusivity(matprop)) {
    ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(mpfao->D_saline)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSalineMolecularWeight(matprop)) {
    ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->mu_saline)); CHKERRQ(ierr);
  }

  // Characteristic curve values
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->Kr)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dKr_dS)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->S)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dS_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->d2S_dP2)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dS_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->Sr)); CHKERRQ(ierr);

  // Water properties
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->rho)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->drho_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->d2rho_dP2)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->vis)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dvis_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->d2vis_dP2)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->h)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dh_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dh_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->drho_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->u)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->du_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->du_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->dvis_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(mpfao->m_nacl)); CHKERRQ(ierr);

  // Initialize characteristic curve parameters on all cells.
  PetscInt points[nc];
  for (PetscInt c = 0; c < nc; ++c) {
    points[c] = cStart + c;
  }

  // By default, we use the Van Genuchten saturation model.
  {
    PetscReal parameters[2*nc];
    for (PetscInt c = 0; c < nc; ++c) {
      parameters[2*c]   = mpfao->vangenuchten_m;
      parameters[2*c+1] = mpfao->vangenuchten_alpha;
    }
    ierr = SaturationSetType(cc->saturation, SAT_FUNC_VAN_GENUCHTEN, nc, points,
                             parameters); CHKERRQ(ierr);
  }

  // By default, we use the the Mualem relative permeability model.
  {
    PetscInt num_params = 9;
    PetscReal parameters[num_params*nc];
    for (PetscInt c = 0; c < nc; ++c) {
      PetscReal m = mpfao->vangenuchten_m;
      PetscReal poly_x0 = mpfao->mualem_poly_x0;
      PetscReal poly_x1 = mpfao->mualem_poly_x1;
      PetscReal poly_x2 = mpfao->mualem_poly_x2;
      PetscReal poly_dx = mpfao->mualem_poly_dx;

      PetscInt offset = num_params*c;
      parameters[offset    ]   = m;
      parameters[offset + 1] = poly_x0;
      parameters[offset + 2] = poly_x1;
      parameters[offset + 3] = poly_x2;
      parameters[offset + 4] = poly_dx;

      // Set up cubic polynomial coefficients for the cell.
      PetscReal coeffs[4];
      ierr = RelativePermeability_Mualem_GetSmoothingCoeffs(m, poly_x0, poly_x1, poly_x2, poly_dx, coeffs);
      CHKERRQ(ierr);
      parameters[offset + 5] = coeffs[0];
      parameters[offset + 6] = coeffs[1];
      parameters[offset + 7] = coeffs[2];
      parameters[offset + 8] = coeffs[3];
    }
    ierr = RelativePermeabilitySetType(cc->rel_perm, REL_PERM_FUNC_MUALEM, nc,
                                       points, parameters); CHKERRQ(ierr);
  }

  // Compute material properties.
  ierr = MaterialPropComputePermeability(matprop, nc, mpfao->X, mpfao->K0); CHKERRQ(ierr);
  memcpy(mpfao->K, mpfao->K0, 9*nc*sizeof(PetscReal));
  ierr = MaterialPropComputePorosity(matprop, nc, mpfao->X, mpfao->porosity); CHKERRQ(ierr);
  ierr = MaterialPropComputeResidualSaturation(matprop, nc, mpfao->X, mpfao->Sr); CHKERRQ(ierr);
  if (MaterialPropHasThermalConductivity(matprop)) {
    ierr = MaterialPropComputeThermalConductivity(matprop, nc, mpfao->X, mpfao->Kappa); CHKERRQ(ierr);
    memcpy(mpfao->Kappa0, mpfao->Kappa, 9*nc*sizeof(PetscReal));
  }
  if (MaterialPropHasSoilSpecificHeat(matprop)) {
    ierr = MaterialPropComputeSoilSpecificHeat(matprop, nc, mpfao->X, mpfao->c_soil); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilDensity(matprop)) {
    ierr = MaterialPropComputeSoilDensity(matprop, nc, mpfao->X, mpfao->rho_soil); CHKERRQ(ierr);
  }
  if (MaterialPropHasSalineDiffusivity(matprop)) {
    ierr = MaterialPropComputeSalineDiffusivity(matprop, nc, mpfao->X, mpfao->D_saline); CHKERRQ(ierr);
  }
  if (MaterialPropHasSalineMolecularWeight(matprop)) {
    ierr = MaterialPropComputeSalineMolecularWeight(matprop, nc, mpfao->X, mpfao->mu_saline); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// Create a section containing a given number of fields with given names and
// numbers of degrees of freedom, and attach it to the given DM.
static PetscErrorCode SetFields(DM dm, PetscInt num_fields,
                                const char* field_names[num_fields],
                                PetscInt num_field_dof[num_fields]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  // Create the section and register fields and components.
  PetscSection sec;
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, num_fields); CHKERRQ(ierr);
  PetscInt total_num_dof = 0; // total number of field dofs/components per point
  for (PetscInt f = 0; f < num_fields; ++f) {
    ierr = PetscSectionSetFieldName(sec, f, field_names[f]); CHKERRQ(ierr);
    // TODO: should we distinguish between field components and dof?
    ierr = PetscSectionSetFieldComponents(sec, f, num_field_dof[f]); CHKERRQ(ierr);
    total_num_dof += num_field_dof[f];
  }

  // Create a chart on cells.
  PetscInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,cStart,cEnd); CHKERRQ(ierr);

  // Assign degrees of freedom to each field on each cell.
  for(PetscInt c=cStart; c<cEnd; c++) {
    for (PetscInt f = 0; f < num_fields; ++f) {
      ierr = PetscSectionSetFieldDof(sec, c, f, num_field_dof[f]); CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(sec, c, total_num_dof); CHKERRQ(ierr);
  }

  // Assign the section to the DM.
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetLocalSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);

  // TODO: Does this really belong here, or can we move it elsewhere?
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetNumDMFields_Richards_MPFAO(void *context) {
  PetscFunctionBegin;
  PetscInt ndof = 1; // LiquidPressure
  PetscFunctionReturn(ndof);
}

PetscErrorCode TDyGetNumDMFields_Richards_MPFAO_DAE(void *context) {
  PetscFunctionBegin;
  PetscInt ndof = 2; // LiquidPressure, LiquidMass
  PetscFunctionReturn(ndof);
}

PetscErrorCode TDyGetNumDMFields_TH_MPFAO(void *context) {
  PetscFunctionBegin;
  PetscInt ndof = 2; // LiquidPressure, LiquidTemperature
  PetscFunctionReturn(ndof);
}

PetscErrorCode TDyGetNumDMFields_Salinity_MPFAO(void *context) {
  PetscFunctionBegin;
  PetscInt ndof = 2; // LiquidPressure, LiquidTemperature
  PetscFunctionReturn(ndof);
}

PetscErrorCode TDySetDMFields_Richards_MPFAO(void *context, DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // Set up the section, 1 dof per cell
  ierr = SetFields(dm, 1, (const char*[1]){"LiquidPressure"}, (PetscInt[1]){1});
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDMFields_Richards_MPFAO_DAE(void *context, DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // Set up the section, 2 dofs per cell.
  ierr = SetFields(dm, 2, (const char*[2]){"LiquidPressure", "LiquidMass"},
                   (PetscInt[2]){1, 1}); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDMFields_TH_MPFAO(void *context, DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // Set up the section, 2 dofs per cell.
  ierr = SetFields(dm, 2, (const char*[2]){"LiquidPressure", "LiquidTemperature"},
                   (PetscInt[2]){1, 1}); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDMFields_Salinity_MPFAO(void *context, DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // Set up the section, 2 dofs per cell.
  ierr = SetFields(dm, 2, (const char*[2]){"LiquidPressure", "SalineConcentration"},
                   (PetscInt[2]){1, 1}); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ExtractSubMatrix(PetscReal **M, PetscInt rStart, PetscInt rEnd,
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

static PetscErrorCode ExtractsubCMatrices(PetscInt nrow, PetscInt ncol, PetscReal **C,
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

static PetscErrorCode ExtractsubFMatrices(PetscReal **F,
                                          PetscInt nrow, PetscInt ncol,
                                          PetscInt nrow_1, PetscInt nrow_2,
                                          PetscInt nrow_3, PetscReal ***F_1,
                                          PetscReal ***F_2, PetscReal ***F_3) {

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

static PetscErrorCode DetermineNumberOfUpAndDownBoundaryFaces(TDyMPFAO *mpfao,
    PetscInt ivertex, PetscInt *nflux_bc_up, PetscInt *nflux_bc_dn) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  TDyMesh *mesh = mpfao->mesh;
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

static PetscErrorCode ComputeCandFmatrix(TDyMPFAO *mpfao, PetscInt ivertex,
  PetscInt varID, PetscReal **Cup, PetscReal **Cdn, PetscReal **Fup,
  PetscReal **Fdn) {

  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  TDyVertex *vertices = &mesh->vertices;

  PetscInt vOffsetCell    = vertices->internal_cell_offset[ivertex];
  PetscInt vOffsetSubcell = vertices->subcell_offset[ivertex];

  PetscInt npcen = vertices->num_internal_cells[ivertex];
  PetscReal **Gmatrix;
  PetscInt ndim = 3;

  ierr = TDyAllocate_RealArray_2D(&Gmatrix, ndim   , ndim   ); CHKERRQ(ierr);

  for (PetscInt i=0; i<npcen; i++) {
    PetscInt icell    = vertices->internal_cell_ids[vOffsetCell + i];
    PetscInt isubcell = vertices->subcell_ids[vOffsetSubcell + i];

    PetscInt subcell_id = icell*cells->num_subcells[icell]+isubcell;

    PetscInt *face_unknown_idx, *is_face_up, *face_flux_idx, subcell_num_faces;
    ierr = TDyMeshGetSubcellFaceUnknownIdxs(mesh, subcell_id, &face_unknown_idx, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellIsFaceUp(mesh, subcell_id, &is_face_up, &subcell_num_faces); CHKERRQ(ierr);
    ierr = TDyMeshGetSubcellFaceFluxIdxs(mesh, subcell_id, &face_flux_idx, &subcell_num_faces); CHKERRQ(ierr);

    if (varID == VAR_PRESSURE) {
      ierr = ExtractSubGmatrix(mpfao, icell, isubcell, ndim, Gmatrix);
    } else if (varID == VAR_TEMPERATURE){
      ierr = ExtractTempSubGmatrix(mpfao, icell, isubcell, ndim, Gmatrix);
    } else if (varID == VAR_SALINE_CONCENTRATION){
      ierr = ExtractPsiSubGmatrix(mpfao, icell, isubcell, ndim, Gmatrix);
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

  ierr = TDyDeallocate_RealArray_2D(Gmatrix, ndim); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeAinvB(PetscInt A_nrow, PetscReal *A,
    PetscInt B_ncol, PetscReal *B, PetscReal *AinvB) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt m, n;
  PetscErrorCode ierr;

  m = A_nrow; n = A_nrow;

  PetscBLASInt info, pivots[n+1];
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

static PetscErrorCode ComputeCtimesAinvB(PetscInt C_nrow, PetscInt AinvB_ncol, PetscInt C_ncol,
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

static PetscErrorCode ComputeTransmissibilityMatrix_ForNonCornerVertex(
    TDyMPFAO *mpfao, PetscInt ivertex, TDyCell *cells, PetscInt varID) {

  TDyMesh *mesh = mpfao->mesh;
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
  PetscInt ndim = 3;
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

  PetscInt npcen        = vertices->num_internal_cells[ivertex];
  PetscInt npitf_bc_all = vertices->num_boundary_faces[ivertex];

  PetscInt nflux_all_bc_up, nflux_all_bc_dn;
  PetscInt nflux_dir_bc_up, nflux_dir_bc_dn;
  PetscInt nflux_neu_bc_up, nflux_neu_bc_dn;
  ierr = DetermineNumberOfUpAndDownBoundaryFaces(mpfao, ivertex, &nflux_all_bc_up, &nflux_all_bc_dn);

  PetscInt npitf_dir_bc_all, npitf_neu_bc_all;

  if (mpfao->bc_type == DIRICHLET_BC ||
      mpfao->bc_type == SEEPAGE_BC) {
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
  //  (1) number of internal and boundary fluxes,
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

  ierr = ComputeCandFmatrix(mpfao, ivertex, varID, Cup_all, Cdn_all, Fup_all, Fdn_all); CHKERRQ(ierr);

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
    Trans = &mpfao->Trans;
    Trans_mat = &mpfao->Trans_mat;
  } else if (varID == VAR_TEMPERATURE) {
    Trans = &mpfao->Temp_Trans;
    Trans_mat = &mpfao->Temp_Trans_mat;
  } else if (varID == VAR_SALINE_CONCENTRATION) {
     Trans = &mpfao->Psi_Trans;
     Trans_mat = &mpfao->Psi_Trans_mat;
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
      col = idxBnd[j] + mpfao->mesh->num_cells;
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

  ierr = TDyDeallocate_RealArray_2D(Cup_11, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_12, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_13, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_21, nflux_dir_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_22, nflux_dir_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_23, nflux_dir_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_31, nflux_neu_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_32, nflux_neu_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cup_33, nflux_neu_bc_up); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_11, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_12, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_13, nflux_in); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_21, nflux_dir_bc_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_22, nflux_dir_bc_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_23, nflux_dir_bc_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_31, nflux_neu_bc_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_32, nflux_neu_bc_dn); CHKERRQ(ierr);
  ierr = TDyDeallocate_RealArray_2D(Cdn_33, nflux_neu_bc_dn); CHKERRQ(ierr);

  TDyFree(AinvB_1d);
  TDyFree(CupInxIntimesAinvB_1d);
  TDyFree(CupBCxIntimesAinvB_1d);
  TDyFree(CdnBCxIntimesAinvB_1d);
  TDyFree(AINBCxINBC_1d);
  TDyFree(BINBCxCDBC_1d);
  TDyFree(CupINBCxINBC_1d);
  TDyFree(CupDBCxIn_1d);
  TDyFree(CdnDBCxIn_1d);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

static PetscErrorCode ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(
    TDyMPFAO *mpfao, DM dm, PetscInt ivertex, TDyCell *cells, PetscInt varID) {
  TDyMesh *mesh = mpfao->mesh;
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
    Trans = &mpfao->Trans;
    Trans_mat = &mpfao->Trans_mat;
    ierr = ExtractSubGmatrix(mpfao, icell, isubcell, dim, Gmatrix);
  } else if (varID == VAR_TEMPERATURE) {
    ierr = ExtractTempSubGmatrix(mpfao, icell, isubcell, dim, Gmatrix);
    Trans = &mpfao->Temp_Trans;
    Trans_mat = &mpfao->Temp_Trans_mat;
  } else if (varID == VAR_SALINE_CONCENTRATION) {
    ierr = ExtractPsiSubGmatrix(mpfao, icell, isubcell, dim, Gmatrix);
    Trans = &mpfao->Psi_Trans;
    Trans_mat = &mpfao->Psi_Trans_mat;
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
      col = idxBnd[j] + mpfao->mesh->num_cells;
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
  ierr = TDyDeallocate_RealArray_2D(Gmatrix, dim); CHKERRQ(ierr);

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

static PetscErrorCode UpdateTransmissibilityMatrix(TDyMPFAO *mpfao) {

  TDyMesh       *mesh = mpfao->mesh;
  TDyRegion     *region = &mesh->region_connected;
  PetscInt       iface, isubface;
  PetscInt       num_subfaces = 4;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  // If a face is shared by two cells that belong to different
  // regions, zero the rows in the transmissiblity matrix

  for (iface=0; iface<mesh->num_faces; iface++) {

    PetscInt *face_cell_ids, num_cell_ids;
    ierr = TDyMeshGetFaceCells(mesh, iface, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
    PetscInt cell_id_up = face_cell_ids[0];
    PetscInt cell_id_dn = face_cell_ids[1];

    if (cell_id_up>=0 && cell_id_dn>=0) {
      if (!TDyRegionAreCellsInTheSameRegion(region, cell_id_up, cell_id_dn)) {
        for (isubface=0; isubface<4; isubface++) {
          PetscInt row[1];
          row[0] = iface*num_subfaces + isubface;
          ierr = MatZeroRows(mpfao->Trans_mat,1,row,0.0,0,0); CHKERRQ(ierr);
          if (mpfao->Temp_subc_Gmatrix) { // TH
            ierr = MatZeroRows(mpfao->Temp_Trans_mat,1,row,0.0,0,0); CHKERRQ(ierr);
          }
          if (mpfao->Psi_subc_Gmatrix) { // SALINITY
            ierr = MatZeroRows(mpfao->Psi_Trans_mat,1,row,0.0,0,0); CHKERRQ(ierr);
          }
        }
      }
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

static PetscErrorCode ComputeTransmissibilityMatrix(TDyMPFAO *mpfao, DM dm) {

  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  TDyVertex     *vertices = &mesh->vertices;
  PetscInt       ivertex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  for (ivertex=0; ivertex<mesh->num_vertices; ivertex++) {

    if (!vertices->is_local[ivertex]) continue;

    if (vertices->num_boundary_faces[ivertex] == 0 || vertices->num_internal_cells[ivertex] > 1) {
      ierr = ComputeTransmissibilityMatrix_ForNonCornerVertex(mpfao, ivertex, cells, 0); CHKERRQ(ierr);
      if (mpfao->Temp_subc_Gmatrix) { // TH
        ierr = ComputeTransmissibilityMatrix_ForNonCornerVertex(mpfao, ivertex, cells, 1); CHKERRQ(ierr);
      }
    } else {
      // It is assumed that neumann boundary condition is a zero-flux boundary condition.
      // Thus, compute transmissiblity entries only for dirichlet boundary condition.
      if (mpfao->bc_type == DIRICHLET_BC ||
          mpfao->bc_type == SEEPAGE_BC) {
        ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(mpfao, dm, ivertex, cells, 0); CHKERRQ(ierr);
        if (mpfao->Temp_subc_Gmatrix) { // TH
          ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(mpfao, dm, ivertex, cells, 1); CHKERRQ(ierr);
        } else if (mpfao->Psi_subc_Gmatrix) { // SALINITY
          ierr = ComputeTransmissibilityMatrix_ForBoundaryVertex_NotSharedWithInternalVertices(mpfao, dm, ivertex, cells, 2); CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = MatAssemblyBegin(mpfao->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mpfao->Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (mpfao->Temp_subc_Gmatrix) { // TH
    ierr = MatAssemblyBegin(mpfao->Temp_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mpfao->Temp_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  if (mpfao->Psi_subc_Gmatrix) { // SALINITY
    ierr = MatAssemblyBegin(mpfao->Psi_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mpfao->Psi_Trans_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  TDyRegion *region = &mesh->region_connected;
  if (region->num_cells > 0){
    if (mpfao->gmatrix_method == MPFAO_GMATRIX_TPF ) {
      ierr = UpdateTransmissibilityMatrix(mpfao); CHKERRQ(ierr);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"WARNING -- Connected region option is only supported with MPFA-O TPF\n");
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/// Computes unit vector joining upwind and downwind cells that share a face.
/// The unit vector points from upwind to downwind cell.
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] up2dn_uvec Unit vector from upwind to downwind cell
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ComputeUpDownUnitVector(TDyMPFAO *mpfao, PetscInt face_id, PetscReal up2dn_uvec[3]) {

  PetscFunctionBegin;

  TDyMesh *mesh = mpfao->mesh;
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

/// Computes upwind and downwind distance of cells sharing a face. If the face is a
/// boundary face, one of the distance is zero
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] dist_up Distance between the upwind cell centroid and face centroid
/// @param [out] dist_dn Distance between the downwind cell centroid and face centroid
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeUpAndDownDist(TDyMPFAO *mpfao, PetscInt face_id, PetscReal *dist_up, PetscReal *dist_dn) {

  PetscFunctionBegin;

  TDyMesh *mesh = mpfao->mesh;
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

/// Computes face permeability tensor as a harmonically distance-weighted
//  permeability of upwind and downwind permeability tensors
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] Kup components of the upwind permeability tensor in a row-major order
/// @param [out] Kdn components of the downwind permeability tensor in a row-major order
PetscErrorCode ExtractUpAndDownPermeabilityTensors(TDyMPFAO *mpfao,
    MaterialProp *matprop, PetscInt face_id, PetscInt dim,
    PetscReal Kup[dim*dim], PetscReal Kdn[dim*dim]) {

  PetscFunctionBegin;

  TDyMesh *mesh = mpfao->mesh;
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
      if (cell_id_up >= 0) Kup[mm * dim + kk] = mpfao->K0[cell_id_up * dim * dim + kk * dim + mm];
      else                 Kup[mm * dim + kk] = mpfao->K0[cell_id_dn * dim * dim + kk * dim + mm];

      if (cell_id_dn >= 0) Kdn[mm * dim + kk] = mpfao->K0[cell_id_dn * dim * dim + kk * dim + mm];
      else                 Kdn[mm * dim + kk] = mpfao->K0[cell_id_up * dim * dim + kk * dim + mm];
    }
  }

  PetscFunctionReturn(0);
}

/// Computes face permeability tensor as a harmonically distance-weighted
//  permeability of upwind and downwind permeability tensors
///
/// K = (wt_1 * K_u^{-1} + (1-wt_1) * K_d^{-1})^{-1}
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] Kface components of face permeability tensor in row-major order
static PetscErrorCode ComputeFacePermeabilityTensor(TDyMPFAO *mpfao,
    MaterialProp *matprop, PetscInt face_id, PetscReal Kface[9]){

  PetscFunctionBegin;

  PetscErrorCode ierr;
  PetscReal dist_up, dist_dn;

  ierr = ComputeUpAndDownDist(mpfao, face_id, &dist_up, &dist_dn); CHKERRQ(ierr);

  PetscInt dim = 3;

  PetscReal Kup[dim*dim], Kdn[dim*dim];
  PetscReal KupInv[dim*dim], KdnInv[dim*dim];

  ierr = ExtractUpAndDownPermeabilityTensors(mpfao, matprop, face_id, dim, Kup, Kdn); CHKERRQ(ierr);

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

/// For the TPF approach, compute the value of permeability at face center that
/// is used in computing the discretization of gravity term.
///
/// - First, compute permeability scalar values from the up and down cell permeability
///   tensor using PFLOTRAN's approach
/// - Second, compute distance-weighted harmonic-average of scalar permeabilities
///
/// @param [in] tdy A TDy struct
/// @param [in] dim dimension of the problem
/// @param [in] face_id ID of the face
/// @param [out] *Kface_value Permeability value at the face
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ComputeFacePermeabilityValueTPF(TDyMPFAO *mpfao, MaterialProp *matprop, PetscInt dim, PetscInt face_id, PetscReal *Kface_value) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscReal u_up2dn[dim];
  ierr = ComputeUpDownUnitVector(mpfao, face_id, u_up2dn); CHKERRQ(ierr);

  PetscReal Kup[dim*dim], Kdn[dim*dim];
  ierr = ExtractUpAndDownPermeabilityTensors(mpfao, matprop, face_id, dim, Kup, Kdn); CHKERRQ(ierr);

  PetscReal Kup_value = 0.0, Kdn_value = 0.0;
  for (PetscInt kk=0; kk<dim; kk++) {
    Kup_value += pow(u_up2dn[kk],2.0)/Kup[kk*dim + kk];
    Kdn_value += pow(u_up2dn[kk],2.0)/Kdn[kk*dim + kk];
  }

  Kup_value = 1.0/Kup_value;
  Kdn_value = 1.0/Kdn_value;

  PetscReal dist_up, dist_dn;
  ierr = ComputeUpAndDownDist(mpfao, face_id, &dist_up, &dist_dn); CHKERRQ(ierr);

  PetscReal wt_up = dist_up / (dist_up + dist_dn);

  *Kface_value = (Kup_value*Kdn_value)/(wt_up*Kdn_value + (1.0-wt_up)*Kup_value);

  PetscFunctionReturn(0);
}

/// For the MPFAO approach, compute the value of permeability at face center that
/// is used in computing the discretization of gravity term.
///
/// - First, permeability tensor at the face is computed by harmonically averaging
///    permeability tensors of cells share the face
/// - Second, the face permeability tensor is projected along the unit normal
///   along the line joining up and down cells
/// - Third, dot producot of the projected face permeability vector and unit
///   normal to the face is computed
///
/// @param [in] tdy A TDy struct
/// @param [in] dim dimension of the problem
/// @param [in] face_id ID of the face
/// @param [out] *k_face_value Permeability value at the face
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode ComputeFacePermeabilityValueMPFAO(TDyMPFAO *mpfao, MaterialProp *matprop, PetscInt dim, PetscInt face_id, PetscReal *Kface_value) {

  PetscFunctionBegin;

  TDyMesh *mesh = mpfao->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscReal u_up2dn[dim];
  ierr = ComputeUpDownUnitVector(mpfao, face_id, u_up2dn); CHKERRQ(ierr);

  PetscReal Kface[dim*dim];
  ierr = ComputeFacePermeabilityTensor(mpfao, matprop, face_id, Kface);

  // Ku = Kface x u_up2dn
  PetscReal Ku[dim];
  for (PetscInt ii = 0; ii < dim; ii++ ){
    Ku[ii] = 0.0;
    for (PetscInt jj = 0; jj < dim; jj++ ){
      Ku[ii] += Kface[ii*dim + jj] * u_up2dn[jj];
    }
  }

  PetscReal n_face[dim];
  ierr = TDyFace_GetNormal(faces, face_id, dim, &n_face[0]); CHKERRQ(ierr);

  // dot (n_face, K_face x u_up2dn)
  ierr = TDyDotProduct(n_face, Ku, Kface_value); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
static PetscErrorCode ComputeGravityDiscretization(TDyMPFAO *mpfao, DM dm,
                                                   MaterialProp *matprop) {

  PetscFunctionBegin;

  TDY_START_FUNCTION_TIMER()

  TDyMesh *mesh = mpfao->mesh;
  TDyFace *faces = &mesh->faces;
  TDyVertex *vertices = &mesh->vertices;
  TDySubcell *subcells = &mesh->subcells;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscScalar *gradDisPtr;
  ierr = VecGetArray(mpfao->GravDisVec, &gradDisPtr); CHKERRQ(ierr);

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
      if (mpfao->bc_type == NEUMANN_BC && (cell_id_up < 0 || cell_id_dn < 0)) continue;

      PetscInt cell_id;
      if (cell_id_up < 0) {
        cell_id = cell_id_dn;
      } else {
        cell_id = cell_id_up;
      }

      PetscInt subcell_id;
      ierr = TDyMeshGetSubcellIDGivenCellIdVertexIdFaceId(mesh, cell_id,
        ivertex, face_id, &subcell_id); CHKERRQ(ierr);

      // area of subface
      PetscReal area = subcells->face_area[subcell_id];

      PetscReal u_up2dn[dim];
      ierr = ComputeUpDownUnitVector(mpfao, face_id, u_up2dn); CHKERRQ(ierr);

      PetscReal k_face_value;
      switch (mpfao->gmatrix_method) {
        case MPFAO_GMATRIX_DEFAULT:
          ierr = ComputeFacePermeabilityValueMPFAO(mpfao, matprop, dim, face_id, &k_face_value);
          break;

      case MPFAO_GMATRIX_TPF:
        ierr = ComputeFacePermeabilityValueTPF(mpfao, matprop, dim, face_id, &k_face_value); CHKERRQ(ierr);

        PetscReal n_face[dim], dot_prod;
        ierr = TDyFace_GetNormal(faces, face_id, dim, &n_face[0]); CHKERRQ(ierr);

        // dot product between face normal and unit vector along up-down cell
        ierr = TDyDotProduct(n_face, u_up2dn, &dot_prod); CHKERRQ(ierr);

        area *= dot_prod;
        break;
      }

      // dot(g, u_up2dn)
      PetscReal dot_prod_2;
      ierr = ComputeGtimesZ(mpfao->gravity,u_up2dn, dim, &dot_prod_2); CHKERRQ(ierr);

      // GravDis = A_face * dot (n_face, K_face x u_up2dn) * dot(g, u_up2dn)
      PetscReal GravDis = area * k_face_value * dot_prod_2;

      PetscInt isubcell = subface_ids[iface];
      PetscInt num_subcells = 4;
      PetscInt irow = face_id*num_subcells + isubcell;
      gradDisPtr[irow] = GravDis;

    }
  }

  ierr = VecRestoreArray(mpfao->GravDisVec, &gradDisPtr); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

// Setup function for Richards + MPFA_O
PetscErrorCode TDySetup_Richards_MPFAO(void *context,
                                       TDyDiscretizationType* discretization,
                                       EOS *eos,
                                       MaterialProp *matprop,
                                       CharacteristicCurves *cc,
                                       Conditions *conditions) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  TDyMPFAO *mpfao = context;
  DM dm;
  ierr = TDyDiscretizationGetDM(discretization,&dm); CHKERRQ(ierr);

  ierr = TDyMeshCreateFromPlex(dm, &mpfao->V, &mpfao->X, &mpfao->N, &mpfao->mesh);
  ierr = TDyMeshGetMaxVertexConnectivity(mpfao->mesh, &mpfao->ncv, &mpfao->nfv);

  ierr = TDyAllocate_RealArray_1D(&(mpfao->vel), mpfao->mesh->num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&(mpfao->vel_count), mpfao->mesh->num_faces); CHKERRQ(ierr);

  ierr = InitMaterials(mpfao, dm, matprop, cc); CHKERRQ(ierr);

  // Gather mesh data.
  PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
  PetscInt nrow, ncol, nz;

  nFaces = mpfao->mesh->num_faces;
  nLocalCells = mpfao->mesh->num_cells_local;
  nNonLocalFaces = TDyMeshGetNumberOfNonLocalFaces(mpfao->mesh);
  nNonInternalFaces = TDyMeshGetNumberOfNonInternalFaces(mpfao->mesh);

  nrow = 4*nFaces;
  ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
  nz   = mpfao->nfv;
  ierr = TDyAllocate_RealArray_3D(&mpfao->Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv + mpfao->ncv); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Trans_mat); CHKERRQ(ierr);
  ierr = MatSetOption(mpfao->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->P_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesP_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->GravDisVec);
  ierr = VecZeroEntries(mpfao->GravDisVec);
  PetscInt nsubcells = 8;
  ierr = TDyAllocate_RealArray_4D(&mpfao->subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, 3, 3); CHKERRQ(ierr);

  // Set up data structures for the discretization.
  ierr = ComputeGMatrix(mpfao, dm, matprop); CHKERRQ(ierr);
  ierr = ComputeTransmissibilityMatrix(mpfao, dm); CHKERRQ(ierr);
  ierr = ComputeGravityDiscretization(mpfao, dm, matprop); CHKERRQ(ierr);

  ierr = AllocateMemoryForBoundaryValues(mpfao, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForSourceSinkValues(mpfao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup function for Richards + MPFAO_DAE
PetscErrorCode TDySetup_Richards_MPFAO_DAE(void *context, TDyDiscretizationType *discretization, EOS *eos,
                                           MaterialProp *matprop,
                                           CharacteristicCurves *cc,
                                           Conditions* conditions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);

  TDyMPFAO *mpfao = context;
  DM dm;
  ierr = TDyDiscretizationGetDM(discretization,&dm); CHKERRQ(ierr);

  ierr = TDyMeshCreateFromPlex(dm, &mpfao->V, &mpfao->X, &mpfao->N, &mpfao->mesh);
  ierr = TDyMeshGetMaxVertexConnectivity(mpfao->mesh, &mpfao->ncv, &mpfao->nfv);

  ierr = TDyAllocate_RealArray_1D(&(mpfao->vel), mpfao->mesh->num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&(mpfao->vel_count), mpfao->mesh->num_faces); CHKERRQ(ierr);

  ierr = InitMaterials(mpfao, dm, matprop, cc); CHKERRQ(ierr);

  // Gather mesh data.
  PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
  PetscInt nrow, ncol, nz;

  nFaces = mpfao->mesh->num_faces;
  nLocalCells = mpfao->mesh->num_cells_local;
  nNonLocalFaces = TDyMeshGetNumberOfNonLocalFaces(mpfao->mesh);
  nNonInternalFaces = TDyMeshGetNumberOfNonInternalFaces(mpfao->mesh);

  nrow = 4*nFaces;
  ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
  nz   = mpfao->nfv;
  ierr = TDyAllocate_RealArray_3D(&mpfao->Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv + mpfao->ncv); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Trans_mat); CHKERRQ(ierr);
  ierr = MatSetOption(mpfao->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->P_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesP_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->GravDisVec);
  ierr = VecZeroEntries(mpfao->GravDisVec);
  PetscInt nsubcells = 8;
  ierr = TDyAllocate_RealArray_4D(&mpfao->subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, 3, 3); CHKERRQ(ierr);

  // Set up data structures for the discretization.
  ierr = ComputeGMatrix(mpfao, dm, matprop); CHKERRQ(ierr);
  ierr = ComputeTransmissibilityMatrix(mpfao, dm); CHKERRQ(ierr);
  ierr = ComputeGravityDiscretization(mpfao, dm, matprop); CHKERRQ(ierr);

  ierr = AllocateMemoryForBoundaryValues(mpfao, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForSourceSinkValues(mpfao); CHKERRQ(ierr);
}

// Setup function for TH + MPFA-O
PetscErrorCode TDySetup_TH_MPFAO(void *context, TDyDiscretizationType *discretization, EOS *eos,
                                 MaterialProp *matprop,
                                 CharacteristicCurves *cc,
                                 Conditions* conditions) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  TDyMPFAO* mpfao = context;
  DM dm;
  ierr = TDyDiscretizationGetDM(discretization,&dm); CHKERRQ(ierr);

  ierr = TDyMeshCreateFromPlex(dm, &mpfao->V, &mpfao->X, &mpfao->N, &mpfao->mesh);
  ierr = TDyMeshGetMaxVertexConnectivity(mpfao->mesh, &mpfao->ncv, &mpfao->nfv);

  ierr = TDyAllocate_RealArray_1D(&(mpfao->vel), mpfao->mesh->num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&(mpfao->vel_count), mpfao->mesh->num_faces); CHKERRQ(ierr);

  ierr = InitMaterials(mpfao, dm, matprop, cc); CHKERRQ(ierr);

  // Gather mesh data.
  PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
  PetscInt nrow, ncol, nz;

  nFaces = mpfao->mesh->num_faces;
  nLocalCells = mpfao->mesh->num_cells_local;
  nNonLocalFaces = TDyMeshGetNumberOfNonLocalFaces(mpfao->mesh);
  nNonInternalFaces = TDyMeshGetNumberOfNonInternalFaces(mpfao->mesh);

  nrow = 4*nFaces;
  ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
  nz   = mpfao->nfv;
  ierr = TDyAllocate_RealArray_3D(&mpfao->Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv + mpfao->ncv); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Trans_mat); CHKERRQ(ierr);
  ierr = MatSetOption(mpfao->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->P_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesP_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->GravDisVec);
  ierr = VecZeroEntries(mpfao->GravDisVec);

  ierr = TDyAllocate_RealArray_3D(&mpfao->Temp_Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Temp_Trans_mat); CHKERRQ(ierr);
  ierr = MatSetOption(mpfao->Temp_Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->Temp_P_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->Temp_TtimesP_vec);

  PetscInt nsubcells = 8;
  ierr = TDyAllocate_RealArray_4D(&mpfao->subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, 3, 3); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_4D(&mpfao->Temp_subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, 3, 3); CHKERRQ(ierr);

  // Compute matrices for our discretization.
  ierr = ComputeGMatrix(mpfao, dm, matprop); CHKERRQ(ierr);
  ierr = ComputeTransmissibilityMatrix(mpfao, dm); CHKERRQ(ierr);
  ierr = ComputeGravityDiscretization(mpfao, dm, matprop); CHKERRQ(ierr);

  ierr = AllocateMemoryForBoundaryValues(mpfao, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForSourceSinkValues(mpfao); CHKERRQ(ierr);
  ierr = AllocateMemoryForEnergyBoundaryValues(mpfao, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForEnergySourceSinkValues(mpfao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup function for SALINITY + MPFA-O
PetscErrorCode TDySetup_Salinity_MPFAO(void *context,
                                       TDyDiscretizationType *discretization,
                                       EOS *eos,
                                       MaterialProp *matprop,
                                       CharacteristicCurves *cc,
                                       Conditions* conditions) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  TDyMPFAO* mpfao = context;
  DM dm;
  ierr = TDyDiscretizationGetDM(discretization,&dm); CHKERRQ(ierr);

  ierr = TDyMeshCreateFromPlex(dm, &mpfao->V, &mpfao->X, &mpfao->N, &mpfao->mesh);
  ierr = TDyMeshGetMaxVertexConnectivity(mpfao->mesh, &mpfao->ncv, &mpfao->nfv);

  ierr = TDyAllocate_RealArray_1D(&(mpfao->vel), mpfao->mesh->num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&(mpfao->vel_count), mpfao->mesh->num_faces); CHKERRQ(ierr);

  ierr = InitMaterials(mpfao, dm, matprop, cc); CHKERRQ(ierr);

  // Gather mesh data.
  PetscInt nLocalCells, nFaces, nNonLocalFaces, nNonInternalFaces;
  PetscInt nrow, ncol, nz;

  nFaces = mpfao->mesh->num_faces;
  nLocalCells = mpfao->mesh->num_cells_local;
  nNonLocalFaces = TDyMeshGetNumberOfNonLocalFaces(mpfao->mesh);
  nNonInternalFaces = TDyMeshGetNumberOfNonInternalFaces(mpfao->mesh);

  nrow = 4*nFaces;
  ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;
  nz   = mpfao->nfv;
  ierr = TDyAllocate_RealArray_3D(&mpfao->Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv + mpfao->ncv); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Trans_mat); CHKERRQ(ierr);
  ierr = MatSetOption(mpfao->Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->P_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesP_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->GravDisVec);
  ierr = VecZeroEntries(mpfao->GravDisVec);

  ierr = TDyAllocate_RealArray_3D(&mpfao->Psi_Trans, mpfao->mesh->num_vertices,
                                  mpfao->nfv, mpfao->nfv); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&mpfao->Psi_Trans_mat); CHKERRQ(ierr);
  ierr = MatSetOption(mpfao->Psi_Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&mpfao->Psi_vec);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&mpfao->TtimesPsi_vec);

  PetscInt nsubcells = 8;
  ierr = TDyAllocate_RealArray_4D(&mpfao->subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, 3, 3); CHKERRQ(ierr);
  ierr = TDyAllocate_RealArray_4D(&mpfao->Psi_subc_Gmatrix, mpfao->mesh->num_cells,
                                  nsubcells, 3, 3); CHKERRQ(ierr);

  // Compute matrices for our discretization.
  ierr = ComputeGMatrix(mpfao, dm, matprop); CHKERRQ(ierr);
  ierr = ComputeTransmissibilityMatrix(mpfao, dm); CHKERRQ(ierr);
  ierr = ComputeGravityDiscretization(mpfao, dm, matprop); CHKERRQ(ierr);

  ierr = AllocateMemoryForBoundaryValues(mpfao, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForSourceSinkValues(mpfao); CHKERRQ(ierr);
  ierr = AllocateMemoryForEnergyBoundaryValues(mpfao, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForEnergySourceSinkValues(mpfao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

//-----------------------
// UpdateState functions
//-----------------------

PetscErrorCode TDyUpdateState_Richards_MPFAO(void *context, DM dm,
                                             EOS *eos,
                                             MaterialProp *matprop,
                                             CharacteristicCurves *cc,
                                             PetscInt num_cells,
                                             PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyMPFAO *mpfao = context;

  PetscInt dim = 3;
  PetscInt dim2 = dim*dim;
  PetscInt cStart = 0, cEnd = num_cells;
  PetscInt nc = cEnd - cStart;

  // Compute the capillary pressure on all cells.
  PetscReal Pc[nc];
  for (PetscInt c=0;c<cEnd-cStart;c++) {
    Pc[c] = mpfao->Pref - U[c];
  }

  // Compute the saturation and its derivatives.
  ierr = SaturationCompute(cc->saturation, mpfao->Sr, Pc, mpfao->S, mpfao->dS_dP,
                           mpfao->d2S_dP2);

  // Compute the effective saturation on cells.
  PetscReal Se[nc];
  for (PetscInt c=0;c<nc;c++) {
    Se[c] = (mpfao->S[c] - mpfao->Sr[c])/(1.0 - mpfao->Sr[c]);
  }

  // Compute the relative permeability and its derivative (w.r.t. Se).
  ierr = RelativePermeabilityCompute(cc->rel_perm, Se, mpfao->Kr, mpfao->dKr_dS);

  // Correct dKr/dS using the chain rule, and update the permeability.
  for (PetscInt c=0;c<nc;c++) {
    PetscReal dSe_dS = 1.0/(1.0 - mpfao->Sr[c]);
    mpfao->dKr_dS[c] *= dSe_dS; // correct dKr/dS

    for(PetscInt j=0; j<dim2; j++) {
      mpfao->K[c*dim2+j] = mpfao->K0[c*dim2+j] * mpfao->Kr[c];
    }

    // Also update water properties.
    PetscReal P = mpfao->Pref - Pc[c]; // pressure
    ierr = EOSComputeWaterDensity(eos,
      P, mpfao->Tref, mpfao->m_nacl[c],
      &(mpfao->rho[c]), &(mpfao->drho_dP[c]),
      &(mpfao->d2rho_dP2[c])); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos,
      P, mpfao->Tref, mpfao->m_nacl[c],
      &(mpfao->vis[c]), &(mpfao->dvis_dP[c]),
      &(mpfao->d2vis_dP2[c])); CHKERRQ(ierr);
  }

  PetscReal *p_vec_ptr, gz;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;

  ierr = VecGetArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=0; c<nc; ++c) {
    ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[c].X,dim,&gz);
    PetscReal P = mpfao->Pref - Pc[c]; // pressure
    p_vec_ptr[c] = P;
  }
  ierr = VecRestoreArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateState_TH_MPFAO(void *context, DM dm,
                                       EOS *eos, MaterialProp *matprop,
                                       CharacteristicCurves *cc,
                                       PetscInt num_cells, PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  TDyMPFAO *mpfao = context;

  PetscInt dim = 3;
  PetscInt dim2 = dim*dim;
  PetscInt cStart = 0, cEnd = num_cells;
  PetscInt nc = cEnd - cStart;

  // Obtain the capillary pressure and the temperature on all cells.
  PetscReal Pc[nc], temp[nc];
  for (PetscInt c=0;c<cEnd-cStart;c++) {
    Pc[c] = mpfao->Pref - U[2*c];
    temp[c] = U[2*c+1];
  }

  // Compute the saturation and its derivatives.
  ierr = SaturationCompute(cc->saturation, mpfao->Sr, Pc, mpfao->S, mpfao->dS_dP,
                           mpfao->d2S_dP2);

  // Compute the effective saturation on cells.
  PetscReal Se[nc];
  for (PetscInt c=0;c<nc;c++) {
    Se[c] = (mpfao->S[c] - mpfao->Sr[c])/(1.0 - mpfao->Sr[c]);
  }

  // Compute the relative permeability and its derivative (w.r.t. Se).
  ierr = RelativePermeabilityCompute(cc->rel_perm, Se, mpfao->Kr, mpfao->dKr_dS);

  // Correct dKr/dS using the chain rule, and update the permeability.
  for (PetscInt c=0;c<nc;c++) {
    PetscReal dSe_dS = 1.0/(1.0 - mpfao->Sr[c]);
    mpfao->dKr_dS[c] *= dSe_dS; // correct dKr/dS

    for(PetscInt j=0; j<dim2; j++) {
      mpfao->K[c*dim2+j] = mpfao->K0[c*dim2+j] * mpfao->Kr[c];
    }

    // Also update water properties.
    PetscReal P = mpfao->Pref - Pc[c]; // pressure
    ierr = EOSComputeWaterDensity(eos,
      P, mpfao->Tref, mpfao->m_nacl[c],
      &(mpfao->rho[c]), &(mpfao->drho_dP[c]),
      &(mpfao->d2rho_dP2[c])); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos,
      P, mpfao->Tref, mpfao->m_nacl[c],
      &(mpfao->vis[c]), &(mpfao->dvis_dP[c]),
      &(mpfao->d2vis_dP2[c])); CHKERRQ(ierr);

    // Update the thermal conductivity based on Kersten number, etc.
    for(PetscInt j=0; j<dim2; ++j)
      mpfao->Kappa[c*dim2+j] = mpfao->Kappa0[c*dim2+j];
    ierr = EOSComputeWaterEnthalpy(eos, temp[c], P, &(mpfao->h[c]),
                                   &(mpfao->dh_dP[c]),
                                   &(mpfao->dh_dT[c])); CHKERRQ(ierr);
    mpfao->u[c] = mpfao->h[c] - P/mpfao->rho[c];
  }

  PetscReal *p_vec_ptr, gz;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;

  ierr = VecGetArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=0; c<nc; c++) {
    ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[c].X,dim,&gz);
    PetscReal P = mpfao->Pref - Pc[c]; // pressure
    p_vec_ptr[c] = P;
  }
  ierr = VecRestoreArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscReal *t_vec_ptr;
  ierr = VecGetArray(mpfao->Temp_P_vec, &t_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=0; c<nc; c++) {
    t_vec_ptr[c] = temp[c];
  }
  ierr = VecRestoreArray(mpfao->Temp_P_vec, &t_vec_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateState_Salinity_MPFAO(void *context, DM dm,
                                             EOS *eos, MaterialProp *matprop,
                                             CharacteristicCurves *cc,
                                             PetscInt num_cells, PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  TDyMPFAO *mpfao = context;

  PetscInt dim = 3;
  PetscInt dim2 = dim*dim;
  PetscInt cStart = 0, cEnd = num_cells;
  PetscInt nc = cEnd - cStart;

  // Obtain the capillary pressure and the salinity concentration on all cells.
  PetscReal Pc[nc], Psi[nc];
  for (PetscInt c=0;c<cEnd-cStart;c++) {
    Pc[c] = mpfao->Pref - U[2*c];
    Psi[c] = U[2*c+1];
  }

  // Compute the saturation and its derivatives.
  ierr = SaturationCompute(cc->saturation, mpfao->Sr, Pc, mpfao->S, mpfao->dS_dP,
                           mpfao->d2S_dP2);

  // Compute the effective saturation on cells.
  PetscReal Se[nc];
  for (PetscInt c=0;c<nc;c++) {
    Se[c] = (mpfao->S[c] - mpfao->Sr[c])/(1.0 - mpfao->Sr[c]);
  }

  // Compute the relative permeability and its derivative (w.r.t. Se).
  ierr = RelativePermeabilityCompute(cc->rel_perm, Se, mpfao->Kr, mpfao->dKr_dS);

  // Correct dKr/dS using the chain rule, and update the permeability.
  for (PetscInt c=0;c<nc;c++) {
    PetscReal dSe_dS = 1.0/(1.0 - mpfao->Sr[c]);
    mpfao->dKr_dS[c] *= dSe_dS; // correct dKr/dS

    for(PetscInt j=0; j<dim2; j++) {
      mpfao->K[c*dim2+j] = mpfao->K0[c*dim2+j] * mpfao->Kr[c];
    }

    // Also update water properties.
    PetscReal P = mpfao->Pref - Pc[c]; // pressure
    ierr = EOSComputeSalinityFraction(eos,
      Psi[c], mpfao->mu_saline[c], mpfao->rho[c],
      &(mpfao->m_nacl[c])); CHKERRQ(ierr);
    ierr = EOSComputeWaterDensity(eos,
      P, mpfao->Tref, mpfao->m_nacl[c],
      &(mpfao->rho[c]), &(mpfao->drho_dP[c]),
      &(mpfao->d2rho_dP2[c])); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos,
      P, mpfao->Tref, mpfao->m_nacl[c],
      &(mpfao->vis[c]), &(mpfao->dvis_dP[c]),
      &(mpfao->d2vis_dP2[c])); CHKERRQ(ierr);
  }

  PetscReal *p_vec_ptr, gz;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;

  ierr = VecGetArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=0; c<nc; c++) {
    ierr = ComputeGtimesZ(mpfao->gravity,cells->centroid[c].X,dim,&gz);
    PetscReal P = mpfao->Pref - Pc[c]; // pressure
    p_vec_ptr[c] = P;
  }
  ierr = VecRestoreArray(mpfao->P_vec,&p_vec_ptr); CHKERRQ(ierr);

  PetscReal *psi_vec_ptr;
  ierr = VecGetArray(mpfao->Psi_vec, &psi_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=0; c<nc; c++) {
    psi_vec_ptr[c] = Psi[c];
  }
  ierr = VecRestoreArray(mpfao->Psi_vec, &psi_vec_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateDiagnostics_MPFAO(void *context,
                                          DM diags_dm,
                                          Vec diags_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  TDyMPFAO *mpfao = context;
  TDyMesh  *mesh = mpfao->mesh;
  TDyCell  *cells = &mesh->cells;

  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(diags_dm,0,&c_start,&c_end); CHKERRQ(ierr);
  PetscReal *v;
  VecGetArray(diags_vec, &v);
  PetscInt count = 0;
  for (PetscInt c = c_start; c < c_end; ++c) {

    if (!cells->is_local[c]) continue;

    v[2*count + DIAG_LIQUID_SATURATION] = mpfao->S[c];
    v[2*count + DIAG_LIQUID_MASS] = mpfao->rho[c] * mpfao->V[c];

    count++;
  }
  VecRestoreArray(diags_vec, &v);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyMPFAOComputeSystem(TDy tdy,Mat K,Vec F) {

  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
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

  Conditions *conditions = tdy->conditions;
  if (ConditionsHasForcing(conditions)) {
    for (PetscInt icell=0; icell<mesh->num_cells; icell++) {
      if (cells->is_local[icell]) {
        PetscReal f;
        ierr = ConditionsComputeForcing(conditions, 1, &(mpfao->X[icell*dim]), &f);CHKERRQ(ierr);
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

