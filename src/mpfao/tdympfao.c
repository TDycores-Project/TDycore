#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdympfao2Dimpl.h>
#include <private/tdympfao3Dimpl.h>
#include <private/tdyeosimpl.h>
#include <petscblaslapack.h>


/* -------------------------------------------------------------------------- */
PetscErrorCode SetPermeabilityFromFunction(TDy tdy) {

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
          tdy->K0[icell*dim*dim + ii*dim + jj] = localK[count];
          count++;
        }
      }
    }
    ierr = PetscFree(localK); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetPorosityFromFunction(TDy tdy) {

  PetscInt dim;
  PetscInt c,cStart,cEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(tdy->porosity,sizeof(PetscReal)*(cEnd-cStart));CHKERRQ(ierr);

  for(c=cStart; c<cEnd; c++) {
    ierr = (*tdy->ops->computeporosity)(tdy, &(tdy->X[c*dim]), &(tdy->porosity[c]), tdy->porosityctx);CHKERRQ(ierr);
  }

  /*
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
          tdy->K0[icell*dim*dim + ii*dim + jj] = localK[count];
          count++;
        }
      }
    }
    ierr = PetscFree(localK); CHKERRQ(ierr);
  }
  */

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGMatrix(TDy tdy) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);
 
  switch (dim) {
  case 2:
    ierr = TDyComputeGMatrixFor2DMesh(tdy); CHKERRQ(ierr);
    break;
  case 3:
    ierr = TDyComputeGMatrixFor3DMesh(tdy); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
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
    ierr = TDyComputeTransmissibilityMatrix2DMesh(tdy); CHKERRQ(ierr);
    break;
  case 3:
    ierr = TDyComputeTransmissibilityMatrix3DMesh(tdy); CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAO_AllocateMemoryForBoundaryValues(TDy tdy) {

  TDy_mesh *mesh;
  PetscInt nbnd_faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh  = tdy->mesh;
  nbnd_faces = mesh->num_boundary_faces;

  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->P_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->rho_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->vis_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->Kr_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->dKr_dS_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->S_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->dS_dP_BND)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(tdy->d2S_dP2_BND)); CHKERRQ(ierr);

  PetscInt i;
  PetscReal dden_dP, d2den_dP2, dmu_dP, d2mu_dP2;
  for (i=0;i<nbnd_faces;i++) {
    ierr = ComputeWaterDensity(tdy->Pref, tdy->rho_type, &(tdy->rho_BND[i]), &dden_dP, &d2den_dP2); CHKERRQ(ierr);
    ierr = ComputeWaterViscosity(tdy->Pref, tdy->mu_type, &(tdy->vis_BND[i]), &dmu_dP, &d2mu_dP2); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOInitialize(TDy tdy) {

  PetscErrorCode ierr;
  MPI_Comm       comm;
  DM             dm;
  PetscInt       dim;
  PetscInt       nrow,ncol,nsubcells;

  PetscFunctionBegin;

  dm = tdy->dm;

  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  tdy->mesh = (TDy_mesh *) malloc(sizeof(TDy_mesh));

  ierr = TDyAllocateMemoryForMesh(tdy); CHKERRQ(ierr);

  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  switch (dim) {
  case 2:
    ierr = TDyAllocate_RealArray_3D(&tdy->Trans, tdy->mesh->num_vertices, 5, 5);
    CHKERRQ(ierr);
    ierr = PetscMalloc(tdy->mesh->num_edges*sizeof(PetscReal),
                     &(tdy->vel )); CHKERRQ(ierr);
    ierr = TDyInitialize_RealArray_1D(tdy->vel, tdy->mesh->num_edges, 0.0); CHKERRQ(ierr);
    ierr = PetscMalloc(tdy->mesh->num_edges*sizeof(PetscInt),
                     &(tdy->vel_count)); CHKERRQ(ierr);
    ierr = TDyInitialize_IntegerArray_1D(tdy->vel_count, tdy->mesh->num_edges, 0); CHKERRQ(ierr);

    nsubcells = 4;
    nrow = 2;
    ncol = 2;

    break;
  case 3:
    ierr = TDyAllocate_RealArray_3D(&tdy->Trans, tdy->mesh->num_vertices, 12, 12);
    CHKERRQ(ierr);
    ierr = PetscMalloc(tdy->mesh->num_faces*sizeof(PetscReal),
                     &(tdy->vel )); CHKERRQ(ierr);
    ierr = TDyInitialize_RealArray_1D(tdy->vel, tdy->mesh->num_faces, 0.0); CHKERRQ(ierr);
    ierr = PetscMalloc(tdy->mesh->num_faces*sizeof(PetscInt),
                     &(tdy->vel_count)); CHKERRQ(ierr);
    ierr = TDyInitialize_IntegerArray_1D(tdy->vel_count, tdy->mesh->num_faces, 0); CHKERRQ(ierr);

    nsubcells = 8;
    nrow = 3;
    ncol = 3;

    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in TDyMPFAOInitialize");
    break;
  }

  ierr = TDyAllocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells,
                               nsubcells, nrow, ncol); CHKERRQ(ierr);

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

  ierr = TDyBuildMesh(tdy); CHKERRQ(ierr);

  if (tdy->ops->computepermeability) { ierr = SetPermeabilityFromFunction(tdy); CHKERRQ(ierr);}

  ierr = ComputeGMatrix(tdy); CHKERRQ(ierr);

  ierr = ComputeTransmissibilityMatrix(tdy); CHKERRQ(ierr);

  if (dim == 3) {
    ierr = TDyMPFAO_AllocateMemoryForBoundaryValues(tdy); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSetFromOptions(TDy tdy) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);
  if (tdy->ops->computeporosity) { ierr = SetPorosityFromFunction(tdy); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  ierr = TDyInitialize_IntegerArray_1D(tdy->vel_count, tdy->mesh->num_faces, 0); CHKERRQ(ierr);

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
PetscReal TDyMPFAOVelocityNorm_3DMesh(TDy tdy) {

  DM             dm;
  TDy_mesh       *mesh;
  TDy_face       *faces, *face;
  TDy_cell       *cells, *cell;
  PetscInt       dim;
  PetscInt       icell, iface, face_id;
  PetscInt       fStart, fEnd;
  PetscReal      norm, norm_sum, vel_normal;
  PetscReal      vel[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm    = tdy->dm;
  mesh  = tdy->mesh;
  cells = mesh->cells;
  faces = mesh->faces;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  norm_sum = 0.0;
  norm     = 0.0;

  for (icell=0; icell<mesh->num_cells; icell++) {

    cell = &(cells[icell]);

    if (!cell->is_local) continue;

    for (iface=0; iface<cell->num_faces; iface++) {
      face_id = cell->face_ids[iface];
      face    = &(faces[face_id]);

      ierr = (*tdy->ops->computedirichletflux)(tdy, &(tdy->X[(face_id + fStart)*dim]), vel, tdy->dirichletfluxctx);CHKERRQ(ierr);
      vel_normal = TDyADotB(vel,&(face->normal.V[0]),dim);
      if (tdy->vel_count[face_id] != 4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"tdy->vel_count != 4");

      norm += PetscSqr((vel_normal - tdy->vel[face_id]))*cell->volume;
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
    norm_sum = TDyMPFAOVelocityNorm_3DMesh(tdy);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in ComputeGMatrix");
    break;
  }

  PetscFunctionReturn(norm_sum);
}
