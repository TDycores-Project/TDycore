#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdymeshutilsimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdympfao2Dimpl.h>
#include <private/tdympfao3Dcoreimpl.h>
#include <private/tdyeosimpl.h>
#include <petscblaslapack.h>


/* -------------------------------------------------------------------------- */
PetscErrorCode SetPermeabilityFromFunction(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  if (tdy->ops->computepermeability) {

    PetscReal *localK;
    PetscInt icell, ii, jj;
    TDyMesh *mesh = tdy->mesh;

    MaterialProp *matprop = tdy->matprop;

    // If permeability function is set, use it instead.
    // Will need to consolidate this code with code in tdypermeability.c
    ierr = PetscMalloc(9*sizeof(PetscReal),&localK); CHKERRQ(ierr);
    for (icell=0; icell<mesh->num_cells; icell++) {
      ierr = (*tdy->ops->computepermeability)(tdy, &(tdy->X[icell*dim]), localK, tdy->permeabilityctx);CHKERRQ(ierr);

      PetscInt count = 0;
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          matprop->K[icell*dim*dim + ii*dim + jj] = localK[count];
          matprop->K0[icell*dim*dim + ii*dim + jj] = localK[count];
          count++;
        }
      }
    }
    ierr = PetscFree(localK); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetPorosityFromFunction(TDy tdy) {

  PetscInt dim;
  PetscInt c,cStart,cEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);
  MaterialProp *matprop = tdy->matprop;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(matprop->porosity,sizeof(PetscReal)*(cEnd-cStart));CHKERRQ(ierr);

  for(c=cStart; c<cEnd; c++) {
    ierr = (*tdy->ops->computeporosity)(tdy, &(tdy->X[c*dim]), &(matprop->porosity[c]), tdy->porosityctx);CHKERRQ(ierr);
  }

  /*
  if (tdy->ops->computepermeability) {

    PetscReal *localK;
    PetscInt icell, ii, jj;
    TDyMesh *mesh = tdy->mesh;


    // If peremeability function is set, use it instead.
    // Will need to consolidate this code with code in tdypermeability.c
    ierr = PetscMalloc(9*sizeof(PetscReal),&localK); CHKERRQ(ierr);
    for (icell=0; icell<mesh->num_cells; icell++) {
      ierr = (*tdy->ops->computepermeability)(tdy, &(tdy->X[icell*dim]), localK, tdy->permeabilityctx);CHKERRQ(ierr);

      PetscInt count = 0;
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          matprop->K[icell*dim*dim + ii*dim + jj] = localK[count];
          matprop->K0[icell*dim*dim + ii*dim + jj] = localK[count];
          count++;
        }
      }
    }
    ierr = PetscFree(localK); CHKERRQ(ierr);
  }
  */

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode SetThermalConductivityFromFunction(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  if (tdy->ops->computethermalconductivity) {

    PetscReal *localKappa;
    PetscInt icell, ii, jj;
    TDyMesh *mesh = tdy->mesh;
    MaterialProp *matprop = tdy->matprop;


    ierr = PetscMalloc(9*sizeof(PetscReal),&localKappa); CHKERRQ(ierr);
    for (icell=0; icell<mesh->num_cells; icell++) {
      ierr = (*tdy->ops->computethermalconductivity)(tdy, &(tdy->X[icell*dim]), localKappa, tdy->thermalconductivityctx);CHKERRQ(ierr);

      PetscInt count = 0;
      for (ii=0; ii<dim; ii++) {
        for (jj=0; jj<dim; jj++) {
          matprop->Kappa[icell*dim*dim + ii*dim + jj] = localKappa[count];
          matprop->Kappa0[icell*dim*dim + ii*dim + jj] = localKappa[count];
          count++;
        }
      }
    }
    ierr = PetscFree(localKappa); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeGMatrix(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode ComputeTransmissibilityMatrix(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
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
    ierr = ComputeWaterDensity(tdy->Pref, tdy->rho_type, &(tdy->rho_BND[i]), &dden_dP, &d2den_dP2); CHKERRQ(ierr);
    ierr = ComputeWaterViscosity(tdy->Pref, tdy->mu_type, &(tdy->vis_BND[i]), &dmu_dP, &d2mu_dP2); CHKERRQ(ierr);
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
    ierr = ComputeWaterEnthalpy(tdy->Tref, tdy->Pref,tdy->enthalpy_type, &(tdy->h_BND[i]), &dh_dP, &dh_dT); CHKERRQ(ierr);
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

/* -------------------------------------------------------------------------- */

PetscErrorCode TDyMPFAOInitialize(TDy tdy) {

  PetscErrorCode ierr;
  MPI_Comm       comm;
  DM             dm = tdy->dm;
  PetscInt       dim;
  PetscInt       nrow,ncol,nsubcells;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  tdy->mesh = (TDyMesh *) malloc(sizeof(TDyMesh));

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
    ierr = TDyAllocate_RealArray_3D(&tdy->Trans, tdy->mesh->num_vertices, tdy->nfv, tdy->nfv + tdy->ncv); CHKERRQ(ierr);
    if (tdy->mode == TH){ierr = TDyAllocate_RealArray_3D(&tdy->Temp_Trans, 
                         tdy->mesh->num_vertices, tdy->nfv, tdy->nfv); CHKERRQ(ierr);}
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
  if (tdy->mode == TH) {ierr = TDyAllocate_RealArray_4D(&tdy->Temp_subc_Gmatrix, tdy->mesh->num_cells,
                               nsubcells, nrow, ncol); CHKERRQ(ierr);}

  /* Setup the section, 1 dof per cell */
  PetscSection sec;
  PetscInt p, pStart, pEnd;
  PetscBool use_dae;

  use_dae = (tdy->method == MPFA_O_DAE);
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  if (!use_dae) {
    if (tdy->mode == TH){
      ierr = PetscSectionSetNumFields(sec, 2); CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
      ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(sec, 1, "LiquidTemperature"); CHKERRQ(ierr);
      ierr = PetscSectionSetFieldComponents(sec, 1, 1); CHKERRQ(ierr);
    } else {
      ierr = PetscSectionSetNumFields(sec, 1); CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
      ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);
    }
  } else {
    if (tdy->mode == TH) {SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"TH unsupported with MPFA_O_DAE");}
    ierr = PetscSectionSetNumFields(sec, 2); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(sec, 0, "LiquidPressure"); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(sec, 1, "LiquidMass"); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(sec, 1, 1); CHKERRQ(ierr);
  }

  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
    if (tdy->mode == TH){
      ierr = PetscSectionSetFieldDof(sec,p,1,1); CHKERRQ(ierr);
      ierr = PetscSectionSetDof(sec,p,2); CHKERRQ(ierr);
    }
    if (use_dae) {
      if (tdy->mode == TH) {SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"TH unsupported with MPFA_O_DAE");}
      ierr = PetscSectionSetFieldDof(sec,p,1,1); CHKERRQ(ierr);
      ierr = PetscSectionSetDof(sec,p,2); CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  ierr = TDyBuildMesh(tdy); CHKERRQ(ierr);

  if (dim == 3) {
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

    if (tdy->mode == TH){
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nrow,ncol,nz,NULL,&tdy->Temp_Trans_mat); CHKERRQ(ierr);
      ierr = MatSetOption(tdy->Temp_Trans_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&tdy->Temp_P_vec);
      ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&tdy->Temp_TtimesP_vec);
    }
  }

  if (tdy->ops->computeporosity) { ierr = SetPorosityFromFunction(tdy); CHKERRQ(ierr); }
  if (tdy->ops->computepermeability) {ierr = SetPermeabilityFromFunction(tdy); CHKERRQ(ierr);}
  if (tdy->mode == TH){
    if (tdy->ops->computethermalconductivity) {ierr = SetThermalConductivityFromFunction(tdy); CHKERRQ(ierr);}
  }

  // why must these be placed after SetPermeabilityFromFunction()?
  ierr = ComputeGMatrix(tdy); CHKERRQ(ierr);
  ierr = ComputeTransmissibilityMatrix(tdy); CHKERRQ(ierr);

  if (dim == 3) {
    ierr = TDyMPFAO_AllocateMemoryForBoundaryValues(tdy); CHKERRQ(ierr);
    ierr = TDyMPFAO_AllocateMemoryForSourceSinkValues(tdy); CHKERRQ(ierr);
    if (tdy->mode == TH) {
      ierr = TDyMPFAO_AllocateMemoryForEnergyBoundaryValues(tdy); CHKERRQ(ierr);
      ierr = TDyMPFAO_AllocateMemoryForEnergySourceSinkValues(tdy); CHKERRQ(ierr);
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSetFromOptions(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);
  if (tdy->ops->computeporosity) { ierr = SetPorosityFromFunction(tdy); CHKERRQ(ierr); }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_InternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(TDy tdy,Mat K,Vec F) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOComputeSystem(TDy tdy,Mat K,Vec F) {

  DM             dm = tdy->dm;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscInt       icell;
  PetscInt       row;
  PetscReal      value;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  
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
      if (cells->is_local[icell]) {
        ierr = (*tdy->ops->computeforcing)(tdy, &(tdy->X[icell*dim]), &f, tdy->forcingctx);CHKERRQ(ierr);
        value = f * cells->volume[icell];
        row = cells->global_id[icell];
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
PetscErrorCode TDyMPFAORecoverVelocity(TDy tdy, Vec U) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
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
  PetscInt       dim;
  PetscInt       icell;
  PetscReal      norm, norm_sum, pressure;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  cells = &mesh->cells;

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

    if (!cells->is_local[icell]) continue;

    ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[icell*dim]), &pressure, tdy->dirichletvaluectx);CHKERRQ(ierr);
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
PetscReal TDyMPFAOVelocityNorm_3DMesh(TDy tdy) {

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

      ierr = (*tdy->ops->computedirichletflux)(tdy, &(tdy->X[(face_id + fStart)*dim]), vel, tdy->dirichletfluxctx);CHKERRQ(ierr);
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

/* -------------------------------------------------------------------------- */
PetscReal TDyMPFAOVelocityNorm(TDy tdy) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(norm_sum);
}
