#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdympfao2Dimpl.h>
#include <private/tdympfao3Dimpl.h>
#include <petscblaslapack.h>


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

  ierr = TDyBuildMesh(tdy); CHKERRQ(ierr);

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


/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {
  
  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_face       *faces, *face;
  DM       dm;
  Vec      Ul;
  PetscReal *p,*dp_dt,*r;
  //PetscInt c,cStart,cEnd,nv,gref,nf,f,fStart,fEnd,i,j,dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  faces    = mesh->faces;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);

  PetscReal vel_error = 0.0;
  PetscInt count = 0;
  ierr = TDyMPFAORecoverVelocity_InternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);
  ierr = TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices_3DMesh(tdy, Ul, &vel_error, &count); CHKERRQ(ierr);

  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  PetscInt iface, cell_id_up, cell_id_dn;
  PetscReal ukvr, den, v_darcy, fluxm;

  for (iface=0; iface<mesh->num_faces; iface++) {
    face = &(faces[iface]);
    cell_id_up = face->cell_ids[0];
    cell_id_dn = face->cell_ids[1];

    if (fabs(tdy->vel[iface])<PETSC_MACHINE_EPSILON) tdy->vel[iface] = 0.0;

    if (face->is_internal) {

      if (tdy->vel[iface] > 0.0) {
        ukvr = tdy->Kr[cell_id_up]/tdy->mu[cell_id_up];
      } else {
        ukvr = tdy->Kr[cell_id_dn]/tdy->mu[cell_id_dn];
      }
      den = 0.5*(tdy->rho[cell_id_up] + tdy->rho[cell_id_dn]);
      
      v_darcy = ukvr * tdy->vel[iface];
      fluxm = v_darcy * den *face->area;

      if (cells[cell_id_up].is_local) r[cell_id_up] += fluxm;
      if (cells[cell_id_dn].is_local) r[cell_id_dn] -= fluxm;

    } else {

      if (cell_id_up>-1) {
        den = tdy->rho[cell_id_up];
        ukvr = tdy->Kr[cell_id_up]/tdy->mu[cell_id_up];
      } else {
        den = tdy->rho[cell_id_dn];
        ukvr = tdy->Kr[cell_id_dn]/tdy->mu[cell_id_dn];
      }

      v_darcy = ukvr * tdy->vel[iface];
      fluxm = v_darcy * den *face->area;

      if (cell_id_up>-1) r[cell_id_up] += fluxm;
      if (cell_id_dn>-1) r[cell_id_dn] -= fluxm;

    }
  }

  PetscReal dporosity_dP = 0.0;
  PetscReal drho_dP = 0.0;
  PetscReal dmass_dP;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    cell = &cells[icell];

    if (!cell->is_local) continue;

    // d(rho*phi*s)/dP * dP/dt * Vol
    dmass_dP = tdy->rho[icell] * dporosity_dP         * tdy->S[icell] +
               drho_dP         * tdy->porosity[icell] * tdy->S[icell] +
               tdy->rho[icell] * tdy->porosity[icell] * tdy->dS_dP[icell];
    r[icell] += dmass_dP * dp_dt[icell] * cell->volume;
  }

  /* Cleanup */
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIJacobian(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal shift,Mat A,Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells, *cell;
  TDy_face       *faces, *face;
  PetscReal *dp_dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mesh     = tdy->mesh;
  cells    = mesh->cells;
  faces    = mesh->faces;

  ierr = MatZeroEntries(A); CHKERRQ(ierr);

  PetscInt iface, cell_id_up, cell_id_dn;
  PetscReal ukvr, den, v_darcy, fluxm;
  PetscReal dukvr_dPup, dukvr_dPdn, Jup, Jdn;
  PetscReal dden_dPup, dden_dPdn;
  PetscReal dvel_dPup, dvel_dPdn;
  PetscReal dv_darcy_dPup, dv_darcy_dPdn;

  ierr = VecGetArray(U_t,&dp_dt); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {
    face = &(faces[iface]);
    cell_id_up = face->cell_ids[0];
    cell_id_dn = face->cell_ids[1];

    if (face->is_internal) {

      dukvr_dPup = 0.0;
      dukvr_dPdn = 0.0;

      if (tdy->vel[iface] > 0.0) {
        ukvr = tdy->Kr[cell_id_up]/tdy->mu[cell_id_up];
        dukvr_dPup = tdy->dKr_dS[cell_id_up]*tdy->dS_dP[cell_id_up]/tdy->mu[cell_id_up];
      } else {
        ukvr = tdy->Kr[cell_id_dn]/tdy->mu[cell_id_dn];
        dukvr_dPdn = tdy->dKr_dS[cell_id_dn]*tdy->dS_dP[cell_id_dn]/tdy->mu[cell_id_dn];
      }
      den = 0.5*(tdy->rho[cell_id_up] + tdy->rho[cell_id_dn]);
      dden_dPup = 0.0;
      dden_dPdn = 0.0;
      
      v_darcy = ukvr * tdy->vel[iface];
      fluxm = v_darcy * den *face->area;
      
      dvel_dPup =  1.e-10*3.0;
      dvel_dPdn = -1.e-10*3.0;

      dv_darcy_dPup =dvel_dPup*ukvr + dukvr_dPup*tdy->vel[iface];
      dv_darcy_dPdn =dvel_dPdn*ukvr + dukvr_dPdn*tdy->vel[iface];

      // d(fluxm_ij)/dP_k = d(v_darcy_ij*den_ij)/dP_k * area
      //                  = d(ukvr_ij*vel_ij*den_ij)/dP_k * area
      //                  = d(ukvr_ij)/dP_k * vel_ij         * den_ij         * area +
      //                    ukvr_ij         * d(vel_ij)/dP_k * den_ij         * area +
      //                    ukvr_ij         * vel_ij         * d(den_ij)/dP_k * area
      //
      // If 'k' is not 'i' or 'j' then
      //  - d(ukvr_ij)/dP_k = 0, and
      //  - d(den_ij)/dP_k  = 0.
      //
      Jup = (dukvr_dPup * tdy->vel[iface]       * den +
             ukvr       * dvel_dPup * den +
             ukvr       * tdy->vel[iface]       * dden_dPup) * face->area;

      Jdn = (dukvr_dPdn * tdy->vel[iface]       * den +
             ukvr       * dvel_dPdn * den +
             ukvr       * tdy->vel[iface]       * dden_dPdn) * face->area;

      if (cells[cell_id_up].is_local) {
        ierr = MatSetValuesLocal(A,1,&cell_id_up,1,&cell_id_up,&Jup,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&cell_id_up,1,&cell_id_dn,&Jdn,ADD_VALUES);CHKERRQ(ierr);
      }

      if (cells[cell_id_dn].is_local) {
        Jup = -Jup;
        Jdn = -Jdn;
        ierr = MatSetValuesLocal(A,1,&cell_id_dn,1,&cell_id_up,&Jup,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&cell_id_dn,1,&cell_id_dn,&Jdn,ADD_VALUES);CHKERRQ(ierr);
      }

    } else {

      dukvr_dPup = 0.0;
      dukvr_dPdn = 0.0;
      dvel_dPup = 0.0;
      dvel_dPdn = 0.0;

      if (tdy->vel[iface] >= 0.0) {
        // up ---> dn
        if (cell_id_up>-1) {
          //     up   --->    dn
          // (interal)     (boundary)
          ukvr = tdy->Kr[cell_id_up]/tdy->mu[cell_id_up];
          dukvr_dPup = tdy->dKr_dS[cell_id_up]*tdy->dS_dP[cell_id_up]/tdy->mu[cell_id_up];
          den = tdy->rho[cell_id_up];
        } else {
          //     up   --->    dn
          // (boundary)     (internal)
          ukvr = tdy->Kr[cell_id_dn]/tdy->mu[cell_id_dn];
          den = tdy->rho[cell_id_dn];
        }
      } else {
        // up <--- dn
        if (cell_id_up>-1) {
          //     up   <---    dn
          // (interal)     (boundary)
          ukvr = tdy->Kr[cell_id_up]/tdy->mu[cell_id_up];
          den = tdy->rho[cell_id_up];
        } else {
          //     up   <---    dn
          // (boundary)     (internal)
          ukvr = tdy->Kr[cell_id_dn]/tdy->mu[cell_id_dn];
          dukvr_dPdn = tdy->dKr_dS[cell_id_dn]*tdy->dS_dP[cell_id_dn]/tdy->mu[cell_id_dn];
          den = tdy->rho[cell_id_dn];
        }
      }

      dden_dPup = 0.0;
      dden_dPdn = 0.0;

      v_darcy = ukvr * tdy->vel[iface];
      fluxm = v_darcy * den *face->area;


      Jup = (dukvr_dPup * tdy->vel[iface]   * den +
             ukvr       * dvel_dPup * den +
             ukvr       * tdy->vel[iface]   * dden_dPup) * face->area;

      Jdn = (dukvr_dPdn * tdy->vel[iface]   * den +
             ukvr       * dvel_dPdn * den +
             ukvr       * tdy->vel[iface]   * dden_dPdn) * face->area;

      if (cell_id_up>-1) {
        ierr = MatSetValuesLocal(A,1,&cell_id_up,1,&cell_id_up,&Jup,ADD_VALUES);CHKERRQ(ierr);
      }
      if (cell_id_dn>-1) {
        Jdn = -Jdn;
        ierr = MatSetValuesLocal(A,1,&cell_id_dn,1,&cell_id_dn,&Jdn,ADD_VALUES);CHKERRQ(ierr);
      }
    }

  }

  PetscReal dporosity_dP = 0.0, d2porosity_dP2 = 0.0;
  PetscReal drho_dP = 0.0, d2rho_dP2=0.0;
  PetscReal dmass_dP, d2mass_dP2;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    cell = &cells[icell];

    if (!cell->is_local) continue;

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

    PetscReal value = (shift*dmass_dP + d2mass_dP2*dp_dt[icell])*cell->volume;

    ierr = MatSetValuesLocal(A,1,&icell,1,&icell,&value,ADD_VALUES);CHKERRQ(ierr);

  }

  ierr = VecRestoreArray(U_t,&dp_dt); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A !=B ) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
