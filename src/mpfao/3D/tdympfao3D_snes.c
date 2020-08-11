#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdysaturationimpl.h>
#include <private/tdypermeabilityimpl.h>
#include <private/tdympfao3Dutilsimpl.h>
#include <private/tdympfao3Dtsimpl.h>

#include <petscviewerhdf5.h>

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
  ierr = MatMult(tdy->Trans_mat, tdy->P_vec, tdy->TtimesP_vec);

  PetscReal *accum_prev;
  
  ierr = TDyMPFAOIFunction_Vertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);

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

#if 0
  ierr = VecCopy(R,tdy->F); CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"F.vec",&viewer);
         CHKERRQ(ierr);
  ierr = VecView(R,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

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

  ierr = TDyMPFAOIJacobian_Vertices_3DMesh(Ul, A, ctx);

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

#if 0
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"A.mat",&viewer);
//  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"A.h5",FILE_MODE_WRITE,&viewer);
         CHKERRQ(ierr);
  ierr = MatView(A,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = MatCopy(A,tdy->G,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"G.mat",&viewer);
//  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"G.h5",FILE_MODE_WRITE,&viewer);
         CHKERRQ(ierr);
  ierr = MatView(tdy->G,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"F.vec",&viewer);
//  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"F.h5",FILE_MODE_WRITE,&viewer);
         CHKERRQ(ierr);
  ierr = VecView(tdy->F,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,tdy->G,tdy->G); CHKERRQ(ierr);
  PC pc;
  ierr = KSPSetType(ksp,KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetZeroPivot(pc,1.e-20); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,tdy->F,tdy->update); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"U.vec",&viewer);
         CHKERRQ(ierr);
  ierr = VecView(tdy->update,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

#if 0
  Mat AA;
//  ierr = MatDuplicate(A,MAT_COPY_VALUES,&AA); CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,&AA); CHKERRQ(ierr);
  ierr = MatCopy(A,AA,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  IS is1, is2;
  PetscInt iarray[8];
  MatFactorInfo info;
  for (int i=0; i<8; i++) iarray[i] = i;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,8,iarray,PETSC_COPY_VALUES,&is1);
         CHKERRQ(ierr);
  ierr = ISSetPermutation(is1); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,8,iarray,PETSC_COPY_VALUES,&is2);
         CHKERRQ(ierr);
  ierr = ISSetPermutation(is2); CHKERRQ(ierr);
  ierr = MatLUFactor(AA,is1,is2,&info); CHKERRQ(ierr); CHKERRQ(ierr);
  ierr = MatLUFactor(A,is1,is2,&info); CHKERRQ(ierr); CHKERRQ(ierr);
#endif

#if 0
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"factor.mat",&viewer);
         CHKERRQ(ierr);
  ierr = MatView(AA,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif
#endif

  PetscFunctionReturn(0);
}


