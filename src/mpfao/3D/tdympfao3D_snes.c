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


