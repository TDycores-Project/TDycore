#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdysaturationimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfao3Dutilsimpl.h>
#include <private/tdympfao3Dtsimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOTransientVariable_3DMesh(TS ts, Vec U, Vec C, void *ctx) {

  TDy            tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM             dm;
  Vec            Ul;
  PetscScalar    *c,*p;
  PetscInt       icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  mesh     = tdy->mesh;
  cells    = &mesh->cells;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,Ul); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);

  ierr = VecGetArray(C,&c); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    c[icell] = tdy->rho[icell] * tdy->matprop_porosity[icell] * tdy->S[icell]* cells->volume[icell];
  }

  ierr = VecRestoreArray(C,&c); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_TransientVariable_3DMesh(TS ts,PetscReal t,Vec U,Vec M_t,Vec R,void *ctx) {

  TDy            tdy = (TDy)ctx;
  TDy_mesh       *mesh;
  TDy_cell       *cells;
  DM       dm;
  Vec      Ul;
  PetscReal      *p,*dm_dt,*r;
  PetscInt       icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

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
  ierr = MatMult(tdy->Trans_mat, tdy->P_vec, tdy->TtimesP_vec);

  ierr = TDyMPFAOIFunction_Vertices_3DMesh(Ul,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(M_t,&dm_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(m)/dt * Vol
    r[icell] += dm_dt[icell] * cells->volume[icell];
    r[icell] -= tdy->source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(M_t,&dm_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
