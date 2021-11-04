#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdydiscretization.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfaotsimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOTransientVariable(TS ts, Vec U, Vec C, void *ctx) {

  TDy            tdy = (TDy)ctx;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  DM             dm;
  PetscScalar    *c,*p;
  PetscInt       icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

    CharacteristicCurve *cc = tdy->cc;
  MaterialProp *matprop = tdy->matprop;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = TDyGlobalToLocal(tdy,U,tdy->soln_loc); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(tdy->soln_loc,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_loc,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);

  ierr = VecGetArray(C,&c); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    c[icell] = tdy->rho[icell] * matprop->porosity[icell] * cc->S[icell]* cells->volume[icell];
  }

  ierr = VecRestoreArray(C,&c); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_TransientVariable(TS ts,PetscReal t,Vec U,Vec M_t,Vec R,void *ctx) {

  TDy            tdy = (TDy)ctx;
  TDyMesh        *mesh = tdy->mesh;
  TDyCell        *cells = &mesh->cells;
  DM             dm;
  PetscReal      *p,*dm_dt,*r;
  PetscInt       icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = TDyGlobalToLocal(tdy,U,tdy->soln_loc); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(tdy->soln_loc,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_loc,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(tdy->Trans_mat, tdy->P_vec, tdy->TtimesP_vec);

  ierr = TDyMPFAOIFunction_Vertices(tdy->soln_loc,R,ctx); CHKERRQ(ierr);

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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
