#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdydiscretizationimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfaotsimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOTransientVariable(TS ts, Vec U, Vec C, void *ctx) {

  TDy            tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  PetscScalar    *c,*p;
  PetscInt       icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = TDyGlobalToLocal(&tdy->tdydm,U,tdy->soln_loc); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(tdy->soln_loc,&p); CHKERRQ(ierr);
ierr = TDyUpdateState(tdy, p, mesh->num_cells); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyMPFAOUpdateBoundaryState(tdy); CHKERRQ(ierr);

  ierr = VecGetArray(C,&c); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    c[icell] = mpfao->rho[icell] * mpfao->porosity[icell] * mpfao->S[icell]* cells->volume[icell];
  }

  ierr = VecRestoreArray(C,&c); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_TransientVariable(TS ts,PetscReal t,Vec U,Vec M_t,Vec R,void *ctx) {

  TDy            tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh        *mesh = mpfao->mesh;
  TDyCell        *cells = &mesh->cells;
  DM             dm;
  PetscReal      *p,*dm_dt,*r;
  PetscInt       icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = TDyGlobalToLocal(&tdy->tdydm,U,tdy->soln_loc); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(tdy->soln_loc,&p); CHKERRQ(ierr);
ierr = TDyUpdateState(tdy, p, mesh->num_cells); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_loc,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyMPFAOUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(mpfao->Trans_mat, mpfao->P_vec, mpfao->TtimesP_vec);

  ierr = TDyMPFAOIFunction_Vertices(tdy->soln_loc,R,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(M_t,&dm_dt); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(m)/dt * Vol
    r[icell] += dm_dt[icell] * cells->volume[icell];
    r[icell] -= mpfao->source_sink[icell] * cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(M_t,&dm_dt); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
