#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfaotsimpl.h>
#include <private/tdydiscretization.h>

//#define DEBUG
#if defined(DEBUG)
PetscInt icount_f = 0;
PetscInt icount_j = 0;
PetscInt max_count = 5;
#endif

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESAccumulation(TDy tdy, PetscInt icell, PetscReal *accum) {

  PetscFunctionBegin;

  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;

  *accum = mpfao->rho[icell] * mpfao->porosity[icell] * mpfao->S[icell] * cells->volume[icell] / tdy->dtime;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESPreSolve(TDy tdy) {

  TDyMPFAO *mpfao = tdy->context;
  TDyMesh *mesh = mpfao->mesh;
  TDyCell *cells = &mesh->cells;
  PetscReal *p, *accum_prev;
  PetscInt icell;
  PetscErrorCode ierr;

  TDY_START_FUNCTION_TIMER()


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

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESFunction(SNES snes,Vec U,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  TDyMesh  *mesh = mpfao->mesh;
  TDyCell  *cells = &mesh->cells;
  DM       dm = tdy->dm;
  Vec      Ul;
  PetscReal *p,*r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()


#if defined(DEBUG)
  PetscViewer viewer;
  char word[32];
  sprintf(word,"U%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = VecView(U,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  //ierr = SNESGetDM(snes,&dm); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = TDyGlobalToLocal(tdy,U,Ul); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(Ul,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(mpfao,Ul); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(mpfao); CHKERRQ(ierr);
  ierr = MatMult(mpfao->Trans_mat, mpfao->P_vec, mpfao->TtimesP_vec);

  PetscReal *accum_prev;

  ierr = TDyMPFAOIFunction_Vertices(Ul,R,ctx); CHKERRQ(ierr);

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
    r[icell] -= mpfao->source_sink[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

#if defined(DEBUG)
  sprintf(word,"Function%d.vec",icount_f);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = VecView(R,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  icount_f++;
#endif

  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOSNESJacobian(SNES snes,Vec U,Mat A,Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMPFAO *mpfao = tdy->context;
  DM             dm = tdy->dm;
  TDyMesh       *mesh = mpfao->mesh;
  TDyCell       *cells = &mesh->cells;
  Vec Ul, Udotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = MatZeroEntries(B); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Udotl); CHKERRQ(ierr);

  ierr = TDyGlobalToLocal(tdy,U,Ul); CHKERRQ(ierr);

  ierr = TDyMPFAOIJacobian_Vertices(Ul, B, ctx); CHKERRQ(ierr);

  PetscReal dtInv = 1.0/tdy->dtime;

  PetscReal dporosity_dP = 0.0;
  PetscReal dmass_dP, Jac;
  PetscInt icell;

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d/dP ( d(rho*phi*s)/dt * Vol )
    //  = d/dP [(rho*phi*s)^{t+1} - (rho*phi*s)^t]/dt * Vol
    //  = d/dP [(rho*phi*s)^{t+1}]
    dmass_dP = mpfao->rho[icell]     * dporosity_dP         * mpfao->S[icell] +
               mpfao->drho_dP[icell] * mpfao->porosity[icell] * mpfao->S[icell] +
               mpfao->rho[icell]     * mpfao->porosity[icell] * mpfao->dSdP[icell];
    Jac = dmass_dP * cells->volume[icell] * dtInv;

    ierr = MatSetValuesLocal(B,1,&icell,1,&icell,&Jac,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A !=B ) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Udotl); CHKERRQ(ierr);

#if defined(DEBUG)
  PetscViewer viewer;
  char word[32];
  sprintf(word,"Jacobian%d.mat",icount_j);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer); CHKERRQ(ierr);
  ierr = MatView(A,viewer);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  icount_j++;
  if (icount_j == max_count) exit(0);
#endif

  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}


