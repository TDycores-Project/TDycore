#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdympfaoutilsimpl.h>

#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfao3Dutilsimpl.h>
#include <private/tdympfao3Dtsimpl.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyMPFAOIFunction_DAE_3DMesh(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyMesh       *mesh = tdy->mesh;
  TDyCell       *cells = &mesh->cells;
  DM       dm;
  Vec      Ul,P,M,R_P,R_M;
  PetscReal *p,*u_t,*r,*r_p,*m;
  PetscInt m_idx, p_idx;
  PetscInt icell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

    CharacteristicCurve *cc = tdy->cc;
  MaterialProp *matprop = tdy->matprop;

  ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul); CHKERRQ(ierr);

  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Get sub-vectors
  ierr = ExtractSubVectors(Ul,0,&P);
  ierr = ExtractSubVectors(Ul,1,&M);
  ierr = ExtractSubVectors(R,0,&R_P);
  ierr = ExtractSubVectors(R,1,&R_M);

  // Update the auxillary variables based on the current iterate
  ierr = VecGetArray(P,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p); CHKERRQ(ierr);
  ierr = VecRestoreArray(P,&p); CHKERRQ(ierr);

  ierr = TDyMPFAO_SetBoundaryPressure(tdy,P); CHKERRQ(ierr);
  ierr = TDyUpdateBoundaryState(tdy); CHKERRQ(ierr);
  ierr = MatMult(tdy->Trans_mat, tdy->P_vec, tdy->TtimesP_vec);

  ierr = TDyMPFAOIFunction_Vertices_3DMesh(P,R_P,ctx); CHKERRQ(ierr);

  ierr = VecGetArray(M,&m); CHKERRQ(ierr);
  ierr = VecGetArray(U_t,&u_t); CHKERRQ(ierr);
  ierr = VecGetArray(R,&r); CHKERRQ(ierr);
  ierr = VecGetArray(R_P,&r_p); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    //
    // u   = [P   M  ]^T
    // u_t = [P_t M_t]^T
    //
    // d(M)/dt = - Del(rho * q)
    // M       = rho * phi * sat

    p_idx = icell*2;
    m_idx = p_idx + 1;

    r[p_idx]  = u_t[m_idx] * cells->volume[icell] - tdy->source_sink[icell] * cells->volume[icell]+ r_p[icell];
    r[m_idx]  = m  [icell] - tdy->rho[icell] * matprop->porosity[icell] * cc->S[icell]* cells->volume[icell];
  }

  /* Cleanup */
  ierr = VecRestoreArray(M,&m); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_t,&u_t); CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r); CHKERRQ(ierr);
  ierr = VecRestoreArray(R_P,&r_p); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}


