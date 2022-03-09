#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdyfvtpfimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdydiscretization.h>

PetscErrorCode TDyFVTPFSNESAccumulation(TDy tdy, PetscInt icell, PetscReal *accum) {

  PetscFunctionBegin;

  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh *mesh = fvtpf->mesh;
  TDyCell *cells = &mesh->cells;

  *accum = fvtpf->rho[icell] * fvtpf->porosity[icell] * fvtpf->S[icell] * cells->volume[icell] / tdy->dtime;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyFVTPFSNESPreSolve(TDy tdy) {

  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh *mesh = fvtpf->mesh;
  TDyCell *cells = &mesh->cells;
  PetscReal *p, *accum_prev;
  PetscInt icell;
  PetscErrorCode ierr;

  TDY_START_FUNCTION_TIMER()


  // Update the auxillary variables
  ierr = VecGetArray(tdy->soln_prev,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p, mesh->num_cells_local); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_prev,&p); CHKERRQ(ierr);

  ierr = VecGetArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);

  for (icell=0;icell<mesh->num_cells;icell++){

    if (!cells->is_local[icell]) continue;

    // d(rho*phi*s)/dt * Vol
    //  = [(rho*phi*s)^{t+1} - (rho*phi*s)^t]/dt * Vol
    //  = accum_current - accum_prev
    ierr = TDyFVTPFSNESAccumulation(tdy,icell,&accum_prev[icell]); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode RichardsResidual(TDyFVTPF *fvtpf, DM dm, MaterialProp *matprop, PetscInt face_id, PetscReal *Res) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscInt *cell_ids, num_face_cells;
  ierr = TDyMeshGetFaceCells(fvtpf->mesh, face_id, &cell_ids, &num_face_cells); CHKERRQ(ierr);
  PetscInt cell_id_up = cell_ids[0];
  PetscInt cell_id_dn = cell_ids[1];

  PetscReal dist_gravity, upweight;
  ierr = FVTPFCalculateDistances(fvtpf, dim, face_id, &dist_gravity, &upweight);

  PetscReal perm_face, Dq;
  ierr = FVTPFComputeFacePeremabilityValueTPF(fvtpf, matprop, dim, face_id, &perm_face, &Dq); CHKERRQ(ierr);

  PetscReal kr_eps = 1.d-8;
  PetscReal sat_eps = 1.d-8;
  if (fvtpf->Kr[cell_id_up] > kr_eps || fvtpf->Kr[cell_id_dn] > kr_eps) {
    if (fvtpf->S[cell_id_up] < sat_eps) {
      upweight = 0.0;
    } else if (fvtpf->S[cell_id_dn] < sat_eps) {
      upweight = 1.0;
    }

    PetscReal den_up = fvtpf->rho[cell_id_up];
    PetscReal den_dn = fvtpf->rho[cell_id_dn];
    PetscReal den_aveg = upweight * den_up + (1.0 - upweight) * den_dn;

    PetscReal gravity_term = den_aveg * dist_gravity;
    PetscReal dphi = fvtpf->pressure[cell_id_up] - fvtpf->pressure[cell_id_dn] - gravity_term;

    PetscInt cell_id;
    if (dphi >= 0.0) {
      cell_id = cell_id_up;
    } else {
      cell_id = cell_id_dn;
    }
    PetscReal ukvr = fvtpf->Kr[cell_id]/fvtpf->vis[cell_id];

    PetscReal v_darcy = Dq * ukvr * dphi;

    PetscReal projected_area;
    ierr = FVTPFComputeProjectedArea(fvtpf, dim, face_id, &projected_area);

    PetscReal q = v_darcy * projected_area;
    *Res = q * den_aveg;

  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFVTPFSNESFunction(SNES snes,Vec U,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh  *mesh = fvtpf->mesh;
  TDyCell  *cells = &mesh->cells;
  TDyFace  * faces = &mesh->faces;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = TDyGlobalToLocal(tdy, U, tdy->soln_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(R); CHKERRQ(ierr);

  // Update the auxillary variables based on the current iterate
  PetscReal *p;
  ierr = VecGetArray(tdy->soln_loc,&p); CHKERRQ(ierr);
  ierr = TDyUpdateState(tdy, p, mesh->num_cells); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->soln_loc,&p); CHKERRQ(ierr);

  ierr = TDyFVTPFSetBoundaryPressure(tdy,tdy->soln_loc); CHKERRQ(ierr);
  ierr = TDyFVTPFUpdateBoundaryState(tdy); CHKERRQ(ierr);

  PetscReal *r_ptr;
  PetscReal *accum_prev;

  ierr = VecGetArray(R, &r_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(tdy->accumulation_prev,&accum_prev); CHKERRQ(ierr);

  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (!faces->is_internal[iface]) continue; // skip boundary faces

    PetscInt *cell_ids, num_face_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_face_cells); CHKERRQ(ierr);
    PetscInt cell_id_up = cell_ids[0];
    PetscInt cell_id_dn = cell_ids[1];

    PetscReal Res;
    ierr = RichardsResidual(fvtpf, tdy->dm, tdy->matprop, iface, &Res);

    if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
      r_ptr[cell_id_up] += Res;
    }

    if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
      r_ptr[cell_id_dn] -= Res;
    }

  }

  PetscReal accum_current;
  for (PetscInt icell = 0; icell<mesh->num_cells; icell++) {

    if (!cells->is_local[icell]) continue;

    ierr = TDyFVTPFSNESAccumulation(tdy,icell,&accum_current); CHKERRQ(ierr);

    r_ptr[icell] += accum_current - accum_prev[icell];
    r_ptr[icell] -= fvtpf->source_sink[icell];

  }

  ierr = VecRestoreArray(R, &r_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(tdy->accumulation_prev, &accum_prev); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}