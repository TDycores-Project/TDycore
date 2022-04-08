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

PetscErrorCode TDyFVTPFSNESJacobianAccumulation(TDy tdy, PetscInt icell, PetscReal *J) {

  PetscFunctionBegin;

  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh *mesh = fvtpf->mesh;
  TDyCell *cells = &mesh->cells;

  PetscReal rho = fvtpf->rho[icell];
  PetscReal por = fvtpf->porosity[icell];
  PetscReal S = fvtpf->S[icell];
  PetscReal drho_dP = fvtpf->drho_dP[icell];
  PetscReal dpor_dP = 0.0;
  PetscReal dS_dP = fvtpf->dS_dP[icell];
  PetscReal vol_over_dt = cells->volume[icell] / tdy->dtime;

  // accum = rho * por * S * vol/dtime
  *J = (drho_dP * por * S + rho * dpor_dP * S + rho * por * dS_dP) * vol_over_dt;

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

PetscErrorCode RichardsBCResidual(TDyFVTPF *fvtpf, DM dm, MaterialProp *matprop, PetscInt face_id, PetscReal *Res) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscInt *cell_ids, num_face_cells;
  ierr = TDyMeshGetFaceCells(fvtpf->mesh, face_id, &cell_ids, &num_face_cells); CHKERRQ(ierr);
  PetscInt cell_id_up = -cell_ids[0] + 1; // cell up is a boundary cell with negative ids, determine the id to use from *_bnd variables
  PetscInt cell_id_dn = cell_ids[1];      // cell dn is an internal cell

  PetscReal dist_gravity, upweight;
  ierr = FVTPFCalculateDistances(fvtpf, dim, face_id, &dist_gravity, &upweight);

  PetscReal perm_face, Dq;
  ierr = FVTPFComputeFacePeremabilityValueTPF(fvtpf, matprop, dim, face_id, &perm_face, &Dq); CHKERRQ(ierr);

  PetscReal kr_eps = 1.d-8;
  PetscReal sat_eps = 1.d-8;
  if (fvtpf->Kr_bnd[cell_id_up] > kr_eps || fvtpf->Kr[cell_id_dn] > kr_eps) {
    if (fvtpf->S_bnd[cell_id_up] < sat_eps) {
      upweight = 0.0;
    } else if (fvtpf->S[cell_id_dn] < sat_eps) {
      upweight = 1.0;
    }

    PetscReal den_up = fvtpf->rho_bnd[cell_id_up];
    PetscReal den_dn = fvtpf->rho[cell_id_dn];
    PetscReal den_aveg = upweight * den_up + (1.0 - upweight) * den_dn;

    PetscReal gravity_term = den_aveg * dist_gravity;
    PetscReal dphi = fvtpf->P_bnd[cell_id_up] - fvtpf->pressure[cell_id_dn] - gravity_term;

    PetscReal ukvr;
    if (dphi >= 0.0) {
      ukvr = fvtpf->Kr_bnd[cell_id_up]/fvtpf->vis_bnd[cell_id_up];
    } else {
      ukvr = fvtpf->Kr[cell_id_dn]/fvtpf->vis[cell_id_dn];
    }

    PetscReal v_darcy = Dq * ukvr * dphi;

    PetscReal projected_area;
    ierr = FVTPFComputeProjectedArea(fvtpf, dim, face_id, &projected_area);

    PetscReal q = v_darcy * projected_area;
    *Res = q * den_aveg;

  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode RichardsJacobian(TDyFVTPF *fvtpf, DM dm, MaterialProp *matprop, PetscInt face_id, PetscReal *Jup, PetscReal *Jdn) {

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
    PetscReal dden_ave_dp_up = upweight*fvtpf->drho_dP[cell_id_up];
    PetscReal dden_ave_dp_dn = (1.0 - upweight)*fvtpf->drho_dP[cell_id_dn];

    PetscReal gravity_term = den_aveg * dist_gravity;
    PetscReal dgravity_dden_up = upweight * dist_gravity;
    PetscReal dgravity_dden_dn = (1.0 - upweight) * dist_gravity;

    PetscReal dphi = fvtpf->pressure[cell_id_up] - fvtpf->pressure[cell_id_dn] - gravity_term;
    PetscReal dphi_dp_up =  1.0 - dgravity_dden_up * dden_ave_dp_up;
    PetscReal dphi_dp_dn = -1.0 - dgravity_dden_dn * dden_ave_dp_dn;

    PetscReal ukvr;
    PetscReal dukvr_dp_up = 0.0, dukvr_dp_dn = 0.0;

    if (dphi >= 0.0) {
      PetscReal Kr = fvtpf->Kr[cell_id_up];
      PetscReal vis = fvtpf->vis[cell_id_up];
      ukvr = Kr/vis;

      PetscReal dKr_dS = fvtpf->dKr_dS[cell_id_up];
      PetscReal dS_dp = fvtpf->dS_dP[cell_id_up];
      PetscReal dvis_dp = fvtpf->dvis_dP[cell_id_up];
      PetscReal dukr_dp = dKr_dS * dS_dp/vis - Kr * dvis_dp/PetscPowReal(vis,2.0);

      dukvr_dp_up = dukr_dp;
    } else {
      PetscReal Kr = fvtpf->Kr[cell_id_dn];
      PetscReal vis = fvtpf->vis[cell_id_dn];
      ukvr = Kr/vis;

      PetscReal dKr_dS = fvtpf->dKr_dS[cell_id_dn];
      PetscReal dS_dp = fvtpf->dS_dP[cell_id_dn];
      PetscReal dvis_dp = fvtpf->dvis_dP[cell_id_dn];
      PetscReal dukr_dp = dKr_dS * dS_dp/vis - Kr * dvis_dp/PetscPowReal(vis,2.0);

      dukvr_dp_dn = dukr_dp;
    }

    PetscReal v_darcy = Dq * ukvr * dphi;

    PetscReal projected_area;
    ierr = FVTPFComputeProjectedArea(fvtpf, dim, face_id, &projected_area);

    PetscReal q = v_darcy * projected_area;
    PetscReal dq_dp_up = Dq * (dukvr_dp_up * dphi + ukvr * dphi_dp_up)*projected_area;
    PetscReal dq_dp_dn = Dq * (dukvr_dp_dn * dphi + ukvr * dphi_dp_dn)*projected_area;

    //*Res = q * den_aveg;
    *Jup = dq_dp_up * den_aveg + q * dden_ave_dp_up;
    *Jdn = dq_dp_dn * den_aveg + q * dden_ave_dp_dn;

  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode RichardsBCJacobian(TDyFVTPF *fvtpf, DM dm, MaterialProp *matprop, PetscInt face_id, PetscReal *Jdn) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  PetscInt *cell_ids, num_face_cells;
  ierr = TDyMeshGetFaceCells(fvtpf->mesh, face_id, &cell_ids, &num_face_cells); CHKERRQ(ierr);
  PetscInt cell_id_up = -cell_ids[0] + 1;
  PetscInt cell_id_dn = cell_ids[1];

  PetscReal dist_gravity, upweight;
  ierr = FVTPFCalculateDistances(fvtpf, dim, face_id, &dist_gravity, &upweight);

  PetscReal perm_face, Dq;
  ierr = FVTPFComputeFacePeremabilityValueTPF(fvtpf, matprop, dim, face_id, &perm_face, &Dq); CHKERRQ(ierr);

  PetscReal kr_eps = 1.d-8;
  PetscReal sat_eps = 1.d-8;
  if (fvtpf->Kr_bnd[cell_id_up] > kr_eps || fvtpf->Kr[cell_id_dn] > kr_eps) {
    if (fvtpf->S_bnd[cell_id_up] < sat_eps) {
      upweight = 0.0;
    } else if (fvtpf->S[cell_id_dn] < sat_eps) {
      upweight = 1.0;
    }

    PetscReal den_up = fvtpf->rho_bnd[cell_id_up];
    PetscReal den_dn = fvtpf->rho[cell_id_dn];
    PetscReal den_aveg = upweight * den_up + (1.0 - upweight) * den_dn;
    PetscReal dden_ave_dp_dn = (1.0 - upweight)*fvtpf->drho_dP[cell_id_dn];

    PetscReal gravity_term = den_aveg * dist_gravity;
    PetscReal dgravity_dden_dn = upweight * dist_gravity;

    PetscReal dphi = fvtpf->P_bnd[cell_id_up] - fvtpf->pressure[cell_id_dn] - gravity_term;
    PetscReal dphi_dp_dn = -1.0 - dgravity_dden_dn * dden_ave_dp_dn;

    PetscReal ukvr;
    PetscReal dukvr_dp_dn = 0.0;

    if (dphi >= 0.0) {
    } else {
      PetscReal Kr = fvtpf->Kr[cell_id_dn];
      PetscReal vis = fvtpf->vis[cell_id_dn];
      ukvr = Kr/vis;

      PetscReal dKr_dS = fvtpf->dKr_dS[cell_id_dn];
      PetscReal dS_dp = fvtpf->dS_dP[cell_id_dn];
      PetscReal dvis_dp = fvtpf->dvis_dP[cell_id_dn];
      PetscReal dukr_dp = dKr_dS * dS_dp / vis - Kr * dvis_dp/PetscPowReal(vis,2.0);

      dukvr_dp_dn = dukr_dp;
    }

    PetscReal v_darcy = Dq * ukvr * dphi;

    PetscReal projected_area;
    ierr = FVTPFComputeProjectedArea(fvtpf, dim, face_id, &projected_area);

    PetscReal q = v_darcy * projected_area;
    PetscReal dq_dp_dn = Dq * (dukvr_dp_dn * dphi + ukvr * dphi_dp_dn)*projected_area;

    //*Res = q * den_aveg;
    *Jdn = dq_dp_dn * den_aveg + q * dden_ave_dp_dn;


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

  // Compute contribution to residual for boundary faces
  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (!faces->is_local[iface]) continue; // skip non-local face
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

  // Compute contribution to residual for boundary faces
  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (!faces->is_local[iface]) continue; // skip non-local face
    if (!faces->is_internal[iface]) continue; // skip internal faces
    if (!(faces->bc_type[iface] == NEUMANN_BC)) continue; // skip non-flux faces

    PetscInt *cell_ids, num_face_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_face_cells); CHKERRQ(ierr);

    PetscReal Res;
    ierr = RichardsBCResidual(fvtpf, tdy->dm, tdy->matprop, iface, &Res);

    PetscInt cell_id_dn = cell_ids[1];
    r_ptr[cell_id_dn] -= Res;
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

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFVTPFSNESJacobian(SNES snes,Vec U,Mat A, Mat B,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh  *mesh = fvtpf->mesh;
  TDyCell  *cells = &mesh->cells;
  TDyFace  * faces = &mesh->faces;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = MatZeroEntries(B); CHKERRQ(ierr);

  // Compute contribution to Jacobian for internal faces
  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (!faces->is_local[iface]) continue; // skip non-local face
    if (!faces->is_internal[iface]) continue; // skip boundary faces

    PetscInt *cell_ids, num_face_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_face_cells); CHKERRQ(ierr);
    PetscInt cell_id_up = cell_ids[0];
    PetscInt cell_id_dn = cell_ids[1];

    PetscReal Jup, Jdn;
    ierr = RichardsJacobian(fvtpf, tdy->dm, tdy->matprop, iface, &Jup, &Jdn);

    if (cell_id_up >= 0 && cells->is_local[cell_id_up]) {
      //r_ptr[cell_id_up] += Res;
      ierr = MatSetValuesLocal(B,1,&cell_id_up,1,&cell_id_up,&Jup,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(B,1,&cell_id_up,1,&cell_id_dn,&Jdn,ADD_VALUES);CHKERRQ(ierr);
    }

    if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
      //r_ptr[cell_id_dn] -= Res;
      Jup *= -1.0;
      Jdn *= -1.0;
      ierr = MatSetValuesLocal(B,1,&cell_id_dn,1,&cell_id_dn,&Jdn,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(B,1,&cell_id_dn,1,&cell_id_up,&Jup,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  // Compute contribution to Jacobian for boundary faces
  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (!faces->is_local[iface]) continue; // skip non-local face
    if (!faces->is_internal[iface]) continue; // skip internal faces
    if (!(faces->bc_type[iface] == NEUMANN_BC)) continue; // skip non-flux faces

    PetscInt *cell_ids, num_face_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_face_cells); CHKERRQ(ierr);

    PetscInt cell_id_dn = cell_ids[1];
    PetscReal Jdn;
    ierr = RichardsBCJacobian(fvtpf, tdy->dm, tdy->matprop, iface, &Jdn);

    if (cell_id_dn >= 0 && cells->is_local[cell_id_dn]) {
      //r_ptr[cell_id_dn] -= Res;
      Jdn *= -1.0;
      ierr = MatSetValuesLocal(B,1,&cell_id_dn,1,&cell_id_dn,&Jdn,ADD_VALUES);CHKERRQ(ierr);
    }

  }

  for (PetscInt icell = 0; icell<mesh->num_cells; icell++) {

    if (!cells->is_local[icell]) continue;

    PetscReal J;
    ierr = TDyFVTPFSNESJacobianAccumulation(tdy, icell, &J); CHKERRQ(ierr);

    ierr = MatSetValuesLocal(B,1,&icell,1,&icell,&J,ADD_VALUES);CHKERRQ(ierr);

  }

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A !=B ) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}