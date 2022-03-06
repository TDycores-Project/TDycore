#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyoptions.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyfvtpfimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdydiscretization.h>
#include <petscblaslapack.h>

PetscErrorCode TDyCreate_FVTPF(void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Allocate a new context for the FV-TPF method.
  TDyFVTPF* fvtpf;
  ierr = PetscCalloc(sizeof(TDyFVTPF), &fvtpf); CHKERRQ(ierr);
  *context = fvtpf;

  // Initialize defaults and data.
  fvtpf->bc_type = FVTPF_DIRICHLET_BC;
  fvtpf->Pref = 101325.0;
  fvtpf->Tref = 25.0;
  fvtpf->gravity[0] = 0.0; fvtpf->gravity[1] = 0.0; fvtpf->gravity[2] = 0.0;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy_FVTPF(void *context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyFVTPF* fvtpf = context;

  if (fvtpf->vel   ) { ierr = PetscFree(fvtpf->vel); CHKERRQ(ierr); }

  ierr = PetscFree(fvtpf->V); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->X); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->N); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->rho); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->d2rho_dP2); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->vis); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->dvis_dP); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->d2vis_dP2); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->h); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->dh_dP); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->dh_dT); CHKERRQ(ierr);
  ierr = PetscFree(fvtpf->dvis_dT); CHKERRQ(ierr);

  if (fvtpf->P_vec) { ierr = VecDestroy(&fvtpf->P_vec); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetFromOptions_FVTPF(void *context, TDyOptions *options) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  TDyFVTPF* fvtpf = context;

  // Set FV-TPF options.
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: FV-TPF options",""); CHKERRQ(ierr);
  TDyFVTPFBoundaryConditionType bctype = FVTPF_DIRICHLET_BC;

  PetscBool flag;
  ierr = PetscOptionsEnum("-tdy_fvtpf_boundary_condition_type",
      "FV-TPF boundary condition type", "TDyFVTPFSetBoundaryConditionType",
      TDyFVTPFBoundaryConditionTypes,(PetscEnum)bctype,(PetscEnum *)&bctype,
      &flag); CHKERRQ(ierr);
  if (flag && (bctype != fvtpf->bc_type)) {
    fvtpf->bc_type = bctype;
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Set characteristic curve data.
  fvtpf->vangenuchten_m = options->vangenuchten_m;
  fvtpf->vangenuchten_alpha = options->vangenuchten_alpha;
  fvtpf->mualem_poly_x0 = options->mualem_poly_x0;
  fvtpf->mualem_poly_x1 = options->mualem_poly_x1;
  fvtpf->mualem_poly_x2 = options->mualem_poly_x2;
  fvtpf->mualem_poly_dx = options->mualem_poly_dx;

  // Copy g into place.
  fvtpf->gravity[2] = options->gravity_constant;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

// Create a section containing a given number of fields with given names and
// numbers of degrees of freedom, and attach it to the given DM.
static PetscErrorCode SetFields(DM dm, PetscInt num_fields,
                                const char* field_names[num_fields],
                                PetscInt num_field_dof[num_fields]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  // Create the section and register fields and components.
  PetscSection sec;
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, num_fields); CHKERRQ(ierr);
  PetscInt total_num_dof = 0; // total number of field dofs/components per point
  for (PetscInt f = 0; f < num_fields; ++f) {
    ierr = PetscSectionSetFieldName(sec, f, field_names[f]); CHKERRQ(ierr);
    // TODO: should we distinguish between field components and dof?
    ierr = PetscSectionSetFieldComponents(sec, f, num_field_dof[f]); CHKERRQ(ierr);
    total_num_dof += num_field_dof[f];
  }

  // Create a chart on cells.
  PetscInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,cStart,cEnd); CHKERRQ(ierr);

  // Assign degrees of freedom to each field on each cell.
  for(PetscInt c=cStart; c<cEnd; c++) {
    for (PetscInt f = 0; f < num_fields; ++f) {
      ierr = PetscSectionSetFieldDof(sec, c, f, num_field_dof[f]); CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(sec, c, total_num_dof); CHKERRQ(ierr);
  }

  // Assign the section to the DM.
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetLocalSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);

  // TODO: Does this really belong here, or can we move it elsewhere?
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDMFields_Richards_FVTPF(void *context, DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // Set up the section, 1 dof per cell
  ierr = SetFields(dm, 1, (const char*[1]){"LiquidPressure"}, (PetscInt[1]){1});
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Initializes material properties and characteristic curve data.
static PetscErrorCode InitMaterials(TDyFVTPF *fvtpf,
                                    DM dm,
                                    MaterialProp *matprop,
                                    CharacteristicCurves *cc) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  // Allocate storage for material data and characteristic curves, and set to
  // zero using PetscCalloc instead of PetscMalloc.
  PetscInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscInt nc = cEnd-cStart;

  // Material properties
  ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(fvtpf->K)); CHKERRQ(ierr);
  ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(fvtpf->K0)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->porosity)); CHKERRQ(ierr);
  if (MaterialPropHasThermalConductivity(matprop)) {
    ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(fvtpf->Kappa)); CHKERRQ(ierr);
    ierr = PetscCalloc(9*nc*sizeof(PetscReal),&(fvtpf->Kappa0)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilSpecificHeat(matprop)) {
    ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->c_soil)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilDensity(matprop)) {
    ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->rho_soil)); CHKERRQ(ierr);
  }

  // Characteristic curve values
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->Kr)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dKr_dS)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->S)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dS_dP)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->d2S_dP2)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dS_dT)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->Sr)); CHKERRQ(ierr);

  // Water properties
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->rho)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->drho_dP)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->d2rho_dP2)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->vis)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dvis_dP)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->d2vis_dP2)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->h)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dh_dT)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dh_dP)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->drho_dT)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->u)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->du_dP)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->du_dT)); CHKERRQ(ierr);
  ierr = PetscCalloc(nc*sizeof(PetscReal),&(fvtpf->dvis_dT)); CHKERRQ(ierr);

  // Initialize characteristic curve parameters on all cells.
  PetscInt points[nc];
  for (PetscInt c = 0; c < nc; ++c) {
    points[c] = cStart + c;
  }

  // By default, we use the Van Genuchten saturation model.
  {
    PetscReal parameters[2*nc];
    for (PetscInt c = 0; c < nc; ++c) {
      parameters[2*c]   = fvtpf->vangenuchten_m;
      parameters[2*c+1] = fvtpf->vangenuchten_alpha;
    }
    ierr = SaturationSetType(cc->saturation, SAT_FUNC_VAN_GENUCHTEN, nc, points,
                             parameters); CHKERRQ(ierr);
  }

  // By default, we use the the Mualem relative permeability model.
  {
    PetscInt num_params = 9;
    PetscReal parameters[num_params*nc];
    for (PetscInt c = 0; c < nc; ++c) {
      PetscReal m = fvtpf->vangenuchten_m;
      PetscReal poly_x0 = fvtpf->mualem_poly_x0;
      PetscReal poly_x1 = fvtpf->mualem_poly_x1;
      PetscReal poly_x2 = fvtpf->mualem_poly_x2;
      PetscReal poly_dx = fvtpf->mualem_poly_dx;

      PetscInt offset = num_params*c;
      parameters[offset    ]   = m;
      parameters[offset + 1] = poly_x0;
      parameters[offset + 2] = poly_x1;
      parameters[offset + 3] = poly_x2;
      parameters[offset + 4] = poly_dx;

      // Set up cubic polynomial coefficients for the cell.
      PetscReal coeffs[4];
      ierr = RelativePermeability_Mualem_GetSmoothingCoeffs(m, poly_x0, poly_x1, poly_x2, poly_dx, coeffs);
      CHKERRQ(ierr);
      parameters[offset + 5] = coeffs[0];
      parameters[offset + 6] = coeffs[1];
      parameters[offset + 7] = coeffs[2];
      parameters[offset + 8] = coeffs[3];
    }
    ierr = RelativePermeabilitySetType(cc->rel_perm, REL_PERM_FUNC_MUALEM, nc,
                                       points, parameters); CHKERRQ(ierr);
  }

  // Compute material properties.
  ierr = MaterialPropComputePermeability(matprop, nc, fvtpf->X, fvtpf->K0); CHKERRQ(ierr);
  memcpy(fvtpf->K, fvtpf->K0, 9*nc*sizeof(PetscReal));
  ierr = MaterialPropComputePorosity(matprop, nc, fvtpf->X, fvtpf->porosity); CHKERRQ(ierr);
  ierr = MaterialPropComputeResidualSaturation(matprop, nc, fvtpf->X, fvtpf->Sr); CHKERRQ(ierr);
  if (MaterialPropHasThermalConductivity(matprop)) {
    ierr = MaterialPropComputeThermalConductivity(matprop, nc, fvtpf->X, fvtpf->Kappa); CHKERRQ(ierr);
    memcpy(fvtpf->Kappa0, fvtpf->Kappa, 9*nc*sizeof(PetscReal));
  }
  if (MaterialPropHasSoilSpecificHeat(matprop)) {
    ierr = MaterialPropComputeSoilSpecificHeat(matprop, nc, fvtpf->X, fvtpf->c_soil); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilDensity(matprop)) {
    ierr = MaterialPropComputeSoilDensity(matprop, nc, fvtpf->X, fvtpf->rho_soil); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode AllocateMemoryForBoundaryValues(TDyFVTPF *fvtpf,
                                                      EOS *eos) {

  TDyMesh *mesh = fvtpf->mesh;
  PetscInt nbnd_faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  nbnd_faces = mesh->num_boundary_faces;

  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->Kr_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->dKr_dS_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->S_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->dS_dP_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->d2S_dP2_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->P_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->rho_bnd)); CHKERRQ(ierr);
  ierr = PetscMalloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->vis_bnd)); CHKERRQ(ierr);

  PetscInt i;
  PetscReal dden_dP, d2den_dP2, dmu_dP, d2mu_dP2;
  for (i=0;i<nbnd_faces;i++) {
    ierr = EOSComputeWaterDensity(eos, fvtpf->Pref, &(fvtpf->rho_bnd[i]), &dden_dP, &d2den_dP2); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos, fvtpf->Pref, &(fvtpf->vis_bnd[i]), &dmu_dP, &d2mu_dP2); CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode AllocateMemoryForSourceSinkValues(TDyFVTPF *fvtpf) {

  TDyMesh *mesh = fvtpf->mesh;
  PetscInt ncells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ncells = mesh->num_cells;

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(fvtpf->source_sink)); CHKERRQ(ierr);

  PetscInt i;
  for (i=0;i<ncells;i++) fvtpf->source_sink[i] = 0.0;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyComputeErrorNorms_FVTPF(void *context, DM dm, Conditions *conditions,
                                          Vec U, PetscReal *p_norm, PetscReal *v_norm) {
  TDyFVTPF *fvtpf = context;
  TDyMesh  *mesh = fvtpf->mesh;
  TDyCell  *cells = &mesh->cells;
  TDyFace  *faces = &mesh->faces;
  PetscScalar *u;
  Vec localU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  if (!ConditionsHasBoundaryPressure(conditions)) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the boundary pressure function with TDySetBoundaryPressureFunction");
  }

  PetscInt dim = 3;

  if (p_norm) {

    ierr = DMGetLocalVector(dm,&localU); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm,U,INSERT_VALUES,localU); CHKERRQ(ierr);
    ierr = VecGetArray(localU,&u); CHKERRQ(ierr);

    PetscReal norm_sum = 0.0;
    PetscReal norm = 0.0;

    for (PetscInt icell=0; icell<mesh->num_cells; icell++) {

      if (!cells->is_local[icell]) continue;

      PetscReal pressure;
      ierr = ConditionsComputeBoundaryPressure(conditions, 1, &(fvtpf->X[icell*dim]), &pressure);CHKERRQ(ierr);
      norm += (PetscSqr(pressure - u[icell])) * cells->volume[icell];
    }

    ierr = VecRestoreArray(localU, &u); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localU); CHKERRQ(ierr);

    ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                         PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);

    *p_norm = PetscSqrtReal(norm_sum);
  }

  if (v_norm) {
    PetscInt fStart, fEnd;
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

    PetscReal norm_sum = 0.0;
    PetscReal norm     = 0.0;

    for (PetscInt icell=0; icell<mesh->num_cells; icell++) {

      if (!cells->is_local[icell]) continue;

      for (PetscInt iface=0; iface<cells->num_faces[icell]; iface++) {
        PetscInt faceStart = cells->face_offset[icell];
        PetscInt face_id = cells->face_ids[faceStart + iface];

        PetscReal vel[3];
        ierr = ConditionsComputeBoundaryVelocity(conditions, 1, &(fvtpf->X[(face_id + fStart)*dim]), vel);CHKERRQ(ierr);
        PetscReal vel_normal = TDyADotB(vel,&(faces->normal[face_id].V[0]),dim);
        if (fvtpf->vel_count[face_id] != faces->num_vertices[face_id]) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"tdy->vel_count != faces->num_vertices[face_id]");
        }

        norm += PetscSqr((vel_normal - fvtpf->vel[face_id]))*cells->volume[icell];
      }
    }

    ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                         PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
    *v_norm = PetscSqrtReal(norm_sum);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateDiagnostics_FVTPF(void *context,
                                          DM diags_dm,
                                          Vec diags_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  TDyFVTPF *fvtpf = context;
  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(diags_dm,0,&c_start,&c_end); CHKERRQ(ierr);
  PetscReal *v;
  VecGetArray(diags_vec, &v);
  for (PetscInt c = c_start; c < c_end; ++c) {
    v[2*c+DIAG_LIQUID_SATURATION] = fvtpf->S[c];
    v[2*c+DIAG_LIQUID_MASS] = fvtpf->rho[c] * fvtpf->V[c];
  }
  VecRestoreArray(diags_vec, &v);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateState_Richards_FVTPF(void *context, DM dm,
                                             EOS *eos,
                                             MaterialProp *matprop,
                                             CharacteristicCurves *cc,
                                             PetscInt num_cells,
                                             PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyFVTPF *fvtpf = context;

  PetscInt dim = 3;
  PetscInt dim2 = dim*dim;
  PetscInt cStart = 0, cEnd = num_cells;
  PetscInt nc = cEnd - cStart;

  // Compute the capillary pressure on all cells.
  PetscReal Pc[nc];
  for (PetscInt c=0;c<cEnd-cStart;c++) {
    Pc[c] = fvtpf->Pref - U[c];
  }

  // Compute the saturation and its derivatives.
  ierr = SaturationCompute(cc->saturation, fvtpf->Sr, Pc, fvtpf->S, fvtpf->dS_dP,
                           fvtpf->d2S_dP2);

  // Compute the effective saturation on cells.
  PetscReal Se[nc];
  for (PetscInt c=0;c<nc;c++) {
    Se[c] = (fvtpf->S[c] - fvtpf->Sr[c])/(1.0 - fvtpf->Sr[c]);
  }

  // Compute the relative permeability and its derivative (w.r.t. Se).
  ierr = RelativePermeabilityCompute(cc->rel_perm, Se, fvtpf->Kr, fvtpf->dKr_dS);

  // Correct dKr/dS using the chain rule, and update the permeability.
  for (PetscInt c=0;c<nc;c++) {
    PetscReal dSe_dS = 1.0/(1.0 - fvtpf->Sr[c]);
    fvtpf->dKr_dS[c] *= dSe_dS; // correct dKr/dS

    for(PetscInt j=0; j<dim2; j++) {
      fvtpf->K[c*dim2+j] = fvtpf->K0[c*dim2+j] * fvtpf->Kr[c];
    }

    // Also update water properties.
    PetscReal P = fvtpf->Pref - Pc[c]; // pressure
    ierr = EOSComputeWaterDensity(eos, P, &(fvtpf->rho[c]),
                                  &(fvtpf->drho_dP[c]),
                                  &(fvtpf->d2rho_dP2[c])); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos, P, &(fvtpf->vis[c]),
                                    &(fvtpf->dvis_dP[c]),
                                    &(fvtpf->d2vis_dP2[c])); CHKERRQ(ierr);
  }

  PetscReal *p_vec_ptr;

  ierr = VecGetArray(fvtpf->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  for (PetscInt c=0; c<nc; ++c) {
    p_vec_ptr[c] = fvtpf->Pref - Pc[c]; // pressure
  }
  ierr = VecRestoreArray(fvtpf->P_vec,&p_vec_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeGeometry(TDyFVTPF *fvtpf, DM dm) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  if (dim == 2) {
    SETERRQ(comm,PETSC_ERR_USER,"MPFA-O method supports only 3D calculations.");
  }

  // Compute/store plex geometry.
  PetscInt pStart, pEnd, vStart, vEnd, eStart, eEnd;
  ierr = DMPlexGetChart(dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(fvtpf->V));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(fvtpf->X));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(fvtpf->N));
  CHKERRQ(ierr);

  PetscSection coordSection;
  Vec coordinates;
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal (dm, &coordinates); CHKERRQ(ierr);
  PetscReal *coords;
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(PetscInt p=pStart; p<pEnd; p++) {
    if((p >= vStart) && (p < vEnd)) {
      PetscInt offset;
      ierr = PetscSectionGetOffset(coordSection,p,&offset); CHKERRQ(ierr);
      for(PetscInt d=0; d<dim; d++) fvtpf->X[p*dim+d] = coords[offset+d];
    } else {
      if((dim == 3) && (p >= eStart) && (p < eEnd)) continue;
      PetscLogEvent t11 = TDyGetTimer("DMPlexComputeCellGeometryFVM");
      TDyStartTimer(t11);
      ierr = DMPlexComputeCellGeometryFVM(dm,p,&(fvtpf->V[p]),
                                          &(fvtpf->X[p*dim]),
                                          &(fvtpf->N[p*dim])); CHKERRQ(ierr);
      TDyStopTimer(t11);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

// Creates a TDyMesh object to be used by the MPFA-O method.
static PetscErrorCode CreateMesh(TDyFVTPF *fvtpf, DM dm) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Create the mesh.
  ierr = TDyMeshCreate(dm, fvtpf->V, fvtpf->X, fvtpf->N, &fvtpf->mesh);

/* TODO: this stuff doesn't work with the new mesh construction process, and
 * TODO: I'm not sure we still need it. -JNJ
  // Read/write connectivity and geometry data if requested.
  if (fvtpf->read_geom_attributes) {
    ierr = TDyMeshReadGeometry(fvtpf->mesh, fvtpf->geom_attributes_file); CHKERRQ(ierr);
    fvtpf->read_geom_attributes = 0;
  }

  if (fvtpf->output_geom_attributes) {
    ierr = TDyMeshWriteGeometry(fvtpf->mesh, fvtpf->geom_attributes_file); CHKERRQ(ierr);
    fvtpf->output_geom_attributes = 0;
  }
*/

  ierr = TDyMeshGetMaxVertexConnectivity(fvtpf->mesh, &fvtpf->ncv, &fvtpf->nfv);
  ierr = PetscMalloc(fvtpf->mesh->num_faces*sizeof(PetscReal),
                     &(fvtpf->vel )); CHKERRQ(ierr);
  ierr = TDyInitialize_RealArray_1D(fvtpf->vel, fvtpf->mesh->num_faces, 0.0); CHKERRQ(ierr);
  ierr = PetscMalloc(fvtpf->mesh->num_faces*sizeof(PetscInt),
                     &(fvtpf->vel_count)); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_1D(fvtpf->vel_count, fvtpf->mesh->num_faces, 0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup function for Richards + FV-TPF
PetscErrorCode TDySetup_Richards_FVTPF(void *context, DM dm, EOS *eos,
                                       MaterialProp *matprop,
                                       CharacteristicCurves *cc,
                                       Conditions *conditions) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  TDyFVTPF *fvtpf = context;

  ierr = ComputeGeometry(fvtpf, dm); CHKERRQ(ierr);
  ierr = CreateMesh(fvtpf, dm); CHKERRQ(ierr);
  ierr = InitMaterials(fvtpf, dm, matprop, cc); CHKERRQ(ierr);

  // Gather mesh data.
  PetscInt nLocalCells = fvtpf->mesh->num_cells_local;
  PetscInt nNonLocalFaces = TDyMeshGetNumberOfNonLocalFaces(fvtpf->mesh);
  PetscInt nNonInternalFaces = TDyMeshGetNumberOfNonInternalFaces(fvtpf->mesh);
  PetscInt ncol = nLocalCells + nNonLocalFaces + nNonInternalFaces;

  ierr = VecCreateSeq(PETSC_COMM_SELF,ncol,&fvtpf->P_vec);

  ierr = AllocateMemoryForBoundaryValues(fvtpf, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForSourceSinkValues(fvtpf); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

