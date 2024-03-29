#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyoptions.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyfvtpfimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdydiscretizationimpl.h>
#include <petscblaslapack.h>
#include <private/tdymeshcustomimpl.h>

PetscErrorCode TDyCreate_FVTPF(void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Allocate a new context for the FV-TPF method.
  TDyFVTPF* fvtpf;
  ierr = TDyAlloc(sizeof(TDyFVTPF), &fvtpf); CHKERRQ(ierr);
  *context = fvtpf;

  // Initialize defaults and data.
  fvtpf->bc_type = NEUMANN_BC;
  fvtpf->Pref = 101325.0;
  fvtpf->Tref = 25.0;
  fvtpf->gravity[0] = 0.0; fvtpf->gravity[1] = 0.0; fvtpf->gravity[2] = 0.0;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy_FVTPF(void *context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyFVTPF* fvtpf = context;

  if (fvtpf->vel   ) { ierr = TDyFree(fvtpf->vel); CHKERRQ(ierr); }

  ierr = TDyFree(fvtpf->V); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->X); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->N); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->rho); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->d2rho_dP2); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->vis); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->dvis_dP); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->d2vis_dP2); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->h); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->dh_dP); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->dh_dT); CHKERRQ(ierr);
  ierr = TDyFree(fvtpf->dvis_dT); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetFromOptions_FVTPF(void *context, TDyOptions *options) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  TDyFVTPF* fvtpf = context;

  // Set FV-TPF options.
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: FV-TPF options","");
  TDyBoundaryConditionType bctype = DIRICHLET_BC;

  PetscBool flag;
  ierr = PetscOptionsEnum("-tdy_fvtpf_boundary_condition_type",
      "FV-TPF boundary condition type", "TDyFVTPFSetBoundaryConditionType",
      TDyBoundaryConditionTypes,(PetscEnum)bctype,(PetscEnum *)&bctype,
      &flag); CHKERRQ(ierr);
  if (flag && (bctype != fvtpf->bc_type)) {
    fvtpf->bc_type = bctype;
  }
  PetscOptionsEnd();

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

PetscErrorCode TDyGetNumDMFields_Richards_FVTPF(void *context) {
  PetscFunctionBegin;
  PetscInt ndof = 1; // LiquidPressure
  PetscFunctionReturn(ndof);
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
                                    MaterialProp *matprop,
                                    CharacteristicCurves *cc) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Allocate storage for material data and characteristic curves.
  PetscInt nc = (fvtpf->mesh)->num_cells;

  // Material properties
  ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(fvtpf->K)); CHKERRQ(ierr);
  ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(fvtpf->K0)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->porosity)); CHKERRQ(ierr);
  if (MaterialPropHasThermalConductivity(matprop)) {
    ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(fvtpf->Kappa)); CHKERRQ(ierr);
    ierr = TDyAlloc(9*nc*sizeof(PetscReal),&(fvtpf->Kappa0)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilSpecificHeat(matprop)) {
    ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->c_soil)); CHKERRQ(ierr);
  }
  if (MaterialPropHasSoilDensity(matprop)) {
    ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->rho_soil)); CHKERRQ(ierr);
  }

  // Characteristic curve values
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->Kr)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dKr_dS)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->S)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dS_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->d2S_dP2)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dS_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->Sr)); CHKERRQ(ierr);

  // 
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->pressure)); CHKERRQ(ierr);

  // Water properties
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->rho)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->drho_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->d2rho_dP2)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->vis)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dvis_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->d2vis_dP2)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->h)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dh_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dh_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->drho_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->u)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->du_dP)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->du_dT)); CHKERRQ(ierr);
  ierr = TDyAlloc(nc*sizeof(PetscReal),&(fvtpf->dvis_dT)); CHKERRQ(ierr);

  // Initialize characteristic curve parameters on all cells.
  PetscInt points[nc];
  for (PetscInt c = 0; c < nc; ++c) {
    points[c] = c;
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

  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->Kr_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->dKr_dS_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->S_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->dS_dP_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->d2S_dP2_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->P_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->rho_bnd)); CHKERRQ(ierr);
  ierr = TDyAlloc(nbnd_faces*sizeof(PetscReal),&(fvtpf->vis_bnd)); CHKERRQ(ierr);

  PetscInt i;
  PetscReal dden_dP, d2den_dP2, dmu_dP, d2mu_dP2;
  for (i=0;i<nbnd_faces;i++) {
    ierr = EOSComputeWaterDensity(eos,
      fvtpf->Pref, fvtpf->Tref, 0.0,
      &(fvtpf->rho_bnd[i]), &dden_dP, &d2den_dP2); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos,
      fvtpf->Pref, fvtpf->Tref, 0.0,
      &(fvtpf->vis_bnd[i]), &dmu_dP, &d2mu_dP2); CHKERRQ(ierr);
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

  ierr = TDyAlloc(ncells*sizeof(PetscReal),&(fvtpf->source_sink)); CHKERRQ(ierr);

  PetscInt i;
  for (i=0;i<ncells;i++) fvtpf->source_sink[i] = 0.0;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

static PetscErrorCode SetFaceBoundaryConditionType(TDyFVTPF *fvtpf, Conditions *condition) {

  if (ConditionsHasBoundaryPressureType(condition)) {
    TDyMesh  *mesh = fvtpf->mesh;
    TDyFace  *faces = &mesh->faces;
    PetscErrorCode ierr;

    PetscInt boundary_type;
    for (PetscInt iface=0; iface<mesh->num_faces; iface++){

      // skip non-locan and internal faces
      if (!faces->is_local[iface] || faces->is_internal[iface]) continue;

      TDyCoordinate face_centroid;

      ierr = TDyMeshGetFaceCentroid(mesh, iface, &face_centroid); CHKERRQ(ierr);
      ierr = ConditionsAssignBoundaryPressureType(condition, 1, &(face_centroid.X[0]), &boundary_type); CHKERRQ(ierr);
      if (boundary_type < DIRICHLET_BC || boundary_type > SEEPAGE_BC) {
        char error_msg[100];
        sprintf(error_msg,"The boundary pressure type is %d that is outside the allowable range of [%d %d]\n",boundary_type,DIRICHLET_BC,SEEPAGE_BC);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,error_msg);
      }
      faces->bc_type[iface] = boundary_type;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateDiagnostics_FVTPF(void *context,
                                          DM diags_dm,
                                          Vec diags_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  TDyFVTPF *fvtpf = context;
  TDyMesh  *mesh = fvtpf->mesh;
  TDyCell  *cells = &mesh->cells;

  PetscInt c_start, c_end;
  ierr = DMPlexGetHeightStratum(diags_dm,0,&c_start,&c_end); CHKERRQ(ierr);
  PetscReal *v;
  VecGetArray(diags_vec, &v);

  PetscInt count = 0;
  for (PetscInt c = c_start; c < c_end; ++c) {

    if (!cells->is_local[c]) continue;

    v[2*count + DIAG_LIQUID_SATURATION] = fvtpf->S[c];
    v[2*count + DIAG_LIQUID_MASS] = fvtpf->rho[c] * fvtpf->V[c];

    count++;
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
    fvtpf->pressure[c] = U[c];
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
    ierr = EOSComputeWaterDensity(eos, P, fvtpf->Tref, 0.0,
      &(fvtpf->rho[c]), &(fvtpf->drho_dP[c]),
      &(fvtpf->d2rho_dP2[c])); CHKERRQ(ierr);
    ierr = EOSComputeWaterViscosity(eos, P, fvtpf->Tref, 0.0,
      &(fvtpf->vis[c]), &(fvtpf->dvis_dP[c]),
      &(fvtpf->d2vis_dP2[c])); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// Setup function for Richards + FV-TPF
PetscErrorCode TDySetup_Richards_FVTPF(void *context, TDyDiscretizationType *discretization, EOS *eos,
                                       MaterialProp *matprop,
                                       CharacteristicCurves *cc,
                                       Conditions *conditions) {
  PetscFunctionBegin;

  PetscErrorCode ierr;
  TDyFVTPF *fvtpf = context;
  DM dm;
  ierr = PetscPrintf(PETSC_COMM_WORLD," Setting up RICHARDS FV-TPF method\n");
  ierr = TDyDiscretizationGetDM(discretization,&dm); CHKERRQ(ierr);

  TDyDM *tdydm_ptr;
  ierr = TDyDiscretizationGetTDyDM(discretization, &tdydm_ptr); CHKERRQ(ierr);

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);

  switch (tdydm_ptr->dmtype) {
    case PLEX_TYPE:
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Creating Mesh using DMPlex\n");
    ierr = TDyMeshCreateFromPlex(dm, &fvtpf->V, &fvtpf->X, &fvtpf->N, &fvtpf->mesh);
    ierr = TDyMeshGetMaxVertexConnectivity(fvtpf->mesh, &fvtpf->ncv, &fvtpf->nfv);
    break;

    case TDYCORE_DM_TYPE:
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Creating Mesh using DMShell\n");
    ierr = TDyMeshCreateFromDiscretization(discretization, &fvtpf->mesh);
    break;

    default:
    SETERRQ(comm,PETSC_ERR_USER,"Unknown DMTYPE");

  }

  ierr = TDyAllocate_RealArray_1D(&(fvtpf->vel), fvtpf->mesh->num_faces); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&(fvtpf->vel_count), fvtpf->mesh->num_faces); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD," Initializing material properties\n");
  ierr = InitMaterials(fvtpf, matprop, cc); CHKERRQ(ierr);

  ierr = AllocateMemoryForBoundaryValues(fvtpf, eos); CHKERRQ(ierr);
  ierr = AllocateMemoryForSourceSinkValues(fvtpf); CHKERRQ(ierr);
  ierr = SetFaceBoundaryConditionType(fvtpf, conditions); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFVTPFUpdateBoundaryState(TDy tdy) {

  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh *mesh = fvtpf->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Loop over boundary faces and assemble a list of "boundary cells" attached
  // to them, plus indices for storing quantities on the boundary.
  // num_boundary_cells <= mesh->num_boundary_faces, so we can pre-size
  // our list of cells.
  PetscInt num_boundary_cells = 0;
  PetscInt boundary_cells[mesh->num_boundary_faces];
  PetscInt p_bnd_indices[mesh->num_boundary_faces];
  for (PetscInt iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue; // skip non-boundary faces

    PetscInt *cell_ids, num_face_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_face_cells); CHKERRQ(ierr);

    if (cell_ids[0] >= 0) {
      boundary_cells[num_boundary_cells] = cell_ids[0];
      p_bnd_indices[num_boundary_cells] = -cell_ids[1] - 1;
    } else {
      boundary_cells[num_boundary_cells] = cell_ids[1];
      p_bnd_indices[num_boundary_cells] = -cell_ids[0] - 1;
    }
    ++num_boundary_cells;
  }

  // Store the capillary pressure and residual saturation on the boundary.
  PetscReal Pc[num_boundary_cells], Sr[num_boundary_cells];
  for (PetscInt c = 0; c < num_boundary_cells; ++c) {
    PetscInt c_index = boundary_cells[c];
    Sr[c] = fvtpf->Sr[c_index];
    PetscInt b_index = p_bnd_indices[c];
    Pc[c] = fvtpf->Pref - fvtpf->P_bnd[b_index];
  }

  // Compute the saturation and its derivatives on the boundary.
  CharacteristicCurves *cc = tdy->cc;
  PetscReal S[num_boundary_cells], dS_dP[num_boundary_cells],
            d2S_dP2[num_boundary_cells];
  ierr = SaturationComputeOnPoints(cc->saturation, num_boundary_cells,
                                   boundary_cells, Sr, Pc, S, dS_dP, d2S_dP2);
  CHKERRQ(ierr);

  // Compute the effective saturation and its derivative w.r.t. S on the
  // boundary.
  PetscReal Se[num_boundary_cells], dSe_dS[num_boundary_cells];
  for (PetscInt c = 0; c < num_boundary_cells; ++c) {
    Se[c] = (S[c] - Sr[c])/(1.0 - Sr[c]);
    dSe_dS[c] = 1.0/(1.0 - Sr[c]);
  }

  // Compute the relative permeability and its derivative on the boundary.
  PetscReal Kr[num_boundary_cells], dKr_dSe[num_boundary_cells];
  ierr = RelativePermeabilityComputeOnPoints(cc->rel_perm, num_boundary_cells,
                                             boundary_cells, Se, Kr, dKr_dSe);
  CHKERRQ(ierr);

  // Copy the boundary quantities into place.
  for (PetscInt c = 0; c < num_boundary_cells; ++c) {
    PetscInt p_bnd_idx = p_bnd_indices[c];
    fvtpf->S_bnd[p_bnd_idx] = S[c];
    fvtpf->dS_dP_bnd[p_bnd_idx] = dS_dP[c];
    fvtpf->d2S_dP2_bnd[p_bnd_idx] = d2S_dP2[c];
    fvtpf->Kr_bnd[p_bnd_idx] = Kr[c];
    fvtpf->dKr_dS_bnd[p_bnd_idx] = dKr_dSe[c] * dSe_dS[c];
  }

  PetscFunctionReturn(0);
}
PetscErrorCode TDyFVTPFSetBoundaryPressure(TDy tdy, Vec Ul) {

  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh *mesh = fvtpf->mesh;
  TDyFace *faces = &mesh->faces;
  PetscErrorCode ierr;
  PetscInt dim;
  PetscInt p_bnd_idx, cell_id, iface;
  PetscReal *u_p;
  Conditions *conditions = tdy->conditions;

  PetscFunctionBegin;

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);

  ierr = VecGetArray(Ul,&u_p); CHKERRQ(ierr);

  PetscInt ncells = mesh->num_cells;
  PetscReal p[ncells];
  for (PetscInt c=0;c<ncells;c++) {
    p[c] = u_p[c];
  }

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  for (iface=0; iface<mesh->num_faces; iface++) {

    if (faces->is_internal[iface]) continue;

    PetscInt *cell_ids, num_cells;
    ierr = TDyMeshGetFaceCells(mesh, iface, &cell_ids, &num_cells); CHKERRQ(ierr);

    if (cell_ids[0] >= 0) {
      cell_id = cell_ids[0];
      p_bnd_idx = -cell_ids[1] - 1;
    } else {
      cell_id = cell_ids[1];
      p_bnd_idx = -cell_ids[0] - 1;
    }

    if (ConditionsHasBoundaryPressure(conditions)) {
      ierr = ConditionsComputeBoundaryPressure(conditions, 1,
        faces->centroid[iface].X, &(fvtpf->P_bnd[p_bnd_idx])); CHKERRQ(ierr);
    } else {
      fvtpf->P_bnd[p_bnd_idx] = p[cell_id];
    }

    //p_vec_ptr[p_bnd_idx + ncells] = fvtpf->P_bnd[p_bnd_idx];
  }

  ierr = VecRestoreArray(Ul,&u_p); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Computes face permeability tensor as a harmonically distance-weighted
//  permeability of upwind and downwind permeability tensors
///
/// @param [in] tdy A TDy struct
/// @param [in] face_id ID of the face
/// @param [out] Kup components of the upwind permeability tensor in a row-major order
/// @param [out] Kdn components of the downwind permeability tensor in a row-major order
PetscErrorCode FVTPFExtractUpAndDownPermeabilityTensors(TDyFVTPF *fvtpf,
    MaterialProp *matprop, PetscInt face_id, PetscInt dim,
    PetscReal Kup[dim*dim], PetscReal Kdn[dim*dim]) {

  PetscFunctionBegin;

  TDyMesh *mesh = fvtpf->mesh;
  PetscErrorCode ierr;

  PetscInt *face_cell_ids, num_cell_ids;
  ierr = TDyMeshGetFaceCells(mesh, face_id, &face_cell_ids, &num_cell_ids); CHKERRQ(ierr);
  PetscInt cell_id_up = face_cell_ids[0];
  PetscInt cell_id_dn = face_cell_ids[1];

  if (cell_id_up < 0 && cell_id_dn < 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Both cell IDs sharing a face are not valid");
  }

  for (PetscInt kk = 0; kk < dim; kk++)
  {
    for (PetscInt mm = 0; mm < dim; mm++)
    {
      if (cell_id_up >= 0) Kup[mm * dim + kk] = fvtpf->K0[cell_id_up * dim * dim + kk * dim + mm];
      else                 Kup[mm * dim + kk] = fvtpf->K0[cell_id_dn * dim * dim + kk * dim + mm];

      if (cell_id_dn >= 0) Kdn[mm * dim + kk] = fvtpf->K0[cell_id_dn * dim * dim + kk * dim + mm];
      else                 Kdn[mm * dim + kk] = fvtpf->K0[cell_id_up * dim * dim + kk * dim + mm];
    }
  }

  PetscFunctionReturn(0);
}

/// For the TPF approach, compute the value of permeability at face center that
/// is used in computing the discretization of gravity term.
///
/// - First, compute permeability scalar values from the up and down cell permeability
///   tensor using PFLOTRAN's approach
/// - Second, compute distance-weighted harmonic-average of scalar permeabilities
///
/// @param [in] tdy A TDy struct
/// @param [in] dim dimension of the problem
/// @param [in] face_id ID of the face
/// @param [out] *Kface_value Permeability value at the face
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode FVTPFComputeFacePermeabililtyValueTPF(TDyFVTPF *fvtpf, MaterialProp *matprop, PetscInt dim, PetscInt face_id, PetscReal *Kface_value, PetscReal *Dq) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscReal u_up2dn[dim];
  PetscReal dist_up = (fvtpf->mesh)->faces.dist_up_dn[face_id][0];
  PetscReal dist_dn = (fvtpf->mesh)->faces.dist_up_dn[face_id][1];
  for (PetscInt idim=0; idim<dim; idim++) {
    u_up2dn[idim] = (fvtpf->mesh)->faces.unit_vec_up_dn[face_id][idim];
  }

  PetscReal Kup[dim*dim], Kdn[dim*dim];
  ierr = FVTPFExtractUpAndDownPermeabilityTensors(fvtpf, matprop, face_id, dim, Kup, Kdn); CHKERRQ(ierr);

  PetscReal Kup_value = 0.0, Kdn_value = 0.0;
  for (PetscInt kk=0; kk<dim; kk++) {
    Kup_value += pow(u_up2dn[kk],2.0)/Kup[kk*dim + kk];
    Kdn_value += pow(u_up2dn[kk],2.0)/Kdn[kk*dim + kk];
  }

  Kup_value = 1.0/Kup_value;
  Kdn_value = 1.0/Kdn_value;

  *Kface_value = (Kup_value*Kdn_value)/(dist_up*Kdn_value + dist_dn*Kup_value);
  *Dq = *Kface_value;

  PetscFunctionReturn(0);
}

PetscErrorCode FVTPFCalculateDistances(TDyFVTPF *fvtpf, PetscInt dim, PetscInt face_id, PetscReal *dist_gravity, PetscReal *upweight) {

  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscReal u_up2dn[dim];
  PetscReal dist_up = (fvtpf->mesh)->faces.dist_up_dn[face_id][0];
  PetscReal dist_dn = (fvtpf->mesh)->faces.dist_up_dn[face_id][1];
  for (PetscInt idim=0; idim<dim; idim++) {
    u_up2dn[idim] = (fvtpf->mesh)->faces.unit_vec_up_dn[face_id][idim];
  }

  // In order to match PFLOTRAN results, compute the ratio and use it for computing
  // upweight instead of directly computing upweight as dist_dn/(dist_up+dist_dn)
  PetscReal ratio = dist_up/(dist_up + dist_dn);
  PetscReal up = (dist_up+dist_dn)*ratio;
  PetscReal dn = (dist_up+dist_dn) - up;
  *upweight = dn/(up+dn);

  ierr = TDyDotProduct(fvtpf->gravity, u_up2dn, dist_gravity); CHKERRQ(ierr);
  *dist_gravity *= (dist_up + dist_dn);

  PetscFunctionReturn(0);
}
