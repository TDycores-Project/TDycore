#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdympfao3Dcoreimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdympfao3Dutilsimpl.h>
#include <private/tdydmimpl.h>
#include <private/tdytiimpl.h>
#include <tdytimers.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdyioimpl.h>
#include <petscblaslapack.h>

const char *const TDyMethods[] = {
  "TPF",
  "MPFA_O",
  "MPFA_O_DAE",
  "MPFA_O_TRANSIENTVAR",
  "BDM",
  "WY",
  /* */
  "TDyMethod","TDY_METHOD_",NULL
};

const char *const TDyMPFAOGmatrixMethods[] = {
  "MPFAO_GMATRIX_DEFAULT",
  "MPFAO_GMATRIX_TPF",
  /* */
  "TDyMPFAOGmatrixMethod","TDY_MPFAO_GMATRIX_METHOD_",NULL
};

const char *const TDyMPFAOBoundaryConditionTypes[] = {
  "MPFAO_DIRICHLET_BC",
  "MPFAO_NEUMANN_BC",
  "MPFAO_SEEPAGE_BC",
  /* */
  "TDyMPFAOBoundaryConditionType","TDY_MPFAO_BC_TYPE_",NULL
};

const char *const TDyModes[] = {
  "RICHARDS",
  "TH",
  /* */
  "TDyMode","TDY_MODE_",NULL
};

const char *const TDyQuadratureTypes[] = {
  "LUMPED",
  "FULL",
  /* */
  "TDyQuadratureType","TDY_QUAD_",NULL
};

const char *const TDyWaterDensityTypes[] = {
  "CONSTANT",
  "EXPONENTIAL",
  /* */
  "TDyWaterDensityType","TDY_DENSITY_",NULL
};

PetscClassId TDY_CLASSID = 0;

static PetscBool TDyPackageInitialized = PETSC_FALSE;
PetscLogEvent TDy_ComputeSystem = 0;

static PetscErrorCode TDyInitSubsystems() {
  char           logList[256];
  PetscBool      opt,pkg;

  if (TDyPackageInitialized) PetscFunctionReturn(0);

  // Register a class ID for logging.
  PetscErrorCode ierr = PetscClassIdRegister("TDy",&TDY_CLASSID); CHKERRQ(ierr);

  TDyInitTimers();

  // Process info exclusions.
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),
                               &opt); CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("tdy",logList,',',&pkg); CHKERRQ(ierr);
    if (pkg) {
      ierr = PetscInfoDeactivateClass(TDY_CLASSID);
      CHKERRQ(ierr);
    }
  }

  // Process summary exclusions.
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),
                               &opt); CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("tdy",logList,',',&pkg); CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventDeactivateClass(TDY_CLASSID); CHKERRQ(ierr);}
  }

  // Enable timers if requested.
  PetscBool timersEnabled = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-tdy_timers", &timersEnabled, &opt);
  CHKERRQ(ierr);
  if (timersEnabled)
    TDyEnableTimers();

  TDyPackageInitialized = PETSC_TRUE;

  PetscFunctionReturn(0);
}

// This function initializes the TDycore library and its various subsystems.
PetscErrorCode TDyInit(int argc, char* argv[]) {
  PetscFunctionBegin;

  // Initialize PETSc if we haven't already.
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  // Initialize TDycore-specific subsystems.
  ierr = TDyInitSubsystems(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function initializes the TDycore library and its various subsystems
// without arguments. It's used by the Fortran interface, which calls
// PetscInitialize itself and then this function.
PetscErrorCode TDyInitNoArguments(void) {
  PetscFunctionBegin;
  if (TDyPackageInitialized) PetscFunctionReturn(0);

  // Initialize PETSc if we haven't already.
  PetscErrorCode ierr = PetscInitializeNoArguments(); CHKERRQ(ierr);

  // Initialize TDycore-specific subsystems.
  ierr = TDyInitSubsystems(); CHKERRQ(ierr);

  TDyPackageInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

// This function returns PETSC_TRUE if the TDyCore library has been initialized,
// PETSC_FALSE otherwise.
PetscBool TDyInitialized(void) {
  return TDyPackageInitialized;
}

// A registry of functions called at shutdown. This can be used by all
// subsystems via TDyOnFinalize().
typedef void (*ShutdownFunc)(void);
static ShutdownFunc *shutdown_funcs_ = NULL;
static int num_shutdown_funcs_ = 0;
static int shutdown_funcs_cap_ = 0;

/// Call this to register a shutdown function that is called during TDyFinalize.
PetscErrorCode TDyOnFinalize(void (*shutdown_func)(void)) {
  if (shutdown_funcs_ == NULL) {
    shutdown_funcs_cap_ = 32;
    shutdown_funcs_ = malloc(sizeof(ShutdownFunc) * shutdown_funcs_cap_);
  } else if (num_shutdown_funcs_ == shutdown_funcs_cap_) { // need more space!
    shutdown_funcs_cap_ *= 2;
    shutdown_funcs_ = realloc(shutdown_funcs_, sizeof(ShutdownFunc) * shutdown_funcs_cap_);
  }
  shutdown_funcs_[num_shutdown_funcs_] = shutdown_func;
  ++num_shutdown_funcs_;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyFinalize() {
  PetscFunctionBegin;

  // Call shutdown functions in reverse order, and destroy the list.
  if (shutdown_funcs_ != NULL) {
    for (int i = num_shutdown_funcs_-1; i >= 0; --i) {
      shutdown_funcs_[i]();
    }
    free(shutdown_funcs_);
  }

  // Finalize PETSc.
  PetscFinalize();

  TDyPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetOptionDefaults(TDyOptions *options){
  PetscFunctionBegin;
  PetscErrorCode ierr;

  options->gravity_constant = 9.8068;

  options->porosity=0.25;
  options->permeability=1.e-12;
  options->soil_density=2650.;
  options->soil_specific_heat=1000.0;
  options->thermal_conductivity=1.0;

  options->residual_saturation=0.15;
  options->gardner_n=0.5;
  options->vangenuchten_m=0.8;
  options->vangenuchten_alpha=1.e-4;

  PetscFunctionReturn(0);
}


PetscErrorCode TDyCreate(TDy *_tdy) {
  TDy            tdy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_tdy,1);
  *_tdy = NULL;

  // Initialize TDycore-specific subsystems.
  ierr = TDyInitSubsystems(); CHKERRQ(ierr);

  ierr = PetscHeaderCreate(tdy,TDY_CLASSID,"TDy","TDy","TDy",PETSC_COMM_WORLD,
                           TDyDestroy,TDyView); CHKERRQ(ierr);
  *_tdy = tdy;
  tdy->setupflags |= TDyCreated;

  ierr = TDySetOptionDefaults(&tdy->options);

  ierr = TDyIOCreate(&tdy->io); CHKERRQ(ierr);

  // initialize flags/parameters
  tdy->Pref = 101325;
  tdy->Tref = 25;
  tdy->gravity[0] = 0; tdy->gravity[1] = 0; tdy->gravity[2] = 0;
  tdy->rho_type = WATER_DENSITY_CONSTANT;
  tdy->mu_type = WATER_VISCOSITY_CONSTANT;
  tdy->enthalpy_type = WATER_ENTHALPY_CONSTANT;
  tdy->mpfao_gmatrix_method = MPFAO_GMATRIX_DEFAULT;
  tdy->mpfao_bc_type = MPFAO_DIRICHLET_BC;
  tdy->allow_unsuitable_mesh = PETSC_FALSE;
  tdy->init_with_random_field = PETSC_FALSE;
  tdy->init_from_file = PETSC_FALSE;

  /* initialize method information to null */
  tdy->vmap = NULL; tdy->emap = NULL; tdy->Alocal = NULL; tdy->Flocal = NULL;
  tdy->quad = NULL;
  tdy->faces = NULL; tdy->LtoG = NULL; tdy->orient = NULL;
  tdy->qtype = FULL;

  tdy->setupflags |= TDyParametersInitialized;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDM(TDy tdy, DM dm) {
  PetscFunctionBegin;
  if (!dm) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A DM must be created prior to TDySetDM()");
  }
  tdy->dm = dm;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyMalloc(TDy tdy) {
  PetscInt       dim,c,cStart,cEnd,nc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!tdy->dm) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"tdy->dm must be set prior to TDyMalloc()");
  }

  /* allocate space for a full tensor perm for each cell */
  PetscLogEvent t2 = TDyGetTimer("ComputePlexGeometry");
  TDyStartTimer(t2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nc   = cEnd-cStart;
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->rho)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->drho_dP)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->d2rho_dP2)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->vis)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->dvis_dP)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->d2vis_dP2)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->h)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->dh_dT)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->dh_dP)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->drho_dT)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->u)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->du_dP)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->du_dT)); CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(PetscReal),&(tdy->dvis_dT)); CHKERRQ(ierr);
  TDyStopTimer(t2);

  ierr = CharacteristicCurveCreate(nc, &tdy->cc); CHKERRQ(ierr);
  ierr = MaterialPropertiesCreate(dim, nc, &tdy->matprop); CHKERRQ(ierr);

  /* problem constants FIX: add mutators */
   CharacteristicCurve *cc = tdy->cc;
   TDyOptions *options = &tdy->options;

   PetscReal mualem_poly_low = 0.99;

  for (c=0; c<nc; c++) {
    cc->sr[c] = options->residual_saturation;
    cc->gardner_n[c] = options->gardner_n;
    cc->gardner_m[c] = options->vangenuchten_m;
    cc->vg_m[c] = options->vangenuchten_m;
    cc->mualem_poly_low[c] = mualem_poly_low;
    cc->mualem_m[c] = options->vangenuchten_m;
    cc->irmay_m[c] = options->vangenuchten_m;
    cc->vg_alpha[c] = options->vangenuchten_alpha;
    cc->SatFuncType[c] = SAT_FUNC_VAN_GENUCHTEN;
    cc->RelPermFuncType[c] = REL_PERM_FUNC_MUALEM;
    cc->Kr[c] = 0.0;
    cc->dKr_dS[c] = 0.0;
    cc->S[c] = 0.0;
    cc->dS_dP[c] = 0.0;
    tdy->rho[c] = 0.0;
    tdy->drho_dP[c] = 0.0;
    tdy->vis[c] = 0.0;
    tdy->dvis_dP[c] = 0.0;
    tdy->d2vis_dP2[c] = 0.0;
    tdy->h[c] = 0.0;
    tdy->dh_dT[c] = 0.0;
    tdy->dh_dP[c] = 0.0;
    tdy->drho_dT[c] = 0.0;
    cc->dS_dT[c] = 0.0;
    tdy->u[c] = 0.0;
    tdy->du_dP[c] = 0.0;
    tdy->du_dT[c] = 0.0;
    tdy->dvis_dT[c] = 0.0;
  }
  ierr = RelativePermeability_Mualem_SetupSmooth(tdy->cc, nc);

  tdy->gravity[dim-1] = options->gravity_constant;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateGrid(TDy tdy) {
  PetscInt       d,dim,p,pStart,pEnd,vStart,vEnd,eStart,eEnd,offset;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if ((tdy->setupflags & TDyOptionsSet) == 0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Options must be set prior to TDyCreateGrid()");
  }

  if (!tdy->dm) {
    DM dm;
    ierr = TDyCreateDM(&dm); CHKERRQ(ierr);
    ierr = TDyDistributeDM(&dm); CHKERRQ(ierr);
    tdy->dm = dm;
  }

  /* compute/store plex geometry */
  PetscLogEvent t1 = TDyGetTimer("ComputePlexGeometry");
  TDyStartTimer(t1);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetChart(tdy->dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(tdy->dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(tdy->dm,1,&eStart,&eEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(tdy->V));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(tdy->X));
  CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(tdy->N));
  CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(tdy->dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal (tdy->dm, &coordinates); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(p=pStart; p<pEnd; p++) {
    if((p >= vStart) && (p < vEnd)) {
      ierr = PetscSectionGetOffset(coordSection,p,&offset); CHKERRQ(ierr);
      for(d=0; d<dim; d++) tdy->X[p*dim+d] = coords[offset+d];
    } else {
      if((dim == 3) && (p >= eStart) && (p < eEnd)) continue;
      ierr = DMPlexComputeCellGeometryFVM(tdy->dm,p,&(tdy->V[p]),
                                          &(tdy->X[p*dim]),
                                          &(tdy->N[p*dim])); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  TDyStopTimer(t1);
  ierr = TDyMalloc(tdy); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetGravityVector(TDy tdy, PetscReal *gravity) {

  PetscInt d, dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  for (d=0;d<dim;d++) tdy->gravity[d] = gravity[d];

  PetscFunctionReturn(0);

}

PetscErrorCode TDyDestroy(TDy *_tdy) {
  TDy            tdy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_tdy,1);
  tdy = *_tdy; *_tdy = NULL;

  if (!tdy) PetscFunctionReturn(0);

  ierr = TDyResetDiscretizationMethod(tdy); CHKERRQ(ierr);
  ierr = PetscFree(tdy->V); CHKERRQ(ierr);
  ierr = PetscFree(tdy->X); CHKERRQ(ierr);
  ierr = PetscFree(tdy->N); CHKERRQ(ierr);
  ierr = PetscFree(tdy->rho); CHKERRQ(ierr);
  ierr = PetscFree(tdy->d2rho_dP2); CHKERRQ(ierr);
  ierr = PetscFree(tdy->vis); CHKERRQ(ierr);
  ierr = PetscFree(tdy->dvis_dP); CHKERRQ(ierr);
  ierr = PetscFree(tdy->d2vis_dP2); CHKERRQ(ierr);
  ierr = PetscFree(tdy->h); CHKERRQ(ierr);
  ierr = PetscFree(tdy->dh_dP); CHKERRQ(ierr);
  ierr = PetscFree(tdy->dh_dT); CHKERRQ(ierr);
  ierr = PetscFree(tdy->dvis_dT); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->residual); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->soln_prev); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->accumulation_prev); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->solution); CHKERRQ(ierr);
  ierr = TDyIODestroy(&tdy->io); CHKERRQ(ierr);
  ierr = TDyTimeIntegratorDestroy(&tdy->ti); CHKERRQ(ierr);
  ierr = DMDestroy(&tdy->dm); CHKERRQ(ierr);

  if (tdy->cc) {ierr = CharacteristicCurveDestroy(tdy->cc); CHKERRQ(ierr);}
  if (tdy->cc_bnd) {ierr = CharacteristicCurveDestroy(tdy->cc_bnd); CHKERRQ(ierr);}
  if (tdy->matprop) {ierr = MaterialPropertiesDestroy(tdy->matprop); CHKERRQ(ierr);}

  ierr = PetscFree(tdy); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetDimension(TDy tdy,PetscInt *dim) {
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  ierr = DMGetDimension(tdy->dm,dim); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetDM(TDy tdy,DM *dm) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  *dm = tdy->dm;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetCentroidArray(TDy tdy,PetscReal **X) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  *X = tdy->X;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyResetDiscretizationMethod(TDy tdy) {
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  if (tdy->vmap  ) { ierr = PetscFree(tdy->vmap  ); CHKERRQ(ierr); }
  if (tdy->emap  ) { ierr = PetscFree(tdy->emap  ); CHKERRQ(ierr); }
  if (tdy->Alocal) { ierr = PetscFree(tdy->Alocal); CHKERRQ(ierr); }
  if (tdy->Flocal) { ierr = PetscFree(tdy->Flocal); CHKERRQ(ierr); }
  if (tdy->vel   ) { ierr = PetscFree(tdy->vel   ); CHKERRQ(ierr); }
  if (tdy->fmap  ) { ierr = PetscFree(tdy->fmap  ); CHKERRQ(ierr); }
  if (tdy->faces ) { ierr = PetscFree(tdy->faces ); CHKERRQ(ierr); }
  if (tdy->LtoG  ) { ierr = PetscFree(tdy->LtoG  ); CHKERRQ(ierr); }
  if (tdy->orient) { ierr = PetscFree(tdy->orient); CHKERRQ(ierr); }
  if (tdy->quad  ) { ierr = PetscQuadratureDestroy(&(tdy->quad)); CHKERRQ(ierr); }

  // Need call to destroy TDyMesh
  switch (dim) {
  case 2:
    break;
  case 3:
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unsupported dim in TDyResetDiscretizationMethod");
    break;
  }
  // if (tdy->subc_Gmatrix) { ierr = TDyDeallocate_RealArray_4D(&tdy->subc_Gmatrix, tdy->mesh->num_cells,
  //                                   nsubcells, nrow, ncol); CHKERRQ(ierr); }
  // if (tdy->Trans       ) { ierr = TDyDeallocate_RealArray_3D(&tdy->Trans,
  //                                   tdy->mesh->num_vertices, 12, 12); CHKERRQ(ierr); }
  // if (tdy->Trans_mat   ) { ierr = MatDestroy(&tdy->Trans_mat  ); CHKERRQ(ierr); }
  if (tdy->P_vec       ) { ierr = VecDestroy(&tdy->P_vec      ); CHKERRQ(ierr); }
  if (tdy->TtimesP_vec ) { ierr = VecDestroy(&tdy->TtimesP_vec); CHKERRQ(ierr); }
  // if (tdy->Temp_subc_Gmatrix) { ierr = TDyDeallocate_RealArray_4D(&tdy->Temp_subc_Gmatrix,
  //                                        tdy->mesh->num_cells,
  //                                        nsubcells, nrow, ncol); CHKERRQ(ierr); }
  // if (tdy->Temp_Trans       ) { ierr = TDyDeallocate_RealArray_3D(&tdy->Temp_Trans,
  //                                        tdy->mesh->num_vertices, 12, 12); CHKERRQ(ierr); }
  if (tdy->Temp_Trans_mat   ) { ierr = MatDestroy(&tdy->Temp_Trans_mat  ); CHKERRQ(ierr); }
  if (tdy->Temp_P_vec       ) { ierr = VecDestroy(&tdy->Temp_P_vec      ); CHKERRQ(ierr); }
  if (tdy->Temp_TtimesP_vec ) { ierr = VecDestroy(&tdy->Temp_TtimesP_vec); CHKERRQ(ierr); }
  if (tdy->J           ) { ierr = MatDestroy(&tdy->J   ); CHKERRQ(ierr); }
  if (tdy->Jpre        ) { ierr = MatDestroy(&tdy->Jpre); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyView(TDy tdy,PetscViewer viewer) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)tdy)->comm,&viewer); CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tdy,1,viewer,2);
  PetscFunctionReturn(0);
}

/// Sets options for the dycore based on command line arguments supplied by a
/// user. TDySetFromOptions must be called before TDySetupNumericalMethods,
/// since the latter uses options specified by the former.
/// @param tdy The dycore instance
PetscErrorCode TDySetFromOptions(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if ((tdy->setupflags & TDySetupFinished) != 0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"TDySetFromOptions must be called prior to TDySetupNumericalMethods()");
  }

  TDyOptions *options = &tdy->options;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)tdy); CHKERRQ(ierr);
  PetscBool flag;

  // Material property options
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: Material property options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_porosity", "Value of porosity", NULL, options->porosity, &options->porosity, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_permability", "Value of permeability", NULL, options->permeability, &options->permeability, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_soil_density", "Value of soil density", NULL, options->soil_density, &options->soil_density, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_soil_specific_heat", "Value of soil specific heat", NULL, options->soil_specific_heat, &options->soil_specific_heat, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_thermal_conductivity", "Value of thermal conductivity", NULL, options->porosity, &options->porosity, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Characteristic curve options
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: Characteristic curve options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_residual_satuaration", "Value of residual saturation", NULL, options->residual_saturation, &options->residual_saturation, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_gardner_param_n", "Value of Gardner n parameter", NULL, options->gardner_n, &options->gardner_n, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_vangenuchten_param_m", "Value of VanGenuchten m parameter", NULL, options->vangenuchten_m, &options->vangenuchten_m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_vangenuchten_param_alpha", "Value of VanGenuchten alpha parameter", NULL, options->vangenuchten_alpha, &options->vangenuchten_alpha, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Model options
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: Model options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_gravity", "Magnitude of gravity vector", NULL,
                          options->gravity_constant, &options->gravity_constant,
                          NULL); CHKERRQ(ierr);
  TDyWaterDensityType densitytype = WATER_DENSITY_CONSTANT;
  ierr = PetscOptionsEnum("-tdy_water_density","Water density vertical profile",
                          "TDySetWaterDensityType",TDyWaterDensityTypes,
                          (PetscEnum)densitytype,(PetscEnum *)&densitytype,
                          &flag); CHKERRQ(ierr);
  if (flag) {
    ierr = TDySetWaterDensityType(tdy,densitytype); CHKERRQ(ierr);
  }
  TDyMode mode = RICHARDS;
  ierr = PetscOptionsEnum("-tdy_mode","Flow mode",
                          "TDySetMode",TDyModes,(PetscEnum)mode,(PetscEnum *)&mode,
                          &flag); CHKERRQ(ierr);

  if (flag && (mode != tdy->mode)) {
    ierr = TDySetMode(tdy,mode); CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-tdy_pressure_bc",
                          "Assigns a named pressure function to the domain boundary",
                          NULL, options->gravity_constant, &options->gravity_constant,
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Numerics options
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TDyCore: Numerics options",""); CHKERRQ(ierr);
  TDyMethod method = WY;
  ierr = PetscOptionsEnum("-tdy_method","Discretization method",
                          "TDySetDiscretizationMethod",TDyMethods,
                          (PetscEnum)method,(PetscEnum *)&method, &flag); CHKERRQ(ierr);
  if (flag && (method != tdy->method)) {
    ierr = TDySetDiscretizationMethod(tdy,method); CHKERRQ(ierr);
  }
  TDyQuadratureType qtype = FULL;
  ierr = PetscOptionsEnum("-tdy_quadrature","Quadrature type for finite element methods",
                          "TDySetQuadratureType",TDyQuadratureTypes,(PetscEnum)qtype,(PetscEnum *)&qtype,
                          &flag); CHKERRQ(ierr);
  if (flag && (qtype != tdy->qtype)) {
    ierr = TDySetQuadratureType(tdy,qtype); CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-tdy_tpf_allow_all_meshes",
                          "Enable to allow non-orthgonal meshes in finite volume TPF method",
                          "",tdy->allow_unsuitable_mesh,
                          &(tdy->allow_unsuitable_mesh),NULL); CHKERRQ(ierr);
  TDyMPFAOGmatrixMethod gmatrixmethod = MPFAO_GMATRIX_DEFAULT;
  ierr = PetscOptionsEnum("-tdy_mpfao_gmatrix_method","MPFA-O gmatrix method",
                          "TDySetMPFAOGmatrixMethod",TDyMPFAOGmatrixMethods,(PetscEnum)gmatrixmethod,(PetscEnum *)&gmatrixmethod,
                          &flag); CHKERRQ(ierr);
  if (flag && (gmatrixmethod != tdy->mpfao_gmatrix_method)) {
    ierr = TDySetMPFAOGmatrixMethod(tdy,gmatrixmethod); CHKERRQ(ierr);
  }
  TDyMPFAOBoundaryConditionType bctype = MPFAO_DIRICHLET_BC;
  ierr = PetscOptionsEnum("-tdy_mpfao_boundary_condition_type","MPFA-O boundary condition type",
                          "TDySetMPFAOBoundaryConditionType",TDyMPFAOBoundaryConditionTypes,(PetscEnum)bctype,(PetscEnum *)&bctype,
                          &flag); CHKERRQ(ierr);
  if (flag && (bctype != tdy->mpfao_bc_type)) {
    ierr = TDySetMPFAOBoundaryConditionType(tdy,bctype); CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = PetscOptionsBool("-tdy_init_with_random_field",
                          "Initialize solution with a random field","",
                          tdy->init_with_random_field,
                          &(tdy->init_with_random_field),NULL); CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_init_file", tdy->init_file,
                               sizeof(tdy->init_file),
                               &tdy->init_from_file); CHKERRQ(ierr);

  ierr = PetscOptionsBool("-tdy_regression_test",
                          "Enable output of a regression file","",tdy->regression_testing,
                          &(tdy->regression_testing),NULL); CHKERRQ(ierr);

  ierr = PetscOptionsBool("-tdy_output_mesh",
                          "Enable output of mesh attributes","",tdy->output_mesh,
                          &(tdy->output_mesh),NULL); CHKERRQ(ierr);

  // Wrap up and indicate that options are set.
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  tdy->setupflags |= TDyOptionsSet;

  if (tdy->init_from_file && tdy->init_with_random_field) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Both -tdy_init_from_file and -tdy_init_with_random_field cannot be specified");
  }

  ierr = TDyCreateGrid(tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetupDiscretizationScheme(TDy tdy) {
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)(tdy->dm),&comm); CHKERRQ(ierr);
  switch (tdy->method) {
  case TPF:
    ierr = TDyTPFInitialize(tdy); CHKERRQ(ierr);
    break;
  case MPFA_O:
    ierr = TDyMPFAOInitialize(tdy); CHKERRQ(ierr);
    break;
  case MPFA_O_DAE:
    ierr = TDyMPFAOInitialize(tdy); CHKERRQ(ierr);
    break;
  case MPFA_O_TRANSIENTVAR:
    ierr = TDyMPFAOInitialize(tdy); CHKERRQ(ierr);
    break;
  case BDM:
    ierr = TDyBDMInitialize(tdy); CHKERRQ(ierr);
    break;
  case WY:
    ierr = TDyWYInitialize(tdy); CHKERRQ(ierr);
    break;
  }

  // Record metadata for scaling studies.
  TDySetTimingMetadata(tdy);

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetupNumericalMethods(TDy tdy) {
  /* must follow TDySetFromOptions() is it relies upon options set by
     TDySetFromOptions */
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if ((tdy->setupflags & TDyOptionsSet) == 0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"TDySetFromOptions must be called prior to TDySetupNumericalMethods()");
  }
  TDY_START_FUNCTION_TIMER()
  TDyEnterProfilingStage("TDycore Setup");
  ierr = TDySetupDiscretizationScheme(tdy); CHKERRQ(ierr);
  if (tdy->regression_testing) {
    /* must come after Sections are set up in
       TDySetupDiscretizationScheme->XXXInitialize */
    ierr = TDyRegressionInitialize(tdy); CHKERRQ(ierr);
  }
  if (tdy->output_mesh) {
    if (tdy->method != MPFA_O) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,
              "-tdy_output_mesh only supported for MPFA-O method");
    }
    ierr = TDyOutputMesh(tdy); CHKERRQ(ierr);
  }
  TDyExitProfilingStage("TDycore Setup");
  tdy->setupflags |= TDySetupFinished;
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDiscretizationMethod(TDy tdy,TDyMethod method) {
  PetscFunctionBegin;
  tdy->method = method;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetMode(TDy tdy,TDyMode mode) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  tdy->mode = mode;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetQuadratureType(TDy tdy,TDyQuadratureType qtype) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  tdy->qtype = qtype;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetWaterDensityType(TDy tdy,TDyWaterDensityType dentype) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  switch (dentype) {
  case WATER_DENSITY_CONSTANT:
    tdy->rho_type = WATER_DENSITY_CONSTANT;
    break;
  case WATER_DENSITY_EXPONENTIAL:
    tdy->rho_type = WATER_DENSITY_EXPONENTIAL;
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetMPFAOGmatrixMethod(TDy tdy,TDyMPFAOGmatrixMethod method) {
  PetscFunctionBegin;

  PetscValidPointer(tdy,1);
  tdy->mpfao_gmatrix_method = method;

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetMPFAOBoundaryConditionType(TDy tdy,TDyMPFAOBoundaryConditionType bctype) {
  PetscFunctionBegin;

  PetscValidPointer(tdy,1);
  tdy->mpfao_bc_type = bctype;

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetIFunction(TS ts,TDy tdy) {
  PetscInt       dim, num_fields;
  MPI_Comm       comm;
  DM             dm;
  PetscSection   sec;
  PetscErrorCode ierr;
  PetscValidPointer( ts,1);
  PetscValidPointer(tdy,2);

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)ts,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  ierr = DMGetSection(tdy->dm, &sec);
  ierr = PetscSectionGetNumFields(sec, &num_fields);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  switch (tdy->method) {
  case TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for TPF");
    break;
  case MPFA_O:
    switch (dim) {
    case 3:
      switch (tdy->mode) {
      case RICHARDS:
        ierr = TSSetIFunction(ts,NULL,TDyMPFAOIFunction_3DMesh,tdy); CHKERRQ(ierr);
        break;
      case TH:
        ierr = TSSetIFunction(ts,NULL,TDyMPFAOIFunction_3DMesh_TH,tdy); CHKERRQ(ierr);
        break;
      }
    break;
    default :
      SETERRQ(comm,PETSC_ERR_SUP,"IFunction only implemented for 3D problem MPFA-O");
      break;
    }
    break;
  case MPFA_O_DAE:
    switch (dim) {
    case 3:
      ierr = TSSetIFunction(ts,NULL,TDyMPFAOIFunction_DAE_3DMesh,tdy); CHKERRQ(ierr);
      break;
    default :
      SETERRQ(comm,PETSC_ERR_SUP,"IFunction only implemented for 3D problem MPFA-O");
      break;
    }
    break;
  case MPFA_O_TRANSIENTVAR:
    switch (dim) {
    case 3:
      ierr = DMTSSetIFunction(dm,TDyMPFAOIFunction_TransientVariable_3DMesh,tdy); CHKERRQ(ierr);
      ierr = DMTSSetTransientVariable(dm,TDyMPFAOTransientVariable_3DMesh,tdy); CHKERRQ(ierr);
      break;
    default :
      SETERRQ(comm,PETSC_ERR_SUP,"IFunction only implemented for 3D problem MPFA-O");
      break;
    }
    break;
  case BDM:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for BDM");
    break;
  case WY:
    ierr = TSSetIFunction(ts,NULL,TDyWYResidual,tdy); CHKERRQ(ierr);
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetIJacobian(TS ts,TDy tdy) {
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscValidPointer( ts,1);
  PetscValidPointer(tdy,2);
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)ts,&comm); CHKERRQ(ierr);
  switch (tdy->method) {
  case TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"IJacobian not implemented for TPF");
    break;
  case MPFA_O:
    ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);
    switch (tdy->mode) {
    case RICHARDS:
      ierr = TSSetIJacobian(ts,tdy->J,tdy->J,TDyMPFAOIJacobian_3DMesh,tdy); CHKERRQ(ierr);
      break;
    case TH:
      ierr = TSSetIJacobian(ts,tdy->J,tdy->J,TDyMPFAOIJacobian_3DMesh_TH,tdy); CHKERRQ(ierr);
      break;

    }
    break;
  case MPFA_O_DAE:
    ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);
    break;

  case MPFA_O_TRANSIENTVAR:
    ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);
    break;

  case BDM:
    SETERRQ(comm,PETSC_ERR_SUP,"IJacobian not implemented for BDM");
    break;
  case WY:
    SETERRQ(comm,PETSC_ERR_SUP,"IJacobian not implemented for WY");
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetSNESFunction(SNES snes,TDy tdy) {
  PetscInt       dim;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscValidPointer(snes,1);
  PetscValidPointer(tdy,2);

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)snes,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  switch (tdy->method) {
  case TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESFunction not implemented for TPF");
    break;
  case MPFA_O:
    switch (dim) {
    case 3:
      ierr = SNESSetFunction(snes,tdy->residual,TDyMPFAOSNESFunction_3DMesh,tdy); CHKERRQ(ierr);
      break;
    default :
      SETERRQ(comm,PETSC_ERR_SUP,"SNESFunction only implemented for 3D problem MPFA-O");
      break;
    }
    break;
  case MPFA_O_DAE:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESFunction not implemented for MPFA_O_DAE");
    break;
  case MPFA_O_TRANSIENTVAR:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESFunction not implemented for MPFA_O_TRANSIENTVAR");
    break;
  case BDM:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESFunction not implemented for BDM");
    break;
  case WY:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESFunction not implemented for WY");
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetSNESJacobian(SNES snes,TDy tdy) {
  PetscInt       dim;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscValidPointer(snes,1);
  PetscValidPointer(tdy,2);

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)snes,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  switch (tdy->method) {
  case TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESJacobian not implemented for TPF");
    break;
  case MPFA_O:
    switch (dim) {
    case 3:
      ierr = SNESSetJacobian(snes,tdy->J,tdy->J,TDyMPFAOSNESJacobian_3DMesh,tdy); CHKERRQ(ierr);
      break;
    default :
      SETERRQ(comm,PETSC_ERR_SUP,"SNESJacobian only implemented for 3D problem MPFA-O");
      break;
    }
    break;
  case MPFA_O_DAE:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESJacobian not implemented for MPFA_O_DAE");
    break;
  case MPFA_O_TRANSIENTVAR:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESJacobian not implemented for MPFA_O_TRANSIENTVAR");
    break;
  case BDM:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESJacobian not implemented for BDM");
    break;
  case WY:
    SETERRQ(comm,PETSC_ERR_SUP,"SNESJacobian not implemented for WY");
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyComputeSystem(TDy tdy,Mat K,Vec F) {
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)(tdy->dm),&comm); CHKERRQ(ierr);
  switch (tdy->method) {
  case TPF:
    ierr = TDyTPFComputeSystem(tdy,K,F); CHKERRQ(ierr);
    break;
  case MPFA_O:
    ierr = TDyMPFAOComputeSystem(tdy,K,F); CHKERRQ(ierr);
    break;
  case MPFA_O_DAE:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyComputeSystem not implemented for MPFA_O_DAE");
    break;
  case MPFA_O_TRANSIENTVAR:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyComputeSystem not implemented for MPFA_O_TRANSIENTVAR");
    break;
  case BDM:
    ierr = TDyBDMComputeSystem(tdy,K,F); CHKERRQ(ierr);
    break;
  case WY:
    ierr = TDyWYComputeSystem(tdy,K,F); CHKERRQ(ierr);
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyUpdateState(TDy tdy,PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyEnterProfilingStage("TDycore Setup");
  TDY_START_FUNCTION_TIMER()
  PetscInt  dim,dim2,i,j,c,cStart,cEnd;
  PetscReal Se,dSe_dS,dKr_dSe,Kr;
  PetscReal *P, *temp;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&P);CHKERRQ(ierr);
  ierr = PetscMalloc((cEnd-cStart)*sizeof(PetscReal),&temp);CHKERRQ(ierr);

  if (tdy->mode == TH) {
    for (c=0;c<cEnd-cStart;c++) {
      P[c] = U[c*2];
      temp[c] = U[c*2+1];
    }
  }
  else {
    for (c=0;c<cEnd-cStart;c++) P[c] = U[c];
  }

  CharacteristicCurve *cc = tdy->cc;
  MaterialProp *matprop = tdy->matprop;

  for(c=cStart; c<cEnd; c++) {
    i = c-cStart;

    PetscReal n = cc->gardner_n[c];
    PetscReal alpha = cc->vg_alpha[c];

    switch (cc->SatFuncType[i]) {
    case SAT_FUNC_GARDNER :
      PressureSaturation_Gardner(n,cc->gardner_m[c],alpha,cc->sr[i],tdy->Pref-P[i],&(cc->S[i]),&(cc->dS_dP[i]),&(cc->d2S_dP2[i]));
      break;
    case SAT_FUNC_VAN_GENUCHTEN :
      PressureSaturation_VanGenuchten(cc->vg_m[c],alpha,cc->sr[i],tdy->Pref-P[i],&(cc->S[i]),&cc->dS_dP[i],&(cc->d2S_dP2[i]));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown saturation function");
      break;
    }

    Se = (cc->S[i] - cc->sr[i])/(1.0 - cc->sr[i]);
    dSe_dS = 1.0/(1.0 - cc->sr[i]);

    switch (cc->RelPermFuncType[i]) {
    case REL_PERM_FUNC_IRMAY :
      RelativePermeability_Irmay(cc->irmay_m[c],Se,&Kr,NULL);
      break;
    case REL_PERM_FUNC_MUALEM :
      RelativePermeability_Mualem(cc->mualem_m[c],cc->mualem_poly_low[c],cc->mualem_poly_coeffs[c],Se,&Kr,&dKr_dSe);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown relative permeability function");
      break;
    }
    cc->Kr[i] = Kr;
    cc->dKr_dS[i] = dKr_dSe * dSe_dS;

    for(j=0; j<dim2; j++) matprop->K[i*dim2+j] = matprop->K0[i*dim2+j] * Kr;

    ierr = ComputeWaterDensity(P[i], tdy->rho_type, &(tdy->rho[i]), &(tdy->drho_dP[i]), &(tdy->d2rho_dP2[i])); CHKERRQ(ierr);
    ierr = ComputeWaterViscosity(P[i], tdy->mu_type, &(tdy->vis[i]), &(tdy->dvis_dP[i]), &(tdy->d2vis_dP2[i])); CHKERRQ(ierr);
    if (tdy->mode ==  TH) {
      for(j=0; j<dim2; j++) matprop->Kappa[i*dim2+j] = matprop->Kappa0[i*dim2+j]; // update this based on Kersten number, etc.
      ierr = ComputeWaterEnthalpy(temp[i], P[i], tdy->enthalpy_type, &(tdy->h[i]), &(tdy->dh_dP[i]), &(tdy->dh_dT[i])); CHKERRQ(ierr);
      tdy->u[i] = tdy->h[i] - P[i]/tdy->rho[i];
    }
  }

  if ( (tdy->method == MPFA_O || tdy->method == MPFA_O_DAE || tdy->method == MPFA_O_TRANSIENTVAR) && dim == 3) {
    PetscReal *p_vec_ptr, gz;
    TDyMesh *mesh = tdy->mesh;
    TDyCell *cells = &mesh->cells;

    ierr = VecGetArray(tdy->P_vec,&p_vec_ptr); CHKERRQ(ierr);
    for (c=cStart; c<cEnd; c++) {
      i = c-cStart;
      ierr = ComputeGtimesZ(tdy->gravity,cells->centroid[i].X,dim,&gz);
      p_vec_ptr[i] = P[i];
    }
    ierr = VecRestoreArray(tdy->P_vec,&p_vec_ptr); CHKERRQ(ierr);

    if (tdy->mode == TH) {
      PetscReal *t_vec_ptr;
      ierr = VecGetArray(tdy->Temp_P_vec, &t_vec_ptr); CHKERRQ(ierr);
      for (c=cStart; c<cEnd; c++) {
        i = c-cStart;
        t_vec_ptr[i] = temp[i];
      }
      ierr = VecRestoreArray(tdy->Temp_P_vec, &t_vec_ptr); CHKERRQ(ierr);
    }


  }

  TDY_STOP_FUNCTION_TIMER()
  TDyExitProfilingStage("TDycore Setup");
  PetscFunctionReturn(0);
}

PetscErrorCode TDyQuadrature(PetscQuadrature q,PetscInt dim) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *x,*w;
  PetscInt d,nv=1;
  for(d=0; d<dim; d++) nv *= 2;
  ierr = PetscMalloc1(nv*dim,&x); CHKERRQ(ierr);
  ierr = PetscMalloc1(nv,&w); CHKERRQ(ierr);
  switch(nv*dim) {
  case 2: /* line */
    x[0] = -1.0; w[0] = 1.0;
    x[1] =  1.0; w[1] = 1.0;
    break;
  case 8: /* quad */
    x[0] = -1.0; x[1] = -1.0; w[0] = 1.0;
    x[2] =  1.0; x[3] = -1.0; w[1] = 1.0;
    x[4] = -1.0; x[5] =  1.0; w[2] = 1.0;
    x[6] =  1.0; x[7] =  1.0; w[3] = 1.0;
    break;
  case 24: /* hex */
    x[0]  = -1.0; x[1]  = -1.0; x[2]  = -1.0; w[0] = 1.0;
    x[3]  =  1.0; x[4]  = -1.0; x[5]  = -1.0; w[1] = 1.0;
    x[6]  = -1.0; x[7]  =  1.0; x[8]  = -1.0; w[2] = 1.0;
    x[9]  =  1.0; x[10] =  1.0; x[11] = -1.0; w[3] = 1.0;
    x[12] = -1.0; x[13] = -1.0; x[14] =  1.0; w[4] = 1.0;
    x[15] =  1.0; x[16] = -1.0; x[17] =  1.0; w[5] = 1.0;
    x[18] = -1.0; x[19] =  1.0; x[20] =  1.0; w[6] = 1.0;
    x[21] =  1.0; x[22] =  1.0; x[23] =  1.0; w[7] = 1.0;
  }
  ierr = PetscQuadratureSetData(q,dim,1,nv,x,w); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscInt TDyGetNumberOfCellVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,c,q,i,cStart,cEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nq = -1;
  for(c=cStart; c<cEnd; c++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    q = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh cells must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

PetscInt TDyGetNumberOfFaceVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,f,q,i,fStart,fEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  nq = -1;
  for(f=fStart; f<fEnd; f++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    q = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh faces must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

/* Returns

   |x-y|_L1

   where x, and y are dim-dimensional arrays
 */
PetscReal TDyL1norm(PetscReal *x,PetscReal *y,PetscInt dim) {
  PetscInt i;
  PetscReal norm;
  norm = 0;
  for(i=0; i<dim; i++) norm += PetscAbsReal(x[i]-y[i]);
  return norm;
}

/* Returns

   a * (b - c)

   where a, b, and c are dim-dimensional arrays
 */
PetscReal TDyADotBMinusC(PetscReal *a,PetscReal *b,PetscReal *c,PetscInt dim) {
  PetscInt i;
  PetscReal norm;
  norm = 0;
  for(i=0; i<dim; i++) norm += a[i]*(b[i]-c[i]);
  return norm;
}

PetscReal TDyADotB(PetscReal *a,PetscReal *b,PetscInt dim) {
  PetscInt i;
  PetscReal norm = 0;
  for(i=0; i<dim; i++) norm += a[i]*b[i];
  return norm;
}

/* Check if the image of the quadrature point is coincident with
   the vertex, if so we create a map:

   map(cell,local_cell_vertex) --> vertex

   Allocates memory inside routine, user must free.
*/
PetscErrorCode TDyCreateCellVertexMap(TDy tdy,PetscInt **map) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt dim,i,v,vStart,vEnd,nv,c,cStart,cEnd,closureSize,*closure;
  PetscQuadrature quad;
  PetscReal x[24],DF[72],DFinv[72],J[8];
  DM dm = tdy->dm;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  nv = tdy->ncv;
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quad); CHKERRQ(ierr);
  ierr = TDyQuadrature(quad,dim); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(nv*(cEnd-cStart)*sizeof(PetscInt),map); CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<nv*(cEnd-cStart); c++) { (*map)[c] = -1; }
#endif
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quad,x,DF,DFinv,J); CHKERRQ(ierr);
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    for(v=0; v<nv; v++) {
      for (i=0; i<closureSize*2; i+=2) {
        if ((closure[i] >= vStart) && (closure[i] < vEnd)) {
          if (TDyL1norm(&(x[v*dim]),&(tdy->X[closure[i]*dim]),dim) > 1e-12) continue;
          (*map)[c*nv+v] = closure[i];
          break;
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<nv*(cEnd-cStart); c++) {
    if((*map)[c]<0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,local_vertex) -> vertex");
    }
  }
#endif
  ierr = PetscQuadratureDestroy(&quad); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* Create a map:

   map(cell,local_cell_vertex,direction) --> face

   To do this, I loop over the vertices of this cell and find
   connected faces. Then I use the local ordering of the vertices to
   determine where the normal of this face points. Finally I check if
   the normal points into the cell. If so, then the index is given a
   negative as a flag later in the assembly process. Since the Hasse
   diagram always begins with cells, there isn't a conflict with 0
   being a possible point.
*/
PetscErrorCode TDyCreateCellVertexDirFaceMap(TDy tdy,PetscInt **map) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt d,dim,i,f,fStart,fEnd,v,nv,q,c,cStart,cEnd,closureSize,*closure,
           fclosureSize,*fclosure,local_dirs[24];
  DM dm = tdy->dm;
  if(!(tdy->vmap)) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must first create TDyCreateCellVertexMap on tdy->vmap");
  }
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  if(dim == 2) {
    local_dirs[0] = 2; local_dirs[1] = 1;
    local_dirs[2] = 3; local_dirs[3] = 0;
    local_dirs[4] = 0; local_dirs[5] = 3;
    local_dirs[6] = 1; local_dirs[7] = 2;
  } else if(dim == 3) {
    local_dirs[0]  = 6; local_dirs[1]  = 5; local_dirs[2]  = 3;
    local_dirs[3]  = 7; local_dirs[4]  = 4; local_dirs[5]  = 2;
    local_dirs[6]  = 4; local_dirs[7]  = 7; local_dirs[8]  = 1;
    local_dirs[9]  = 5; local_dirs[10] = 6; local_dirs[11] = 0;
    local_dirs[12] = 2; local_dirs[13] = 1; local_dirs[14] = 7;
    local_dirs[15] = 3; local_dirs[16] = 0; local_dirs[17] = 6;
    local_dirs[18] = 0; local_dirs[19] = 3; local_dirs[20] = 5;
    local_dirs[21] = 1; local_dirs[22] = 2; local_dirs[23] = 4;
  }
  nv = tdy->ncv;
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(dim*nv*(cEnd-cStart)*sizeof(PetscInt),map); CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<dim*nv*(cEnd-cStart); c++) { (*map)[c] = 0; }
#endif
  for(c=cStart; c<cEnd; c++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    for(q=0; q<nv; q++) {
      for (i=0; i<closureSize*2; i+=2) {
        if ((closure[i] >= fStart) && (closure[i] < fEnd)) {
          fclosure = NULL;
          ierr = DMPlexGetTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,
                                            &fclosure); CHKERRQ(ierr);
          for(f=0; f<fclosureSize*2; f+=2) {
            if (fclosure[f] == tdy->vmap[c*nv+q]) {
              for(v=0; v<fclosureSize*2; v+=2) {
                for(d=0; d<dim; d++) {
                  if (fclosure[v] == tdy->vmap[c*nv+local_dirs[q*dim+d]]) {
                    (*map)[c*nv*dim+q*dim+d] = closure[i];
                    if (TDyADotBMinusC(&(tdy->N[closure[i]*dim]),&(tdy->X[closure[i]*dim]),
                                       &(tdy->X[c*dim]),dim) < 0) {
                      (*map)[c*nv*dim+q*dim+d] *= -1;
                      break;
                    }
                  }
                }
              }
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,
                                                &fclosure); CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<dim*nv*(cEnd-cStart); c++) {
    if((*map)[c]==0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,local_vertex,dir) -> face");
    }
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode TDyComputeErrorNorms(TDy tdy,Vec U,PetscReal *normp,
                                    PetscReal *normv) {
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)(tdy->dm),&comm); CHKERRQ(ierr);
  switch (tdy->method) {
  case TPF:
    if(normp != NULL) { *normp = TDyTPFPressureNorm(tdy,U); }
    if(normv != NULL) { *normv = TDyTPFVelocityNorm(tdy,U); }
    break;
  case MPFA_O:
    if(normv) {
      ierr = TDyMPFAORecoverVelocity(tdy,U); CHKERRQ(ierr);
    }
    if(normp != NULL) { *normp = TDyMPFAOPressureNorm(tdy,U); }
    if(normv != NULL) { *normv = TDyMPFAOVelocityNorm(tdy); }
    break;
  case MPFA_O_DAE:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyComputeErrorNorms not implemented for MPFA_O_DAE");
    break;
  case MPFA_O_TRANSIENTVAR:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyComputeErrorNorms not implemented for MPFA_O_TRANSIENTVAR");
    break;
  case BDM:
    if(normp != NULL) { *normp = TDyBDMPressureNorm(tdy,U); }
    if(normv != NULL) { *normv = TDyBDMVelocityNorm(tdy,U); }
    break;
  case WY:
    if(normv) {
      ierr = TDyWYRecoverVelocity(tdy,U); CHKERRQ(ierr);
    }
    if(normp != NULL) { *normp = TDyWYPressureNorm(tdy,U); }
    if(normv != NULL) { *normv = TDyWYVelocityNorm(tdy); }
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyOutputRegression(TDy tdy, Vec U) {

  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tdy->regression_testing){
    ierr = TDyRegressionOutput(tdy,U); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetDtimeForSNESSolver(TDy tdy, PetscReal dtime) {

  PetscFunctionBegin;
  tdy->dtime = dtime;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Sets initial condition for the TDy solver
///
/// @param [inout] tdy A TDy struct
/// @param [in] initial A PETSc vector that is copied as the intial condition.
///                     For RICHARDS mode, the vector contains unknown
///                     pressure values for each grid cell.
///                     For TH mode, the vector contains unknown pressure
///                     and temperature values for each grid cell.
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDySetInitialCondition(TDy tdy, Vec initial) {

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(initial,tdy->solution); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPreviousSolutionForSNESSolver(TDy tdy, Vec soln) {

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(soln,tdy->soln_prev); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyPreSolveSNESSolver(TDy tdy) {
  PetscInt dim;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = PetscObjectGetComm((PetscObject)tdy->dm,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  switch (tdy->method) {
  case TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for TPF");
    break;
  case MPFA_O:
    switch (dim) {
    case 3:
      ierr = TDyMPFAOSNESPreSolve_3DMesh(tdy); CHKERRQ(ierr);
      break;
    default :
      SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver only implemented for 3D problem MPFA-O");
      break;
    }
    break;
  case MPFA_O_DAE:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for MPFA_O_DAE");
    break;
  case MPFA_O_TRANSIENTVAR:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for MPFA_O_TRANSIENTVAR");
    break;
  case BDM:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for BDM");
    break;
  case WY:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for WY");
    break;
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyPostSolveSNESSolver(TDy tdy,Vec U) {

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(U,tdy->soln_prev); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateVectors(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = DMCreateGlobalVector(tdy->dm,&tdy->solution); CHKERRQ(ierr);
  ierr = VecDuplicate(tdy->solution,&tdy->residual); CHKERRQ(ierr);
  ierr = VecDuplicate(tdy->solution,&tdy->accumulation_prev); CHKERRQ(ierr);
  ierr = VecDuplicate(tdy->solution,&tdy->soln_prev); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateJacobian(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = DMCreateMatrix(tdy->dm,&tdy->J); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->J,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->J,MAT_ROW_ORIENTED,PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->J,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->J,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE); CHKERRQ(ierr);
  ierr = DMCreateMatrix(tdy->dm,&tdy->Jpre); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->Jpre,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->Jpre,MAT_ROW_ORIENTED,PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->Jpre,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(tdy->Jpre,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

// Here's a registry of C-backed strings created from Fortran.
KHASH_SET_INIT_STR(TDY_STRING_SET)
static khash_t(TDY_STRING_SET)* fortran_strings_ = NULL;

// This function is called on finalization to destroy the Fortran ѕtring
// registry.
static void DestroyFortranStrings() {
  kh_destroy(TDY_STRING_SET, fortran_strings_);
}

// Returns a newly-allocated C string for the given Fortran string pointer with
// the given length. Resources for this string are managed by the running model.
const char* NewCString(char* f_str_ptr, int f_str_len) {
  if (fortran_strings_ == NULL) {
    fortran_strings_ = kh_init(TDY_STRING_SET);
    TDyOnFinalize(DestroyFortranStrings);
  }

  // Copy the Fortran character array to a C string.
  char c_str[f_str_len+1];
  memcpy(c_str, f_str_ptr, sizeof(char) * f_str_len);
  c_str[f_str_len] = '\0';

  // Does this string already exist?
  khiter_t iter = kh_get(TDY_STRING_SET, fortran_strings_, c_str);
  if (iter != kh_end(fortran_strings_)) { // yep
    return kh_key(fortran_strings_, iter);
  } else { // nope
    int retval;
    const char* str = malloc(sizeof(char) * (f_str_len+1));
    strcpy((char*)str, c_str);
    iter = kh_put(TDY_STRING_SET, fortran_strings_, str, &retval);
    return str;
  }
}
