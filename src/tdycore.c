static char help[] = "TDycore \n\
  -tdy_pressure_bc_func <string>                : select one of the registered pressure boundary function \n\
  -tdy_velocity_bc_func <string>                : select one of the registered velocity boundary function \n\
  -tdy_init_file <input_file>                   : file for reading the initial conditions\n\
  -tdy_read_mesh <input_file>                   : mesh file \n\
  -tdy_output_cell_geom_attributes <output_file> : file to output cell geometric attributes\n\
  -tdy_read_cell_geom_attributes <input_file>    : file for reading cell geometric attribtue\n\n";

#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdythimpl.h>
#include <private/tdybdmimpl.h>
#include <private/tdywyimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdytiimpl.h>
#include <tdytimers.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdyioimpl.h>
#include <private/tdydiscretization.h>
#include <petscblaslapack.h>

const char *const TDyDiscretizations[] = {
  "MPFA_O",
  "MPFA_O_DAE",
  "MPFA_O_TRANSIENTVAR",
  "BDM",
  "WY",
  /* */
  "TDyDiscretization","TDY_DISCRETIZATION_",NULL
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
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

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

// Here's a registry of functions that can be used for boundary conditions and
// forcing terms.
KHASH_MAP_INIT_STR(TDY_FUNC_MAP, TDySpatialFunction)
static khash_t(TDY_FUNC_MAP)* funcs_ = NULL;

// This function is called on finalization to destroy the function registry.
static void DestroyFunctionRegistry() {
  kh_destroy(TDY_FUNC_MAP, funcs_);
}

/// Registers a named function that evaluates a scalar quantity on a set of
/// points in space.
/// @param [in] name a name by which the function may be retrieved with
///                  TDyGetFunction
/// @param [in] f the function, which accepts (n, x, v), where n is the number
///               of points x on which f will be evaluated to produce values v.
PetscErrorCode TDyRegisterFunction(const char* name, TDySpatialFunction f) {
  PetscFunctionBegin;
  if (funcs_ == NULL) {
    funcs_ = kh_init(TDY_FUNC_MAP);
    TDyOnFinalize(DestroyFunctionRegistry);
  }

  int retval;
  khiter_t iter = kh_put(TDY_FUNC_MAP, funcs_, name, &retval);
  kh_val(funcs_, iter) = f;
  PetscFunctionReturn(0);
}

/// Retrieves a named function that was registered with TDyRegister.
/// @param [in] name the name by which the desired function was registered
/// @param [out] f the retrieved function
PetscErrorCode TDyGetFunction(const char* name, TDySpatialFunction* f) {
  PetscFunctionBegin;
  int ierr;

  if (funcs_ != NULL) {
    khiter_t iter = kh_get(TDY_FUNC_MAP, funcs_, name);
    if (iter != kh_end(funcs_)) { // found it!
      *f = kh_val(funcs_, iter);
    } else {
      ierr = -1;
      SETERRQ(PETSC_COMM_WORLD, ierr, "Function not found!");
      return ierr;
    }
  } else {
    ierr = -1;
    SETERRQ(PETSC_COMM_WORLD, ierr, "No functions have been registered!");
    return ierr;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetDefaultOptions(TDy tdy) {
  PetscFunctionBegin;

  TDyOptions *options = &tdy->options;
  options->mode = RICHARDS;
  options->discretization = MPFA_O;
  options->gravity_constant = 9.8068;
  options->rho_type = WATER_DENSITY_CONSTANT;
  options->mu_type = WATER_VISCOSITY_CONSTANT;
  options->enthalpy_type = WATER_ENTHALPY_CONSTANT;

  options->porosity=0.25;
  options->permeability=1.e-12;
  options->soil_density=2650.;
  options->soil_specific_heat=1000.0;
  options->thermal_conductivity=1.0;

  options->residual_saturation=0.15;
  options->gardner_n=0.5;
  options->vangenuchten_m=0.8;
  options->vangenuchten_alpha=1.e-4;
  options->mualem_poly_low=0.99;

  options->boundary_pressure = 0.0;
  options->boundary_temperature = 273.0;
  options->boundary_velocity = 0.0;

  options->init_with_random_field = PETSC_FALSE;
  options->init_from_file = PETSC_FALSE;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreate(MPI_Comm comm, TDy *_tdy) {
  TDy            tdy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_tdy,1);
  *_tdy = NULL;

  // Initialize TDycore-specific subsystems.
  ierr = TDyInitSubsystems(); CHKERRQ(ierr);

  ierr = PetscHeaderCreate(tdy,TDY_CLASSID,"TDy","TDy","TDy",comm,
                           TDyDestroy,TDyView); CHKERRQ(ierr);
  *_tdy = tdy;
  tdy->setup_flags |= TDyCreated;

  SetDefaultOptions(tdy);

  ierr = TDyIOCreate(&tdy->io); CHKERRQ(ierr);

  // initialize flags/parameters
  tdy->dm = NULL;
  tdy->solution = NULL;
  tdy->J = NULL;

  tdy->setup_flags |= TDyParametersInitialized;
  PetscFunctionReturn(0);
}

/// Provides a function to be used to create a DM in special cases where a
/// specific geometry is needed. After the function is executed on the DM,
/// DMSetFromOptions is called to apply overrides, and then the DM is
/// distributed appropriately. This function must be called before
/// TDySetFromOptions.
/// @param [inout] tdy the dycore
/// @param [in] context a pointer to contextual information that can be used by
///                       dm_func to create a DM. This pointer is not managed
///                       by the dycore.
/// @param [in] dm_func A function that, given an MPI communicator and a context
///                     pointer, creates a given DM and returns an error
PetscErrorCode TDySetDMConstructor(TDy tdy, void* context,
                                   PetscErrorCode (*dm_func)(MPI_Comm, void*, DM*)) {
  PetscFunctionBegin;
  MPI_Comm comm;
  int ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);
  if ((tdy->setup_flags & TDyDiscretizationSet) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,
      "You must call TDySetDiscretization before TDySetDMConstructor()");
  }
  if (tdy->setup_flags & TDyOptionsSet) {
    SETERRQ(comm,PETSC_ERR_USER,
      "You must call TDySetDMConstructor before TDySetFromOptions()");
  }
  tdy->create_dm_context = context;
  tdy->ops->create_dm = dm_func;
  PetscFunctionReturn(0);
}

// This function is a wrapper used to eliminate the context pointer argument
// from TDySetDMConstructor so the function can be called from Fortran.
static void (*create_dm_f90_)(MPI_Fint*, DM*, PetscErrorCode*) = NULL;
PetscErrorCode TDySetDMConstructorF90(TDy tdy,
                                      void (*dm_func)(MPI_Fint*, DM*, PetscErrorCode*)) {
  PetscFunctionBegin;
  create_dm_f90_ = dm_func;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroy(TDy *_tdy) {
  TDy            tdy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_tdy,1);
  tdy = *_tdy; *_tdy = NULL;

  if (!tdy) PetscFunctionReturn(0);

  if (tdy->regression) {
    free(tdy->regression);
  }

  // Destroy Jacobian data.
  if (tdy->J           ) { ierr = MatDestroy(&tdy->J   ); CHKERRQ(ierr); }
  if (tdy->Jpre        ) { ierr = MatDestroy(&tdy->Jpre); CHKERRQ(ierr); }

  // Call implementation-specific destructor.
  if (tdy->ops->destroy) {
    tdy->ops->destroy(tdy->context);
  }

  ierr = VecDestroy(&tdy->residual); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->soln_prev); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->accumulation_prev); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->solution); CHKERRQ(ierr);
  ierr = TDyIODestroy(&tdy->io); CHKERRQ(ierr);
  ierr = TDyTimeIntegratorDestroy(&tdy->ti); CHKERRQ(ierr);
  ierr = DMDestroy(&tdy->dm); CHKERRQ(ierr);

  if (tdy->conditions) {
    ierr = ConditionsDestroy(tdy->conditions); CHKERRQ(ierr);
  }
  if (tdy->cc) {
    ierr = CharacteristicCurvesDestroy(tdy->cc); CHKERRQ(ierr);
  }
  if (tdy->matprop) {
    ierr = MaterialPropDestroy(tdy->matprop); CHKERRQ(ierr);
  }

  ierr = PetscFree(tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetDimension(TDy tdy,PetscInt *dim) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  ierr = DMGetDimension(tdy->dm,dim); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetDiscretization(TDy tdy, TDyDiscretization* disc) {
  PetscFunctionBegin;
  *disc = tdy->options.discretization;
  PetscFunctionReturn(0);
}

/// Retrieves the DM used by the dycore. This must be called after
/// TDySetFromOptions.
/// @param dm A pointer that stores the DM in use by the dycore
PetscErrorCode TDyGetDM(TDy tdy,DM *dm) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  *dm = tdy->dm;
  PetscFunctionReturn(0);
}

/// Retrieves the indices of the faces belonging to the domain boundary
/// for the dycore. This must be called after TDySetFromOptions. Call
/// TDyRestoreBoundaryFaces when you're finished manipulating boundary
/// faces.
/// @param num_faces A pointer that stores the number of boundary faces
/// @param faces A pointer to an array that stores the indices of boundary faces
PetscErrorCode TDyGetBoundaryFaces(TDy tdy, PetscInt *num_faces,
                                   const PetscInt **faces) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  DMLabel label;
  IS is;
  ierr = DMGetLabel(tdy->dm, "boundary", &label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(label, 1, num_faces); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, 1, &is); CHKERRQ(ierr);
  ierr = ISGetIndices(is, faces); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Resets the pointers set by TDyGetBoundaryFaces.
/// @param num_faces A pointer that stores the number of boundary faces
/// @param faces A pointer to an array that stores the indices of boundary faces
PetscErrorCode TDyRestoreBoundaryFaces(TDy tdy, PetscInt *num_faces,
                                       const PetscInt** faces) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  DMLabel label;
  IS is;
  ierr = DMGetLabel(tdy->dm, "boundary", &label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, 1, &is); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is, faces); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyView(TDy tdy,PetscViewer viewer) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)tdy)->comm,&viewer);
    CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tdy,1,viewer,2);
  PetscFunctionReturn(0);
}

/// Sets options for the dycore based on command line arguments supplied by a
/// user. TDySetFromOptions must be called before TDySetup,
/// since the latter uses options specified by the former.
/// @param [inout] tdy The dycore instance
PetscErrorCode TDySetFromOptions(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) != 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDySetFromOptions before TDySetup()");
  }

  // Collect options from command line arguments.
  TDyOptions *options = &tdy->options;

  //------------------------------------------
  // Set options using command line parameters
  //------------------------------------------

  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)tdy); CHKERRQ(ierr);
  PetscBool flag;

  // Material property options
  ierr = PetscOptionsBegin(comm,NULL,"TDyCore: Material property options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_porosity", "Value of porosity", NULL, options->porosity, &options->porosity, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_permability", "Value of permeability", NULL, options->permeability, &options->permeability, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_soil_density", "Value of soil density", NULL, options->soil_density, &options->soil_density, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_soil_specific_heat", "Value of soil specific heat", NULL, options->soil_specific_heat, &options->soil_specific_heat, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_thermal_conductivity", "Value of thermal conductivity", NULL, options->porosity, &options->porosity, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Characteristic curve options
  ierr = PetscOptionsBegin(comm,NULL,"TDyCore: Characteristic curve options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_residual_satuaration", "Value of residual saturation", NULL, options->residual_saturation, &options->residual_saturation, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_gardner_param_n", "Value of Gardner n parameter", NULL, options->gardner_n, &options->gardner_n, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_vangenuchten_param_m", "Value of VanGenuchten m parameter", NULL, options->vangenuchten_m, &options->vangenuchten_m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_vangenuchten_param_alpha", "Value of VanGenuchten alpha parameter", NULL, options->vangenuchten_alpha, &options->vangenuchten_alpha, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Model options
  TDyMode mode = options->mode;
  ierr = PetscOptionsBegin(comm,NULL,"TDyCore: Model options",""); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_mode","Flow mode",
                          "TDySetMode",TDyModes,(PetscEnum)options->mode,
                          (PetscEnum *)&mode, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_gravity", "Magnitude of gravity vector", NULL,
                          options->gravity_constant, &options->gravity_constant,
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_water_density","Water density vertical profile",
                          "TDySetWaterDensityType",TDyWaterDensityTypes,
                          (PetscEnum)options->rho_type,
                          (PetscEnum *)&options->rho_type, NULL); CHKERRQ(ierr);

  // Create source/sink/boundary conditions.
  ierr = ConditionsCreate(&tdy->conditions); CHKERRQ(ierr);

  char func_name[PETSC_MAX_PATH_LEN];
  ierr = PetscOptionsGetString(NULL, NULL, "-tdy_pressure_bc_func", func_name,
                               sizeof(func_name), &flag); CHKERRQ(ierr);
  if (flag) {
    ierr = TDySelectBoundaryPressureFunction(tdy, func_name);
  } else {
    ierr = PetscOptionsReal("-tdy_pressure_bc_value", "Constant boundary pressure",
                            NULL, options->boundary_pressure,
                            &options->boundary_pressure,&flag); CHKERRQ(ierr);
    if (flag) {
      ierr = ConditionsSetConstantBoundaryPressure(tdy->conditions,
                                                   options->boundary_pressure); CHKERRQ(ierr);
    } else { // TODO: what goes here??
    }
  }

  ierr = PetscOptionsGetString(NULL, NULL, "-tdy_velocity_bc_func", func_name,
                               sizeof(func_name), &flag); CHKERRQ(ierr);
  if (flag) {
    ierr = TDySelectBoundaryVelocityFunction(tdy, func_name);
  } else {
    ierr = PetscOptionsReal("-tdy_velocity_bc_value", "Constant normal boundary velocity",
                            NULL, options->boundary_velocity,
                            &options->boundary_velocity,&flag); CHKERRQ(ierr);
    if (flag) {
      ierr = ConditionsSetConstantBoundaryVelocity(tdy->conditions,
                                                   options->boundary_velocity);
    } else { // TODO: what goes here??
    }
  }

  ierr = PetscOptionsGetString(NULL, NULL, "-tdy_temperature_bc_func", func_name,
                               sizeof(func_name), &flag); CHKERRQ(ierr);
  if (flag) {
    ierr = TDySelectBoundaryTemperatureFunction(tdy, func_name); CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsReal("-tdy_temperature_bc_value", "Constant boundary temperature",
                            NULL, options->boundary_temperature,
                            &options->boundary_temperature,&flag); CHKERRQ(ierr);
    if (flag) {
      ierr = ConditionsSetConstantBoundaryTemperature(tdy->conditions,
                                                      options->boundary_temperature);
    } else { // TODO: what goes here??
    }
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Numerics options
  TDyDiscretization discretization = options->discretization;
  ierr = PetscOptionsBegin(comm,NULL,"TDyCore: Numerics options",""); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_discretization","Discretization",
                          "TDySetDiscretization",TDyDiscretizations,
                          (PetscEnum)options->discretization,(PetscEnum *)&discretization,
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = PetscOptionsBool("-tdy_init_with_random_field","Initialize solution with a random field","",options->init_with_random_field,&(options->init_with_random_field),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_init_file", options->init_file,sizeof(options->init_file),&options->init_from_file); CHKERRQ(ierr);

  if (options->init_from_file && options->init_with_random_field) {
    SETERRQ(comm,PETSC_ERR_USER,
            "Only one of -tdy_init_from_file and -tdy_init_with_random_field can be specified");
  }

  // Mesh-related options
  ierr = PetscOptionsBegin(comm,NULL,"TDyCore: Mesh options",""); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_read_mesh", options->mesh_file,sizeof(options->mesh_file),&options->read_mesh); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tdy_output_mesh","Enable output of mesh attributes","",options->output_mesh,&(options->output_mesh),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Other options
  ierr = PetscOptionsBool("-tdy_regression_test","Enable output of a regression file","",options->regression_testing,&(options->regression_testing),NULL); CHKERRQ(ierr);

  // Override the mode and/or discretization if needed.
  if (options->mode != mode) {
    TDySetMode(tdy, mode);
  }
  if (options->discretization != discretization) {
    TDySetDiscretization(tdy, discretization);
  }

  // Now that we know the discretization, we can create our implementation-
  // specific context.
  ierr = tdy->ops->create(&tdy->context); CHKERRQ(ierr);

  // Mode/discretization-specific options.
  if (tdy->ops->set_from_options) {
    ierr = tdy->ops->set_from_options(tdy->context, &tdy->options); CHKERRQ(ierr);
  }

  // Wrap up and indicate that options are set.
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  tdy->setup_flags |= TDyOptionsSet;

  // Create our DM.
  if (!tdy->dm) {
    DM dm;
    if (tdy->options.read_mesh) {
      ierr = DMPlexCreateFromFile(comm, tdy->options.mesh_file,
                                  PETSC_TRUE, &dm); CHKERRQ(ierr);
    } else {
      if (tdy->ops->create_dm) {
        // We've been instructed to create a DM ourselves.
        ierr = tdy->ops->create_dm(comm, tdy->create_dm_context, &dm);
      } else if (create_dm_f90_) {
        // Create a DM all-Fortran-like.
        MPI_Fint comm_f = MPI_Comm_c2f(comm);
        create_dm_f90_(&comm_f, &dm, &ierr);
      } else {
        ierr = DMPlexCreate(comm, &dm); CHKERRQ(ierr);
      }
      // Here we lean on PETSc's DM* options for overrides.
      ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    }

    // Set up fields on the DM.
    ierr = tdy->ops->set_dm_fields(tdy->context, dm); CHKERRQ(ierr);

    // Set up a natural -> global ordering on the DM.
    ierr = DMSetUseNatural(dm, PETSC_TRUE); CHKERRQ(ierr);

    // Distribute the mesh, however we got it.
    DM dm_dist;
    ierr = DMPlexDistribute(dm, 1, NULL, &dm_dist); CHKERRQ(ierr);
    if (dm_dist) {
      DMDestroy(&dm); CHKERRQ(ierr);
      dm = dm_dist;
    }
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
    tdy->dm = dm;
  }

  // Mark the grid's boundary faces and their transitive closure. All are
  // stored at their appropriate strata within the label.
  DMLabel boundary_label;
  ierr = DMCreateLabel(tdy->dm, "boundary"); CHKERRQ(ierr);
  ierr = DMGetLabel(tdy->dm, "boundary", &boundary_label); CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(tdy->dm, 1, boundary_label); CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(tdy->dm, boundary_label); CHKERRQ(ierr);

  PetscInt dim;
  ierr = DMGetDimension(tdy->dm, &dim); CHKERRQ(ierr);

  // Create an empty material properties object. Each function must be set
  // explicitly by the driver program.
  ierr = MaterialPropCreate(dim, &tdy->matprop); CHKERRQ(ierr);

  // Create characteristic curves.
  ierr = CharacteristicCurvesCreate(&tdy->cc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Performs setup, including allocation and configuration of any bookkeeping
/// data structures and the configuration of the dycore's DM object, which can
/// subsequently be passed to a solver (e.g. SNES or TS). This function must be
/// called after TDySetOptions.
/// @param [inout] tdy the dycore instance
PetscErrorCode TDySetup(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDyDiscretizationSet) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,
      "You must call TDySetDiscretization before TDySetup()");
  }

  if ((tdy->setup_flags & TDyOptionsSet) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDySetFromOptions before TDySetup()");
  }
  TDyEnterProfilingStage("TDycore Setup");

  // Set the EOS from options.
  tdy->eos.density_type = tdy->options.rho_type;
  tdy->eos.viscosity_type = tdy->options.mu_type;
  tdy->eos.enthalpy_type = tdy->options.enthalpy_type;

  // Perform implementation-specific setup.
  ierr = tdy->ops->setup(tdy->context, tdy->dm, &tdy->eos, tdy->matprop,
                         tdy->cc, tdy->conditions); CHKERRQ(ierr);

  // Record metadata for scaling studies.
  TDySetTimingMetadata(tdy);

  if (tdy->options.regression_testing) {
    /* must come after Sections are set up in
       TDySetupDiscretization->XXXInitialize */
    ierr = TDyRegressionInitialize(tdy); CHKERRQ(ierr);
  }
  if (tdy->options.output_mesh) {
    if (tdy->options.discretization != MPFA_O) {
      SETERRQ(comm,PETSC_ERR_USER,
              "-tdy_output_mesh only supported for MPFA-O discretization");
    }
  }
  TDyExitProfilingStage("TDycore Setup");
  tdy->setup_flags |= TDySetupFinished;
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/// Sets the dycore's "mode", specifying the governing equations it solves.
/// @param [inout] tdy the dycore
/// @param [in] mode the selected mode
PetscErrorCode TDySetMode(TDy tdy, TDyMode mode) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  tdy->options.mode = mode;
  tdy->setup_flags |= TDyModeSet;

  // If we are resetting the mode and have already set the discretization,
  // we call TDySetDiscretization again.
  if (tdy->setup_flags & TDyDiscretizationSet) {
    PetscInt ierr = TDySetDiscretization(tdy, tdy->options.discretization); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/// Sets the discretization used by the dycore. This must be called after
/// TDySetMode.
/// @param [inout] tdy the dycore
/// @param [in] discretization the selected discretization
PetscErrorCode TDySetDiscretization(TDy tdy, TDyDiscretization discretization) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscErrorCode ierr;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDyModeSet) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,
      "You must call TDySetMode before TDySetDiscretization");
  }

  // Set function pointers for operations.
  if (tdy->options.mode == RICHARDS) {
    if (discretization == MPFA_O) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_MPFAO;
      tdy->ops->setup = TDySetup_Richards_MPFAO;
      tdy->ops->update_state = TDyUpdateState_Richards_MPFAO;
      tdy->ops->compute_error_norms = TDyComputeErrorNorms_MPFAO;
      tdy->ops->get_saturation = TDyGetSaturation_MPFAO;
    } else if (discretization == MPFA_O_DAE) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_MPFAO_DAE;
      tdy->ops->setup = TDySetup_Richards_MPFAO_DAE;
      tdy->ops->update_state = TDyUpdateState_Richards_MPFAO;
      tdy->ops->compute_error_norms = TDyComputeErrorNorms_MPFAO;
      tdy->ops->get_saturation = TDyGetSaturation_MPFAO;
    } else if (discretization == MPFA_O_TRANSIENTVAR) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_MPFAO;
      tdy->ops->setup = TDySetup_Richards_MPFAO;
      tdy->ops->update_state = TDyUpdateState_Richards_MPFAO;
      tdy->ops->compute_error_norms = TDyComputeErrorNorms_MPFAO;
      tdy->ops->get_saturation = TDyGetSaturation_MPFAO;
    } else if (discretization == BDM) {
      tdy->ops->create = TDyCreate_BDM;
      tdy->ops->destroy = TDyDestroy_BDM;
      tdy->ops->set_from_options = TDySetFromOptions_BDM;
      tdy->ops->setup = TDySetup_BDM;
      tdy->ops->set_dm_fields = TDySetDMFields_BDM;
      tdy->ops->update_state = NULL; // FIXME: ???
      tdy->ops->compute_error_norms = TDyComputeErrorNorms_BDM;
    } else if (discretization == WY) {
      tdy->ops->create = TDyCreate_WY;
      tdy->ops->destroy = TDyDestroy_WY;
      tdy->ops->set_from_options = TDySetFromOptions_WY;
      tdy->ops->set_dm_fields = TDySetDMFields_WY;
      tdy->ops->setup = TDySetup_WY;
      tdy->ops->update_state = TDyUpdateState_WY;
      tdy->ops->compute_error_norms = TDyComputeErrorNorms_WY;
    } else {
      SETERRQ(comm,PETSC_ERR_USER, "Invalid discretization given!");
    }
  } else if (tdy->options.mode == TH) {
    if (discretization == MPFA_O) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_TH_MPFAO;
      tdy->ops->setup = TDySetup_TH_MPFAO;
      tdy->ops->update_state = TDyUpdateState_TH_MPFAO;
      tdy->ops->get_saturation = TDyGetSaturation_MPFAO;
    } else {
      SETERRQ(comm,PETSC_ERR_USER,
        "The TH mode does not support the selected discretization!");
    }
  }
  tdy->options.discretization = discretization;
  tdy->setup_flags |= TDyDiscretizationSet;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetWaterDensityType(TDy tdy, TDyWaterDensityType dentype) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  tdy->options.rho_type = dentype;
  PetscFunctionReturn(0);
}

/// Sets the porosity used by the dycore to the given constant. May be called
/// anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantPorosity(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantPorosity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute porosities in the dycore. May be called
/// anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetPorosityFunction(TDy tdy,
                                      TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousPorosity(tdy->matprop, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability used by the dycore to the given constant. May be called
/// anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantIsotropicPermeability(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantIsotropicPermeability(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute (isotropic) permeabilities in the dycore.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetIsotropicPermeabilityFunction(TDy tdy,
                                                   TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousIsotropicPermeability(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability used by the dycore to the given diagonal constant. May be
/// called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantDiagonalPermeability(TDy tdy, PetscReal value[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantDiagonalPermeability(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute (diagonal anisotropic) permeabilities in
/// the dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetDiagonalPermeabilityFunction(TDy tdy,
                                                  TDyVectorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousDiagonalPermeability(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability used by the dycore to the given tensor constant. May be
/// called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantTensorPermeability(TDy tdy, PetscReal value[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantTensorPermeability(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute anisotropic permeabilities in the dycore.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetTensorPermeabilityFunction(TDy tdy,
                                                TDyTensorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousTensorPermeability(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity used by the dycore to the given constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantIsotropicThermalConductivity(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantIsotropicThermalConductivity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute (isotropic) thermal conductivities in the
/// dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetIsotropicThermalConductivityFunction(TDy tdy,
                                                          TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousIsotropicThermalConductivity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity used by the dycore to the given diagonal constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantDiagonalThermalConductivity(TDy tdy, PetscReal value[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantDiagonalThermalConductivity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute (diagonal anisotropic) thermal
/// conductivities in the dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetDiagonalThermalConductivityFunction(TDy tdy,
                                                         TDyVectorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousDiagonalThermalConductivity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity used by the dycore to the given tensor constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantTensorThermalConductivity(TDy tdy, PetscReal value[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantTensorThermalConductivity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute anisotropic thermal conductivities in the
/// dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetTensorThermalConductivityFunction(TDy tdy,
                                                       TDyTensorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousTensorThermalConductivity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the residual saturation used by the dycore to the given constant. May be called
/// anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantResidualSaturation(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantResidualSaturation(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute residual saturations in the dycore. May be
/// called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetResidualSaturationFunction(TDy tdy,
                                                TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousResidualSaturation(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the soil density used by the dycore to the given constant. May be called
/// anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantSoilDensity(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantSoilDensity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute soil densities in the dycore. May be
/// called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetSoilDensityFunction(TDy tdy,
                                         TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousSoilDensity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the soil specific heat used by the dycore to the given constant. May be called
/// anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantSoilSpecificHeat(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantSoilSpecificHeat(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute soil specific heats in the dycore. May be
/// called anytime after TDySetFromOptions.
PetscErrorCode TDySetSoilSpecificHeatFunction(TDy tdy,
                                              TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousSoilSpecificHeat(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// This struct is stored in a context and used to call a spatial function with a
// NULL context.
typedef struct WrapperStruct {
  TDySpatialFunction func;
} WrapperStruct;

// This function calls an underlying Function with a NULL context.
static PetscErrorCode WrapperFunction(void *context, PetscInt n, PetscReal *x, PetscReal *v) {
  WrapperStruct *wrapper = context;
  wrapper->func(n, x, v);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetForcingFunction(TDy tdy,
                                     TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
  wrapper->func = f;
  ierr = ConditionsSetForcing(tdy->conditions, wrapper,
                              WrapperFunction, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetEnergyForcingFunction(TDy tdy,
                                           TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
  wrapper->func = f;
  ierr = ConditionsSetEnergyForcing(tdy->conditions, wrapper,
                                    WrapperFunction, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBoundaryPressureFunction(TDy tdy,
                                              TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
  wrapper->func = f;
  ierr = ConditionsSetBoundaryPressure(tdy->conditions, wrapper,
                                       WrapperFunction, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBoundaryTemperatureFunction(TDy tdy,
                                                 TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
  wrapper->func = f;
  ierr = ConditionsSetBoundaryTemperature(tdy->conditions, wrapper,
                                          WrapperFunction, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetBoundaryVelocityFunction(TDy tdy,
                                              TDyVectorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
  wrapper->func = f;
  ierr = ConditionsSetBoundaryVelocity(tdy->conditions, wrapper,
                                       WrapperFunction, free); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectForcingFunction(TDy tdy,
                                        const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetForcingFunction(tdy, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectEnergyForcingFunction(TDy tdy,
                                              const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetEnergyForcingFunction(tdy, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryPressureFunction(TDy tdy,
                                                 const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetBoundaryPressureFunction(tdy, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryTemperatureFunction(TDy tdy,
                                                    const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetBoundaryTemperatureFunction(tdy, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySelectBoundaryVelocityFunction(TDy tdy,
                                                 const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetBoundaryVelocityFunction(tdy, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Updates the secondary state of the system using the solution data.
/// @param [inout] tdy a dycore object
/// @param [in] U an array of solution data
PetscErrorCode TDyUpdateState(TDy tdy,PetscReal *U) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyEnterProfilingStage("TDycore Setup");
  TDY_START_FUNCTION_TIMER()

  // Call the implementation-specific state update.
  ierr = tdy->ops->update_state(tdy->context, tdy->dm, &tdy->eos, tdy->matprop,
                                tdy->cc, U); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  TDyExitProfilingStage("TDycore Setup");
  PetscFunctionReturn(0);
}

PetscInt TDyGetNumberOfCellVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  PetscInt vStart, vEnd, cStart, cEnd;
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  PetscInt nq = -1;
  for(PetscInt c=cStart; c<cEnd; c++) {
    PetscInt *closure = NULL;
    PetscInt closureSize;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    PetscInt q = 0;
    for (PetscInt i=0; i<closureSize*2; i+=2) {
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

/// Computes error norms for the pressure and/or the velocity, given the
/// solution vector.
/// @param [in] tdy the dycore instance
/// @param [in] U the solution vector
/// @param [out] pressure_norm the norm for the pressure (can be NULL)
/// @param [out] velocity_norm the norm for the velocity (can be NULL)
PetscErrorCode TDyComputeErrorNorms(TDy tdy, Vec U,
                                    PetscReal *pressure_norm,
                                    PetscReal *velocity_norm) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = tdy->ops->compute_error_norms(tdy->context, tdy->dm, tdy->conditions,
                                       U, pressure_norm, velocity_norm);
  CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyOutputRegression(TDy tdy, Vec U) {

  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tdy->options.regression_testing){
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
  ierr = TDyNaturalToGlobal(tdy,initial,tdy->solution); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//------------------------------------------
// Solver-related functions (to be deleted)
//------------------------------------------

PetscErrorCode TDySetIFunction(TS ts,TDy tdy) {
  MPI_Comm       comm;
  DM             dm;
  PetscSection   sec;
  PetscErrorCode ierr;
  PetscValidPointer( ts,1);
  PetscValidPointer(tdy,2);

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = PetscObjectGetComm((PetscObject)ts,&comm); CHKERRQ(ierr);
  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  ierr = DMGetSection(tdy->dm, &sec);
  PetscInt num_fields;
  ierr = PetscSectionGetNumFields(sec, &num_fields);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  switch (tdy->options.discretization) {
  case MPFA_O:
    switch (tdy->options.mode) {
    case RICHARDS:
      ierr = TSSetIFunction(ts,NULL,TDyMPFAOIFunction,tdy); CHKERRQ(ierr);
      break;
    case TH:
      ierr = TSSetIFunction(ts,NULL,TDyMPFAOIFunction_TH,tdy); CHKERRQ(ierr);
      break;
    }
    break;
  case MPFA_O_DAE:
    ierr = TSSetIFunction(ts,NULL,TDyMPFAOIFunction_DAE,tdy); CHKERRQ(ierr);
    break;
  case MPFA_O_TRANSIENTVAR:
    ierr = DMTSSetIFunction(dm,TDyMPFAOIFunction_TransientVariable,tdy); CHKERRQ(ierr);
    ierr = DMTSSetTransientVariable(dm,TDyMPFAOTransientVariable,tdy); CHKERRQ(ierr);
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
  switch (tdy->options.discretization) {
  case MPFA_O:
    ierr = TDyCreateJacobian(tdy); CHKERRQ(ierr);
    switch (tdy->options.mode) {
    case RICHARDS:
      ierr = TSSetIJacobian(ts,tdy->J,tdy->J,TDyMPFAOIJacobian,tdy); CHKERRQ(ierr);
      break;
    case TH:
      ierr = TSSetIJacobian(ts,tdy->J,tdy->J,TDyMPFAOIJacobian_TH,tdy); CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscValidPointer(snes,1);
  PetscValidPointer(tdy,2);

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm); CHKERRQ(ierr);
  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  switch (tdy->options.discretization) {
  case MPFA_O:
    ierr = SNESSetFunction(snes,tdy->residual,TDyMPFAOSNESFunction,tdy); CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscValidPointer(snes,1);
  PetscValidPointer(tdy,2);

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm); CHKERRQ(ierr);
  PetscInt dim;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);

  switch (tdy->options.discretization) {
  case MPFA_O:
    ierr = SNESSetJacobian(snes,tdy->J,tdy->J,TDyMPFAOSNESJacobian,tdy); CHKERRQ(ierr);
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

PetscErrorCode TDyPostSolveSNESSolver(TDy tdy,Vec U) {

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(U,tdy->soln_prev); CHKERRQ(ierr);
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

  switch (tdy->options.discretization) {
  case MPFA_O:
    ierr = TDyMPFAOSNESPreSolve(tdy); CHKERRQ(ierr);
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

/// Allocates storage for the vectors used by the dycore. Storage is allocated
/// the first time the function is called. Subsequent calls have no effect.
PetscErrorCode TDyCreateVectors(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  if (tdy->solution == NULL) {
    ierr = DMCreateGlobalVector(tdy->dm,&tdy->solution); CHKERRQ(ierr);
    ierr = VecDuplicate(tdy->solution,&tdy->residual); CHKERRQ(ierr);
    ierr = VecDuplicate(tdy->solution,&tdy->accumulation_prev); CHKERRQ(ierr);
    ierr = VecDuplicate(tdy->solution,&tdy->soln_prev); CHKERRQ(ierr);
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/// Allocates storage for the dycore's Jacobian and preconditioner matrices.
/// Storage is allocated the first time the function is called. Subsequent calls
/// have no effect.
PetscErrorCode TDyCreateJacobian(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  if (tdy->J == NULL) {
    ierr = TDyCreateJacobianMatrix(tdy,&tdy->J); CHKERRQ(ierr);
    ierr = TDyCreateJacobianMatrix(tdy,&tdy->Jpre); CHKERRQ(ierr);
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

// TODO: Temporary method to retrieve saturation values. We need a better I/O
// TODO: strategy.
PetscErrorCode TDyGetSaturation(TDy tdy, PetscReal* saturation) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  if (tdy->ops->get_saturation) {
    ierr = tdy->ops->get_saturation(tdy->context, saturation); CHKERRQ(ierr);
  } else {
    ierr = -1;
    SETERRQ(PETSC_COMM_WORLD, ierr,
      "This implementation does not allow the retrieval of saturation values!");
  }
  PetscFunctionReturn(0);
}
