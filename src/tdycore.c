static char help[] = "TDycore \n\
  -tdy_init_file <input_file>                   : file for reading the initial conditions\n\
  -tdy_read_mesh <input_file>                   : mesh file \n\
  -tdy_read_pflotran_mesh <input_file>          : PFLOTRAN HDF5 mesh file \n\
  -tdy_output_cell_geom_attributes <output_file> : file to output cell geometric attributes\n\
  -tdy_read_cell_geom_attributes <input_file>    : file for reading cell geometric attribtue\n\n";

#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdythimpl.h>
#include <private/tdybdmimpl.h>
#include <private/tdywyimpl.h>
#include <private/tdyfvtpfimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdytiimpl.h>
#include <tdytimers.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdyioimpl.h>
#include <private/tdydiscretizationimpl.h>
#include <petscblaslapack.h>

const char *const TDyDiscretizations[] = {
  "MPFA_O",
  "MPFA_O_DAE",
  "MPFA_O_TRANSIENTVAR",
  "BDM",
  "WY",
  "FV_TPF",
  /* */
  "TDyDiscretization","TDY_DISCRETIZATION_",NULL
};

const char *const TDyMPFAOGmatrixMethods[] = {
  "MPFAO_GMATRIX_DEFAULT",
  "MPFAO_GMATRIX_TPF",
  /* */
  "TDyMPFAOGmatrixMethod","TDY_MPFAO_GMATRIX_METHOD_",NULL
};

const char *const TDyModes[] = {
  "RICHARDS",
  "TH",
  "SALINITY",
  /* */
  "TDyMode","TDY_MODE_",NULL
};

const char *const TDyWaterDensityTypes[] = {
  "CONSTANT",
  "EXPONENTIAL",
  "BATZLE_AND_WANG",
  /* */
  "TDyWaterDensityType","TDY_DENSITY_",NULL
};

const char *const TDyWaterViscosityTypes[] = {
  "CONSTANT",
  "BATZLE_AND_WANG",
  /* */
  "TDyWaterViscosityType","TDY_VISCOSITY_",NULL
};

const char *const TDyWaterEnthalpyTypes[] = {
  "CONSTANT",
  "TDyWaterEnthalpyType","TDY_ENTHALPY_",NULL
};

// This struct is stored in a context and used to call a spatial function with a
// NULL context.
typedef struct WrapperStruct {
  TDySpatialFunction func;
} WrapperStruct;

// This function calls an underlying function with a NULL context.
static PetscErrorCode WrapperFunction(void *context, PetscReal t, PetscInt n, PetscReal *x, PetscReal *v) {
  WrapperStruct *wrapper = context;
  wrapper->func(t, n, x, v);
  PetscFunctionReturn(0);
}

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"+++++++++++++++++ TDycore +++++++++++++++++\n"); CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"+++++++++++++++++ TDycore +++++++++++++++++\n"); CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  if (shutdown_funcs_ == NULL) {
    shutdown_funcs_cap_ = 32;
    ierr = TDyAlloc(sizeof(ShutdownFunc) * shutdown_funcs_cap_, &shutdown_funcs_); CHKERRQ(ierr);
  } else if (num_shutdown_funcs_ == shutdown_funcs_cap_) { // need more space!
    shutdown_funcs_cap_ *= 2;
    ierr = TDyRealloc(sizeof(ShutdownFunc) * shutdown_funcs_cap_, &shutdown_funcs_); CHKERRQ(ierr);
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
    TDyFree(shutdown_funcs_);
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
  options->saline_molecular_weight = 58.44;
  options->saline_diffusivity = 1.e-6;

  options->residual_saturation=0.15;
  options->gardner_n=0.5;
  options->vangenuchten_m=0.8;
  options->vangenuchten_alpha=1.e-4;
  options->mualem_poly_x0=0.995;
  options->mualem_poly_x1=0.99;
  options->mualem_poly_x2=1.00;
  options->mualem_poly_dx=0.005;

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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating TDycore\n"); CHKERRQ(ierr);

  // Initialize TDycore-specific subsystems.
  ierr = TDyInitSubsystems(); CHKERRQ(ierr);

  ierr = PetscHeaderCreate(tdy,TDY_CLASSID,"TDy","TDy","TDy",comm,
                           TDyDestroy,TDyView); CHKERRQ(ierr);
  *_tdy = tdy;
  tdy->setup_flags |= TDyCreated;

  SetDefaultOptions(tdy);

  ierr = TDyIOCreate(&tdy->io); CHKERRQ(ierr);

  // Create source/sink/boundary conditions.
  ierr = ConditionsCreate(&tdy->conditions); CHKERRQ(ierr);

  // initialize flags/parameters
  ierr = TDyDiscretizationCreate(&tdy->discretization); CHKERRQ(ierr);

  tdy->soln = NULL;
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
    ierr = TDyRegressionDestroy(tdy);
  }

  // Destroy Jacobian data.
  if (tdy->J           ) { ierr = MatDestroy(&tdy->J   ); CHKERRQ(ierr); }
  if (tdy->Jpre        ) { ierr = MatDestroy(&tdy->Jpre); CHKERRQ(ierr); }

  // Call implementation-specific destructor.
  if (tdy->ops->destroy) {
    tdy->ops->destroy(tdy->context);
  }

  // Clean up diagnostics.
  if (tdy->ops->update_diagnostics && (tdy->setup_flags & TDySetupFinished)) {
    VecDestroy(&tdy->diag_vec);
    DMDestroy(&tdy->diag_dm);
  }

  ierr = VecDestroy(&tdy->residual); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->soln_prev); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->soln); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->soln_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&tdy->accumulation_prev); CHKERRQ(ierr);
  ierr = TDyIODestroy(&tdy->io); CHKERRQ(ierr);
  ierr = TDyTimeIntegratorDestroy(&tdy->ti); CHKERRQ(ierr);

  if (tdy->conditions) {
    ierr = ConditionsDestroy(tdy->conditions); CHKERRQ(ierr);
  }
  if (tdy->cc) {
    ierr = CharacteristicCurvesDestroy(tdy->cc); CHKERRQ(ierr);
  }
  if (tdy->matprop) {
    ierr = MaterialPropDestroy(tdy->matprop); CHKERRQ(ierr);
  }

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = TDyFree(tdy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetDimension(TDy tdy,PetscInt *dim) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,dim); CHKERRQ(ierr);
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
  *dm = ((tdy->discretization)->tdydm)->dm;
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
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "boundary", &label); CHKERRQ(ierr);
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
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "boundary", &label); CHKERRQ(ierr);
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

static PetscErrorCode GetBCsForBoundaries(TDy tdy,
                                          PetscInt num_boundaries,
                                          char *boundary_names[num_boundaries],
                                          BoundaryConditions *bcs) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  char *spec_boundaries[num_boundaries];
  PetscInt num_spec_boundaries = num_boundaries;
  PetscBool bc_defined;

  // Zero out boundary conditions so they're undefined.
  memset(bcs, 0, sizeof(BoundaryConditions) * num_boundaries);

  // Look for pressure boundary conditions.
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-tdy_bc_pressure",
                                    spec_boundaries, &num_spec_boundaries,
                                    &bc_defined); CHKERRQ(ierr);
  if (bc_defined) {
    for (PetscInt i = 0; i < num_spec_boundaries; ++i) {
      const char *boundary = (const char*)spec_boundaries[i];
      BoundaryConditions *bc_i = &bcs[i];

      // Are we given a pressure value?
      char option_name[256];
      snprintf(option_name, 255, "-tdy_bc_pressure_%s_value", boundary);
      PetscReal p0;
      PetscBool value_defined;
      ierr = PetscOptionsGetReal(NULL, NULL, option_name, &p0, &value_defined);
      if (value_defined) {
        CreateConstantPressureBC(&bc_i->flow_bc, p0);
      } else {
        // No value defined. Is there a function assigned?
        snprintf(option_name, 128, "-tdy_bc_pressure_%s_function", boundary);
        char func_name[64];
        ierr = PetscOptionsGetString(NULL, NULL, option_name, func_name, 64,
                                     &value_defined); CHKERRQ(ierr);
        if (value_defined) {
          // Fetch the registered function.
          WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
          ierr = TDyGetFunction((const char*)func_name, &wrapper->func);
          CHKERRQ(ierr);
          bc_i->flow_bc = (FlowBC){
            .type = PRESSURE_BC,
            .context = wrapper,
            .compute = WrapperFunction,
            .dtor = TDyFree
          };
        } else {
          SETERRQ(comm, PETSC_ERR_USER,
                  "A pressure boundary condition was assigned to a boundary, "
                  "but the boundary condition was not defined on that "
                  "boundary. Please specify a pressure value or the name of a "
                  "pressure function for each relevant boundary/face set.");
        }
      }

      PetscFree(spec_boundaries[i]);
    }
  }

  // Look for no-flow boundary conditions.
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-tdy_bc_noflow",
                                    spec_boundaries, &num_spec_boundaries,
                                    &bc_defined); CHKERRQ(ierr); CHKERRQ(ierr);
  if (bc_defined) {
    for (PetscInt i = 0; i < num_spec_boundaries; ++i) {
      BoundaryConditions *bc_i = &bcs[i];

      if (bc_i->flow_bc.type != UNDEFINED_FLOW_BC) {
        SETERRQ(comm, PETSC_ERR_USER,
                "A noflow boundary condition was assigned to a boundary "
                "previously assigned to a pressure boundary condition. Please "
                "check that each boundary is assigned only one flow BC.");
      }
      CreateConstantVelocityBC(&bc_i->flow_bc, 0.0);

      PetscFree(spec_boundaries[i]);
    }
  }

  // Look for seepage boundary conditions.
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-tdy_bc_seepage",
                                    spec_boundaries, &num_spec_boundaries,
                                    &bc_defined); CHKERRQ(ierr);
  if (bc_defined) {
    for (PetscInt i = 0; i < num_spec_boundaries; ++i) {
      BoundaryConditions *bc_i = &bcs[i];

      if (bc_i->flow_bc.type != UNDEFINED_FLOW_BC) {
        SETERRQ(comm, PETSC_ERR_USER,
                "A seepage boundary condition was assigned to a boundary "
                "previously assigned to a pressure boundary condition. Please "
                "check that each boundary is assigned only one flow BC.");
      }
      CreateSeepageBC(&bc_i->flow_bc);

      PetscFree(spec_boundaries[i]);
    }
  }

  // Look for temperature boundary conditions.
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-tdy_bc_temperature",
                                    spec_boundaries, &num_spec_boundaries,
                                    &bc_defined); CHKERRQ(ierr);
  if (bc_defined) {
    for (PetscInt i = 0; i < num_spec_boundaries; ++i) {
      const char *boundary = (const char*)spec_boundaries[i];
      BoundaryConditions *bc_i = &bcs[i];

      // Are we given a temperature value?
      char option_name[256];
      snprintf(option_name, 255, "-tdy_bc_temperature_%s_value", boundary);
      PetscReal T0;
      PetscBool value_defined;
      ierr = PetscOptionsGetReal(NULL, NULL, option_name, &T0, &value_defined);
      if (value_defined) {
        CreateConstantTemperatureBC(&bc_i->thermal_bc, T0);
      } else {
        // No value defined. Is there a function assigned?
        snprintf(option_name, 128, "-tdy_bc_temperature_%s_function", boundary);
        char func_name[64];
        ierr = PetscOptionsGetString(NULL, NULL, option_name, func_name, 64,
                                     &value_defined); CHKERRQ(ierr);
        if (value_defined) {
          // Fetch the registered function.
          WrapperStruct *wrapper = malloc(sizeof(WrapperStruct));
          ierr = TDyGetFunction((const char*)func_name, &wrapper->func);
          CHKERRQ(ierr);
          bc_i->thermal_bc = (ThermalBC){
            .type = TEMPERATURE_BC,
            .context = wrapper,
            .compute = WrapperFunction,
            .dtor = TDyFree
          };
        } else {
          SETERRQ(comm, PETSC_ERR_USER,
                  "A temperature boundary condition was assigned to a boundary, "
                  "but the boundary condition was not defined on that "
                  "boundary. Please specify a pressure value or the name of a "
                  "pressure function for each relevant boundary/face set.");
        }
      }

      PetscFree(spec_boundaries[i]);
    }
  }

  // Look for thermally insulated boundary conditions.
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-tdy_bc_insulated",
                                    spec_boundaries, &num_spec_boundaries,
                                    &bc_defined); CHKERRQ(ierr);
  if (bc_defined) {
    for (PetscInt i = 0; i < num_spec_boundaries; ++i) {
      BoundaryConditions *bc_i = &bcs[i];

      if (bc_i->thermal_bc.type != UNDEFINED_THERMAL_BC) {
        SETERRQ(comm, PETSC_ERR_USER,
                "An insulated boundary condition was assigned to a boundary "
                "previously assigned to a temperature boundary condition. "
                "Please check that each boundary is assigned only one thermal "
                "BC.");
      }
      CreateConstantHeatFluxBC(&bc_i->thermal_bc, 0.0);

      PetscFree(spec_boundaries[i]);
    }
  }

  PetscFunctionReturn(0);
}

#define MAX_BOUNDARIES 32
#define MAX_FACE_SETS 32
#define MAX_FACE_SETS_PER_BOUNDARY 32
static PetscErrorCode ProcessBCOptions(TDy tdy) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  // Are we provided with names (strings) for numbered face sets?
  char *boundary_names[MAX_BOUNDARIES];
  PetscInt num_boundaries = MAX_BOUNDARIES;
  PetscBool have_boundary_names;
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-tdy_bc_sets", boundary_names,
                                    &num_boundaries, &have_boundary_names);
  if (have_boundary_names && (num_boundaries > 0)) {
    // Named boundaries are allowed only for the TPF approximation.
    if (tdy->options.discretization != FV_TPF) {
      SETERRQ(comm, PETSC_ERR_USER,
        "Named boundaries are supported only for the two-point-flux (TPF) approximation.");
    }

    // Look for boundary conditions assigned to these named boundaries.
    BoundaryConditions bcs[num_boundaries];
    ierr = GetBCsForBoundaries(tdy, num_boundaries, boundary_names, bcs);
    CHKERRQ(ierr);

    // Look for options (in the form "-tdy_bc_BOUNDARY_NAME b1, b2, ..., bn")
    // that define a named boundary (BOUNDARY_NAME) in terms of face set
    // indices (b1, b2, ..., bn).
    for (PetscInt i = 0; i < num_boundaries; ++i) {
      size_t name_len = strlen(boundary_names[i]);
      char option_name[8 + name_len + 1];
      snprintf(option_name, 8 + name_len, "-tdy_bc_%s", boundary_names[i]);
      PetscInt num_face_sets = MAX_FACE_SETS_PER_BOUNDARY;
      PetscInt face_sets[MAX_FACE_SETS_PER_BOUNDARY];
      PetscBool boundary_defined;
      ierr = PetscOptionsGetIntArray(NULL, NULL, option_name, face_sets,
                                     &num_face_sets, &boundary_defined);
      if (boundary_defined) {
        // Assign BCs to the face sets on this boundary with an empty
        // BoundaryFaces object.
        for (PetscInt j = 0; j < num_face_sets; ++j) {
          ierr = ConditionsSetBCs(tdy->conditions, face_sets[j], bcs[i]); CHKERRQ(ierr);
          BoundaryFaces bfaces = {0};
          ierr = ConditionsSetBoundaryFaces(tdy->conditions, face_sets[j], bfaces); CHKERRQ(ierr);
        }
      } else {
        SETERRQ(comm, PETSC_ERR_USER,
                "A boundary is declared but not defined. Please specify the "
                "face sets for each boundary with -tdy_bc_BOUNDARY_NAME.");
      }
    }
  } else { // no named boundaries -- face set indices only
    if (tdy->options.discretization == FV_TPF) {
      // Look for boundary conditions assigned to these face sets.
      BoundaryConditions bcs[MAX_FACE_SETS];
      char *face_set_names[MAX_FACE_SETS];
      for (PetscInt i = 0; i < MAX_FACE_SETS; ++i) {
        ierr = TDyAlloc(sizeof(char) * 33, &face_set_names[i]); CHKERRQ(ierr);
        snprintf(face_set_names[i], 32, "%d", i);
      }
      ierr = GetBCsForBoundaries(tdy, MAX_FACE_SETS, face_set_names, bcs);
      CHKERRQ(ierr);
      for (PetscInt i = 0; i < MAX_FACE_SETS; ++i) {
        ierr = TDyFree(face_set_names[i]); CHKERRQ(ierr);
      }

      // Assign BCs that are actually defined on face sets.
      for (PetscInt i = 0; i < MAX_FACE_SETS; ++i) {
        // Flow BCs are always required, so we check their type to see whether
        // this face set has a BC assigned.
        if (bcs[i].flow_bc.type != UNDEFINED_FLOW_BC) {
          ierr = ConditionsSetBCs(tdy->conditions, i, bcs[i]); CHKERRQ(ierr);
          BoundaryFaces bfaces = {0};
          ierr = ConditionsSetBoundaryFaces(tdy->conditions, i, bfaces); CHKERRQ(ierr);
        }
      }
    } else {
      // We fall back on our single-boundary-condition options.
      // FIXME: Broken at the moment.
      SETERRQ(comm, PETSC_ERR_USER,
              "Boundary conditions are currently broken for the given discretization!");
    }
  }

  PetscFunctionReturn(0);
}

/// Reads command line options specified by the user
///
/// @param [inout] tdy A TDy struct
///
/// @returns 0 on success, or a non-zero error code on failure
static PetscErrorCode ReadCommandLineOptions(TDy tdy) {

  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) != 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDySetFromOptions before TDySetup()");
  }

  // Collect options from command line arguments.
  TDyOptions *options = &tdy->options;

  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  PetscObjectOptionsBegin((PetscObject)tdy);

  TDyMode mode = options->mode;
  TDyDiscretization discretization = options->discretization;

  // Material property options
  PetscOptionsBegin(comm,NULL,"TDyCore: Material property options","");
  ierr = PetscOptionsReal("-tdy_porosity", "Value of porosity", NULL, options->porosity, &options->porosity, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_permability", "Value of permeability", NULL, options->permeability, &options->permeability, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_soil_density", "Value of soil density", NULL, options->soil_density, &options->soil_density, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_soil_specific_heat", "Value of soil specific heat", NULL, options->soil_specific_heat, &options->soil_specific_heat, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_thermal_conductivity", "Value of thermal conductivity", NULL, options->porosity, &options->porosity, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  // Characteristic curve options
  PetscOptionsBegin(comm,NULL,"TDyCore: Characteristic curve options","");
  ierr = PetscOptionsReal("-tdy_residual_satuaration", "Value of residual saturation", NULL, options->residual_saturation, &options->residual_saturation, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_gardner_param_n", "Value of Gardner n parameter", NULL, options->gardner_n, &options->gardner_n, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_vangenuchten_param_m", "Value of VanGenuchten m parameter", NULL, options->vangenuchten_m, &options->vangenuchten_m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_vangenuchten_param_alpha", "Value of VanGenuchten alpha parameter", NULL, options->vangenuchten_alpha, &options->vangenuchten_alpha, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  // Model options
  PetscOptionsBegin(comm,NULL,"TDyCore: Model options","");
  ierr = PetscOptionsEnum("-tdy_mode","Flow mode", "TDySetMode",TDyModes,(PetscEnum)options->mode, (PetscEnum *)&mode, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tdy_gravity", "Magnitude of gravity vector", NULL, options->gravity_constant, &options->gravity_constant, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_water_density","Water density vertical profile", "TDySetWaterDensityType", TDyWaterDensityTypes, (PetscEnum)options->rho_type, (PetscEnum *)&options->rho_type, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_water_viscosity","Water viscosity model", "TDySetWaterViscosityType", TDyWaterViscosityTypes, (PetscEnum)options->mu_type, (PetscEnum *)&options->mu_type, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tdy_water_enthalpy","Water enthalpy model", "TDySetWaterEnthalpyType", TDyWaterEnthalpyTypes, (PetscEnum)options->enthalpy_type, (PetscEnum *)&options->enthalpy_type, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  // Numerics options
  PetscOptionsBegin(comm,NULL,"TDyCore: Numerics options","");
  ierr = PetscOptionsEnum("-tdy_discretization","Discretization", "TDySetDiscretization",TDyDiscretizations, (PetscEnum)options->discretization,(PetscEnum *)&discretization, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  // Override the mode and/or discretization if needed.
  if (options->mode != mode) {
    TDySetMode(tdy, mode);
  }
  if (options->discretization != discretization) {
    TDySetDiscretization(tdy, discretization);
  }

  // Create boundary conditions.
  ierr = ProcessBCOptions(tdy); CHKERRQ(ierr);

  ierr = PetscOptionsBool("-tdy_init_with_random_field","Initialize solution with a random field","",options->init_with_random_field,&(options->init_with_random_field),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_init_file", options->init_file,sizeof(options->init_file),&options->init_from_file); CHKERRQ(ierr);

  if (options->init_from_file && options->init_with_random_field) {
    SETERRQ(comm,PETSC_ERR_USER, "Only one of -tdy_init_from_file and -tdy_init_with_random_field can be specified");
  }

  // Mesh-related options
  PetscOptionsBegin(comm,NULL,"TDyCore: Mesh options","");
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_read_mesh", options->mesh_file,sizeof(options->mesh_file),&options->read_mesh); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_read_pflotran_mesh", options->mesh_file,sizeof(options->mesh_file),&options->read_pflotran_mesh); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tdy_output_mesh","Enable output of mesh attributes","",options->output_mesh,&(options->output_mesh),NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  // Other options
  ierr = PetscOptionsBool("-tdy_regression_test","Enable output of a regression file","",options->regression_testing,&(options->regression_testing),NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode ExtractBoundaryFaces(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  DM dm = ((tdy->discretization)->tdydm)->dm;

  // Get the indices of the face sets within the DM.
  IS fs_is;
  ierr = DMGetLabelIdIS(dm, "Face Sets", &fs_is); CHKERRQ(ierr);
  PetscInt num_face_sets;
  ierr = ISGetSize(fs_is, &num_face_sets); CHKERRQ(ierr);

  // If the requested number of boundaries and/or face sets isn't in the
  // mesh, we report an error.
  PetscInt num_req_face_sets = ConditionsNumFaceSets(tdy->conditions);
  if (num_face_sets < num_req_face_sets) {
    SETERRQ(comm, PETSC_ERR_USER,
      "number of requested face sets exceeds number of labels in the mesh!");
  }

  // Extract boundary faces from the face sets.
  const PetscInt *face_sets;
  ierr = ISGetIndices(fs_is, &face_sets); CHKERRQ(ierr);
  PetscInt req_face_sets[num_face_sets];
  ierr = ConditionsGetAllBoundaryFaces(tdy->conditions, req_face_sets, NULL); CHKERRQ(ierr);
  ierr = ISGetIndices(fs_is, &face_sets); CHKERRQ(ierr);
  for (PetscInt f = 0; f < num_req_face_sets; ++f) {
    PetscInt face_set = req_face_sets[f];

    // Make sure this face set is in the mesh!
    PetscInt index;
    ierr = PetscFindInt(face_set, num_face_sets, face_sets, &index);
    if (index < 0) { // not found
      SETERRQ(comm, PETSC_ERR_USER,
        "a requested face set was not found in the given DM.");
    }

    BoundaryFaces bfaces;
    ierr = BoundaryFacesCreate(dm, face_set, &bfaces); CHKERRQ(ierr);
    ierr = ConditionsSetBoundaryFaces(tdy->conditions, face_set, bfaces); CHKERRQ(ierr);
  }

  ierr = ISDestroy(&fs_is); CHKERRQ(ierr);
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

  //------------------------------------------
  // Set options using command line parameters
  //------------------------------------------
  ierr = ReadCommandLineOptions(tdy); CHKERRQ(ierr);

  // Now that we know the discretization, we can create our implementation-
  // specific context.
  ierr = tdy->ops->create(&tdy->context); CHKERRQ(ierr);

  // Mode/discretization-specific options.
  if (tdy->ops->set_from_options) {
    ierr = tdy->ops->set_from_options(tdy->context, &tdy->options); CHKERRQ(ierr);
  }

  // Wrap up and indicate that options are set.
  tdy->setup_flags |= TDyOptionsSet;

  // Create our DM.
  if (tdy->options.read_pflotran_mesh) {
    PetscInt ndof = tdy->ops->get_num_dm_fields(tdy->context);
    ierr = TDyDiscretizationCreateFromPFLOTRANMesh(tdy->options.mesh_file, ndof, tdy->discretization); CHKERRQ(ierr);
  } else if (!((tdy->discretization)->tdydm)->dm) {
    DM dm;
    if (tdy->options.read_mesh) {
      ierr = DMPlexCreateFromFile(comm, tdy->options.mesh_file, "tdycore-dm",
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
      ierr = DMDestroy(&dm); CHKERRQ(ierr);
      dm = dm_dist;
    }
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
    ((tdy->discretization)->tdydm)->dm = dm;
    //ierr = DMClone(dm, &((tdy->discretization)->tdydm)->dm); CHKERRQ(ierr);

    // Mark the grid's boundary faces and their transitive closure. All are
    // stored at their appropriate strata within the label.
    DMLabel boundary_label;
    ierr = DMCreateLabel(((tdy->discretization)->tdydm)->dm, "boundary"); CHKERRQ(ierr);
    ierr = DMGetLabel(((tdy->discretization)->tdydm)->dm, "boundary", &boundary_label); CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(((tdy->discretization)->tdydm)->dm, 1, boundary_label); CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(((tdy->discretization)->tdydm)->dm, boundary_label); CHKERRQ(ierr);
  }

  // Now that the DM has been created, fish out the indices of boundary
  // faces belonging to face sets.
  ierr = ExtractBoundaryFaces(tdy); CHKERRQ(ierr);

  // Create an empty material properties object. Each function must be set
  // explicitly by the driver program.
  PetscInt dim = 3;
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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"TDycore setup\n"); CHKERRQ(ierr);

  // Set EOS parameters from options.
  tdy->eos.density_type = tdy->options.rho_type;
  tdy->eos.viscosity_type = tdy->options.mu_type;
  tdy->eos.enthalpy_type = tdy->options.enthalpy_type;

  // Perform implementation-specific setup.
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = tdy->ops->setup(tdy->context, tdy->discretization, &tdy->eos, tdy->matprop,
                         tdy->cc, tdy->conditions); CHKERRQ(ierr);

  // If we support diagnostics, set up the DMs and the diagnostic vector.
  if ( ((tdy->discretization)->tdydm)->dmtype == TDYCORE_DM_TYPE) {
    tdy->ops->update_diagnostics = NULL;
  }

  if (tdy->ops->update_diagnostics) {
    // Create a DM for diagnostic fields, and set its layout.
    ierr = DMClone(dm, &tdy->diag_dm); CHKERRQ(ierr);

    // We define two cell-centered scalar fields: saturation and liquid mass
    PetscInt num_fields = 2;
    ierr = DMSetNumFields(tdy->diag_dm, num_fields); CHKERRQ(ierr);
    PetscInt num_comp[2] = {1, 1};
    // Assign a single DOF to cells for each field.
    PetscInt dim = 3;
    PetscInt num_dof[num_fields*(dim+1)];
    memset(num_dof, 0, sizeof(PetscInt)*num_fields*(dim+1));
    num_dof[0*(dim+1)+dim] = 1;
    num_dof[1*(dim+1)+dim] = 1;
    PetscSection section;
    ierr = DMPlexCreateSection(tdy->diag_dm, NULL, num_comp, num_dof, 0, NULL, NULL,
                               NULL, NULL, &section); CHKERRQ(ierr);

    // Add names to the fields.
    ierr = PetscSectionSetFieldName(section, DIAG_LIQUID_SATURATION, "LiquidSaturation");
    CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, DIAG_LIQUID_MASS, "LiquidMass");
    CHKERRQ(ierr);
    ierr = DMSetLocalSection(tdy->diag_dm, section); CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);

    // Create a Vec that can store the diagnostic fields.
    ierr = DMCreateGlobalVector(tdy->diag_dm, &tdy->diag_vec); CHKERRQ(ierr);
  }

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

  // Are we enforcing boundary conditions on than one face set? We only support
  // this in TPF mode.
  int num_face_sets = ConditionsNumFaceSets(tdy->conditions);
  if ((num_face_sets > 1) && (discretization != FV_TPF)) {
    SETERRQ(comm,PETSC_ERR_USER,
      "Multiple face sets have been referenced, but only FV_TPF supports this!");
  }

  // If we are resetting the mode and have already set the discretization,
  // we call TDySetDiscretization again.
  if (tdy->setup_flags & TDyDiscretizationSet) {
    PetscInt ierr = TDySetDiscretization(tdy, tdy->options.discretization); CHKERRQ(ierr);
  }

  // Set function pointers for operations.
  if (tdy->options.mode == RICHARDS) {
    if (discretization == MPFA_O) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_MPFAO;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_Richards_MPFAO;
      tdy->ops->setup = TDySetup_Richards_MPFAO;
      tdy->ops->update_state = TDyUpdateState_Richards_MPFAO;
      tdy->ops->update_diagnostics = TDyUpdateDiagnostics_MPFAO;
    } else if (discretization == MPFA_O_DAE) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_MPFAO_DAE;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_Richards_MPFAO_DAE;
      tdy->ops->setup = TDySetup_Richards_MPFAO_DAE;
      tdy->ops->update_state = TDyUpdateState_Richards_MPFAO;
      tdy->ops->update_diagnostics = TDyUpdateDiagnostics_MPFAO;
    } else if (discretization == MPFA_O_TRANSIENTVAR) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_MPFAO;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_Richards_MPFAO;
      tdy->ops->setup = TDySetup_Richards_MPFAO;
      tdy->ops->update_state = TDyUpdateState_Richards_MPFAO;
      tdy->ops->update_diagnostics = TDyUpdateDiagnostics_MPFAO;
    } else if (discretization == BDM) {
      tdy->ops->create = TDyCreate_BDM;
      tdy->ops->destroy = TDyDestroy_BDM;
      tdy->ops->set_from_options = TDySetFromOptions_BDM;
      tdy->ops->setup = TDySetup_BDM;
      tdy->ops->set_dm_fields = TDySetDMFields_BDM;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_BDM;
      tdy->ops->update_state = NULL; // FIXME: ???
      tdy->ops->update_diagnostics = NULL; // FIXME
    } else if (discretization == WY) {
      tdy->ops->create = TDyCreate_WY;
      tdy->ops->destroy = TDyDestroy_WY;
      tdy->ops->set_from_options = TDySetFromOptions_WY;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_BDM;
      tdy->ops->set_dm_fields = TDySetDMFields_WY;
      tdy->ops->setup = TDySetup_WY;
      tdy->ops->update_state = TDyUpdateState_WY;
      tdy->ops->update_diagnostics = NULL; // FIXME
    } else if (discretization == FV_TPF) {
      tdy->ops->create = TDyCreate_FVTPF;
      tdy->ops->destroy = TDyDestroy_FVTPF;
      tdy->ops->set_from_options = TDySetFromOptions_FVTPF;
      tdy->ops->set_dm_fields = TDySetDMFields_Richards_FVTPF;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_Richards_FVTPF;
      tdy->ops->setup = TDySetup_Richards_FVTPF;
      tdy->ops->update_state = TDyUpdateState_Richards_FVTPF;
      tdy->ops->update_diagnostics = TDyUpdateDiagnostics_FVTPF;
    } else {
      SETERRQ(comm,PETSC_ERR_USER, "Invalid discretization given!");
    }
  } else if (tdy->options.mode == TH) {
    PetscPrintf(PETSC_COMM_WORLD,"Running TH mode.\n");
    if (discretization == MPFA_O) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_TH_MPFAO;
      tdy->ops->setup = TDySetup_TH_MPFAO;
      tdy->ops->update_state = TDyUpdateState_TH_MPFAO;
      tdy->ops->update_diagnostics = TDyUpdateDiagnostics_MPFAO;
    } else {
      SETERRQ(comm,PETSC_ERR_USER,
        "The TH mode does not support the selected discretization!");
    }
  } else if (tdy->options.mode == SALINITY) {
    PetscPrintf(PETSC_COMM_WORLD,"Running SALINITY mode.\n");
    if (discretization == MPFA_O) {
      tdy->ops->create = TDyCreate_MPFAO;
      tdy->ops->destroy = TDyDestroy_MPFAO;
      tdy->ops->set_from_options = TDySetFromOptions_MPFAO;
      tdy->ops->get_num_dm_fields = TDyGetNumDMFields_Salinity_MPFAO;
      tdy->ops->set_dm_fields = TDySetDMFields_Salinity_MPFAO;
      tdy->ops->setup = TDySetup_Salinity_MPFAO;
      tdy->ops->update_state = TDyUpdateState_Salinity_MPFAO;
      tdy->ops->update_diagnostics = TDyUpdateDiagnostics_MPFAO;
    } else {
      SETERRQ(comm,PETSC_ERR_USER,
        "The SALINITY mode does not support the selected discretization!");
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

PetscErrorCode TDySetWaterViscosityType(TDy tdy, TDyWaterViscosityType vistype) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  tdy->options.mu_type = vistype;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetWaterEnthalpyType(TDy tdy, TDyWaterEnthalpyType enthtype) {
  PetscValidPointer(tdy,1);
  PetscFunctionBegin;
  tdy->options.enthalpy_type = enthtype;
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

/// Sets the saline diffusivity used by the dycore to the given constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantIsotropicSalineDiffusivity(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantIsotropicSalineDiffusivity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute (isotropic) saline diffusivities in the
/// dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetIsotropicSalineDiffusivityFunction(TDy tdy,
                                                        TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousIsotropicSalineDiffusivity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the saline diffusivity used by the dycore to the given diagonal constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantDiagonalSalineDiffusivity(TDy tdy, PetscReal value[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantDiagonalSalineDiffusivity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute (diagonal anisotropic) saline
/// diffusivities in the dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetDiagonalSalineDiffusivityFunction(TDy tdy,
                                                       TDyVectorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousDiagonalSalineDiffusivity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the saline diffusivity used by the dycore to the given tensor constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantTensorSalineDiffusivity(TDy tdy, PetscReal value[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantTensorSalineDiffusivity(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute anisotropic saline diffusivities in the
/// dycore. May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] f the function to use
PetscErrorCode TDySetTensorSalineDiffusivityFunction(TDy tdy,
                                                     TDyTensorSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousTensorSalineDiffusivity(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the saline molecular weight used by the dycore to the given constant.
/// May be called anytime after TDySetFromOptions.
/// @param [in] tdy the dycore instance
/// @param [in] value the constant value to use
PetscErrorCode TDySetConstantSalineMolecularWeight(TDy tdy, PetscReal value) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetConstantSalineMolecularWeight(tdy->matprop, value); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute saline molecular weights in the dycore.
/// May be called anytime after TDySetFromOptions.
PetscErrorCode TDySetSalineMolecularWeightFunction(TDy tdy,
                                                   TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MaterialPropSetHeterogeneousSalineMolecularWeight(tdy->matprop, f);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetForcingFunction(TDy tdy,
                                     TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  wrapper->func = f;
  ierr = ConditionsSetForcing(tdy->conditions, wrapper,
                              WrapperFunction, TDyFree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetEnergyForcingFunction(TDy tdy,
                                           TDyScalarSpatialFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  wrapper->func = f;
  ierr = ConditionsSetEnergyForcing(tdy->conditions, wrapper,
                                    WrapperFunction, TDyFree); CHKERRQ(ierr);
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

/// Sets a flow boundary condition of the given type (implemented by the
/// given scalar spatial function) on all faces belonging to the face set with
/// the given index.
/// @param [in] tdy the dycore instance
/// @param [in] face_set the index of the face set identifying boundary faces
/// @param [in] bc_type the type of the desired flow boundary condition
/// @param [in] func the scalar spatial function implementing the condition
PetscErrorCode TDySetFlowBCFunction(TDy tdy,
                                    PetscInt face_set,
                                    TDyFlowBCType bc_type,
                                    TDyScalarSpatialFunction func) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  BoundaryConditions bcs;
  ierr = ConditionsGetBCs(tdy->conditions, face_set, &bcs); CHKERRQ(ierr);
  if (bcs.flow_bc.context && bcs.flow_bc.dtor) {
    bcs.flow_bc.dtor(bcs.flow_bc.context);
  }
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  bcs.flow_bc = (FlowBC){
    .type = bc_type,
    .context = wrapper,
    .compute = WrapperFunction,
    .dtor = TDyFree
  };
  ierr = ConditionsSetBCs(tdy->conditions, face_set, bcs); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets a thermal boundary condition of the given type (implemented by the
/// given scalar spatial function) on all faces belonging to the face set with
/// the given index.
/// @param [in] tdy the dycore instance
/// @param [in] face_set the index of the face set identifying boundary faces
/// @param [in] bc_type the type of the desired thermal boundary condition
/// @param [in] func the scalar spatial function implementing the condition
PetscErrorCode TDySetThermalBCFunction(TDy tdy,
                                       PetscInt face_set,
                                       TDyThermalBCType bc_type,
                                       TDyScalarSpatialFunction func) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  BoundaryConditions bcs;
  ierr = ConditionsGetBCs(tdy->conditions, face_set, &bcs); CHKERRQ(ierr);
  if (bcs.thermal_bc.context && bcs.thermal_bc.dtor) {
    bcs.thermal_bc.dtor(bcs.thermal_bc.context);
  }
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  bcs.thermal_bc = (ThermalBC){
    .type = bc_type,
    .context = wrapper,
    .compute = WrapperFunction,
    .dtor = TDyFree
  };
  ierr = ConditionsSetBCs(tdy->conditions, face_set, bcs); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets a salinity boundary condition of the given type (implemented by the
/// given scalar spatial function) on all faces belonging to the face set with
/// the given index.
/// @param [in] tdy the dycore instance
/// @param [in] face_set the index of the face set identifying boundary faces
/// @param [in] bc_type the type of the desired salinity boundary condition
/// @param [in] func the scalar spatial function implementing the condition
PetscErrorCode TDySetSalinityBCFunction(TDy tdy,
                                        PetscInt face_set,
                                        TDySalinityBCType bc_type,
                                        TDyScalarSpatialFunction func) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  BoundaryConditions bcs;
  ierr = ConditionsGetBCs(tdy->conditions, face_set, &bcs); CHKERRQ(ierr);
  if (bcs.salinity_bc.context && bcs.salinity_bc.dtor) {
    bcs.salinity_bc.dtor(bcs.salinity_bc.context);
  }
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  bcs.salinity_bc = (SalinityBC){
    .type = bc_type,
    .context = wrapper,
    .compute = WrapperFunction,
    .dtor = TDyFree
  };
  ierr = ConditionsSetBCs(tdy->conditions, face_set, bcs); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Selects a flow boundary condition of the given type (implemented by the
/// scalar spatial function registered with the given name) on all faces
/// belonging to the face set with the given index.
/// @param [in] tdy the dycore instance
/// @param [in] face_set the index of the face set identifying boundary faces
/// @param [in] bc_type the type of the desired flow boundary condition
/// @param [in] func_name the registered name of a scalar spatial function
///                       that implements the desired boundary condition
PetscErrorCode TDySelectFlowBCFunction(TDy tdy,
                                       PetscInt face_set,
                                       TDyFlowBCType bc_type,
                                       const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetFlowBCFunction(tdy, face_set, bc_type, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Selects a thermal boundary condition of the given type (implemented by the
/// scalar spatial function registered with the given name) on all faces
/// belonging to the face set with the given index.
/// @param [in] tdy the dycore instance
/// @param [in] face_set the index of the face set identifying boundary faces
/// @param [in] bc_type the type of the desired thermal boundary condition
/// @param [in] func_name the registered name of a scalar spatial function
///                       that implements the desired boundary condition
PetscErrorCode TDySelectThermalBCFunction(TDy tdy,
                                          PetscInt face_set,
                                          TDyThermalBCType bc_type,
                                          const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetThermalBCFunction(tdy, face_set, bc_type, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Selects a salinity boundary condition of the given type (implemented by the
/// scalar spatial function registered with the given name) on all faces
/// belonging to the face set with the given index.
/// @param [in] tdy the dycore instance
/// @param [in] face_set the index of the face set identifying boundary faces
/// @param [in] bc_type the type of the desired salinity boundary condition
/// @param [in] func_name the registered name of a scalar spatial function
///                       that implements the desired boundary condition
PetscErrorCode TDySelectSalinityBCFunction(TDy tdy,
                                           PetscInt face_set,
                                           TDySalinityBCType bc_type,
                                           const char *func_name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDySpatialFunction f;
  ierr = TDyGetFunction(func_name, &f); CHKERRQ(ierr);
  ierr = TDySetSalinityBCFunction(tdy, face_set, bc_type, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Updates the secondary state of the system using the solution data.
/// @param [inout] tdy a dycore object
/// @param [in] U an array of solution data
PetscErrorCode TDyUpdateState(TDy tdy,PetscReal *U, PetscInt num_cells) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDyEnterProfilingStage("TDycore Setup");
  TDY_START_FUNCTION_TIMER()

  // Call the implementation-specific state update.
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = tdy->ops->update_state(tdy->context, dm, &tdy->eos, tdy->matprop,
                                tdy->cc, num_cells, U); CHKERRQ(ierr);

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
PetscErrorCode TDyNaturalToGlobal(TDy tdy, Vec natural, Vec global) {

  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = TDyDiscretizationNaturalToGlobal(tdy->discretization, natural, global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyGlobalToLocal(TDy tdy, Vec global, Vec local) {

  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = TDyDiscretizationGlobalToLocal(tdy->discretization, global, local); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyGlobalToNatural(TDy tdy, Vec global, Vec natural) {

  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = TDyDiscretizationGlobalToNatural(tdy->discretization, global, natural); CHKERRQ(ierr);

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
  ierr = TDyNaturalToGlobal(tdy,initial,tdy->soln); CHKERRQ(ierr);
  ierr = TDyNaturalToGlobal(tdy,initial,tdy->soln_prev); CHKERRQ(ierr);
  ierr = TDyDiscretizationNaturaltoLocal(tdy->discretization,initial,&(tdy->soln_loc)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Gets initial condition for the TDy solver
///
/// @param [in] tdy A TDy struct
/// @param [out] initial A copies initial condition into a PETSc vector.
///                     For RICHARDS mode, the vector contains unknown
///                     pressure values for each grid cell.
///                     For TH mode, the vector contains unknown pressure
///                     and temperature values for each grid cell.
///                     The initial condition returned is in PETSc's global
///                     numbering order.
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyGetInitialCondition(TDy tdy, Vec initial) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCopy(tdy->soln,initial); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


//-------------------------------
// Diagnostics-related functions
//-------------------------------

/// This function updates all diagnostic fields for the model. These diagnostic
/// fields (e.g. liquid mass, saturation) can be extracted by creating
/// individual diagnostic vectors with TDyCreateDiagnosticVector and calling
/// e.g. TDyGetSaturation to store the saturation field in such a vector.
/// @param [in] tdy A TDy object
PetscErrorCode TDyUpdateDiagnostics(TDy tdy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDyUpdateDiagnostics after TDySetup()");
  } else if (!tdy->ops->update_diagnostics) {
    SETERRQ(comm,PETSC_ERR_USER,"TDyUpdateDiagnostics is not supported by this implementation.");
  }

  // Update the diagnostic fields.
  ierr = tdy->ops->update_diagnostics(tdy->context, tdy->diag_dm, tdy->diag_vec);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a Vec that can store a cell-centered scalar diagnostic field such as
/// the saturation or liquid mass.
/// @param [in] tdy A TDy object
/// @param [out] diag_vec A Vec that can store a diagnostic field.
PetscErrorCode TDyCreateDiagnosticVector(TDy tdy, Vec *diag_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDyCreateDiagnosticVector after TDySetup()");
  } else if (!tdy->ops->update_diagnostics) {
    SETERRQ(comm,PETSC_ERR_USER,"Diagnostic fields are not supported by this implementation.");
  }

  // Create a cell-centered scalar field vector.
  PetscInt vecsize, blocksize;
  ierr = VecGetLocalSize(tdy->diag_vec, &vecsize); CHKERRQ(ierr);
  ierr = VecGetBlockSize(tdy->diag_vec, &blocksize); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, (PetscInt)(vecsize/blocksize), PETSC_DECIDE, diag_vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateGlobalVector(TDy tdy, Vec *vec) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  
  ierr = TDyDiscretizationCreateGlobalVector(tdy->discretization, vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateLocalVector(TDy tdy, Vec *vec) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  
  ierr = TDyDiscretizationCreateLocalVector(tdy->discretization, vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a Vec that can store a cell-centered scalar prognostic field such as
/// the saturation or liquid mass.
/// @param [in] tdy A TDy object
/// @param [out] prog_vec A Vec that can store a prognostic field.
PetscErrorCode TDyCreatePrognosticVector(TDy tdy, Vec *prog_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDyCreatePrognosticVector after TDySetup()");
  }

  // Create a cell-centered scalar field vector.
  ierr = TDyCreateGlobalVector(tdy, prog_vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ExtractDiagnosticField(TDy tdy, PetscInt index, Vec vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"Diagnostic fields cannot be extracted before TDySetup()");
  } else if (!tdy->ops->update_diagnostics) {
    SETERRQ(comm,PETSC_ERR_USER,"Diagnostic fields are not supported by this implementation.");
  }

  // Extract the field.
  ierr = VecStrideGather(tdy->diag_vec, index, vec, INSERT_VALUES);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a matrix
/// @param [in] tdy A TDy object
/// @param [out] mat A matrix
PetscErrorCode TDyCreateMatrix(TDy tdy, Mat *mat) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"You must call TDyCreateMatrix after TDySetup()");
  }

  // Create a cell-centered scalar field vector.
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, mat); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Given a Vec created with TDyCreateDiagnosticVector, populates that vector
/// with the saturation values for each cell in the grid.
/// @param [in] tdy A TDy object
/// @param [out] sat_vec The Vec that stores the cell-centered saturation field.
///                      The values in the Vec are in natural-order.
PetscErrorCode TDyGetLiquidSaturation(TDy tdy, Vec sat_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec tmp_vec;
  ierr = TDyCreateDiagnosticVector(tdy, &tmp_vec); CHKERRQ(ierr);
  ierr = ExtractDiagnosticField(tdy, DIAG_LIQUID_SATURATION, tmp_vec); CHKERRQ(ierr);
  ierr = TDyGlobalToNatural(tdy, tmp_vec, sat_vec);CHKERRQ(ierr); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_vec); CHKERRQ(ierr);

  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Given a Vec created with TDyCreateDiagnosticVector, populates that vector
/// with the liquid mass values for each cell in the grid.
/// @param [in] tdy A TDy object
/// @param [out] mass_vec The Vec that stores the cell-centered liquid mass field.
///                       The values in the Vec are in natural-order.
PetscErrorCode TDyGetLiquidMass(TDy tdy, Vec mass_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec tmp_vec;
  ierr = TDyCreateDiagnosticVector(tdy, &tmp_vec); CHKERRQ(ierr);
  ierr = ExtractDiagnosticField(tdy, DIAG_LIQUID_MASS, tmp_vec); CHKERRQ(ierr);
  ierr = TDyGlobalToNatural(tdy, tmp_vec, mass_vec);CHKERRQ(ierr); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_vec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ExtractPrognosticField(TDy tdy, PetscInt index, Vec vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)tdy, &comm); CHKERRQ(ierr);

  if ((tdy->setup_flags & TDySetupFinished) == 0) {
    SETERRQ(comm,PETSC_ERR_USER,"Prognostic fields cannot be extracted before TDySetup()");
  }

  // Extract the field.
  ierr = VecStrideGather(tdy->soln_prev, index, vec, INSERT_VALUES);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Given a Vec created with TDyCreatePrognosticVector, populates that vector
/// with the saturation values for each cell in the grid.
/// @param [in] tdy A TDy object
/// @param [out] liq_pres_vec The Vec that stores the cell-centered liquid pressure field.
///                           The values in the Vec are in natural-order.
PetscErrorCode TDyGetLiquidPressure(TDy tdy, Vec liq_press_vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec tmp_vec;
  ierr = TDyCreateGlobalVector(tdy, &tmp_vec); CHKERRQ(ierr);
  ierr = ExtractPrognosticField(tdy, VAR_PRESSURE, tmp_vec); CHKERRQ(ierr);
  ierr = TDyGlobalToNatural(tdy, tmp_vec, liq_press_vec);CHKERRQ(ierr); CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_vec); CHKERRQ(ierr);

  CHKERRQ(ierr);
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
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  ierr = DMGetLocalSection(dm, &sec);
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
    case SALINITY:
      SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for Salinity");
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
  case FV_TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"IFunction not implemented for FV_TPF");
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
    case SALINITY:
      //ierr = TSSetIJacobian(ts,tdy->J,tdy->J,TDyMPFAOIJacobian,tdy); CHKERRQ(ierr);
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
  case FV_TPF:
    SETERRQ(comm,PETSC_ERR_SUP,"IJacobian not implemented for FV_TPF");
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
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  switch (tdy->options.discretization) {
    case MPFA_O:
      switch (tdy->options.mode) {
        case RICHARDS:
          ierr = SNESSetFunction(snes,tdy->residual,TDyMPFAOSNESFunction,tdy); CHKERRQ(ierr);
          break;
        case TH:
          ierr = SNESSetFunction(snes,tdy->residual,TDyMPFAOSNESFunction,tdy); CHKERRQ(ierr);
          break;
        case SALINITY:
          ierr = SNESSetFunction(snes,tdy->residual,TDyMPFAOSNESFunction_Salinity,tdy); CHKERRQ(ierr);
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
    case FV_TPF:
      ierr = SNESSetFunction(snes,tdy->residual,TDyFVTPFSNESFunction,tdy); CHKERRQ(ierr);
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
  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  PetscInt dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

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
  case FV_TPF:
    ierr = SNESSetJacobian(snes,tdy->J,tdy->J,TDyFVTPFSNESJacobian,tdy); CHKERRQ(ierr);
    break;
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyPostSolve(TDy tdy,Vec U) {

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

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  switch (tdy->options.discretization) {
  case MPFA_O:
    switch (tdy->options.mode) {
      case RICHARDS:
        ierr = TDyMPFAOSNESPreSolve(tdy); CHKERRQ(ierr);
        break;
      case TH:
        SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for TH");
        break;
      case SALINITY:
        ierr = TDyMPFAOSNESPreSolve_Salinity(tdy); CHKERRQ(ierr);
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
  case FV_TPF:
    ierr = TDyFVTPFSNESPreSolve(tdy); CHKERRQ(ierr);
    break;
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/// Resets solver when a time step is cut
/// @param [inout] TDy struct
///
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyTimeCut(TDy tdy) {
  PetscInt dim;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  DM dm;
  ierr = TDyGetDM(tdy, &dm); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);

  switch (tdy->options.discretization) {
  case MPFA_O:
    SETERRQ(comm,PETSC_ERR_SUP,"TDyPreSolveSNESSolver not implemented for MPFA_O");
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
  case FV_TPF:
    ierr = TDyFVTPFSNESTimeCut(tdy); CHKERRQ(ierr);
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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating Vectors\n");
  if (tdy->soln == NULL) {
    ierr = TDyCreateGlobalVector(tdy, &tdy->soln); CHKERRQ(ierr);
    ierr = VecDuplicate(tdy->soln,&tdy->residual); CHKERRQ(ierr);
    ierr = VecDuplicate(tdy->soln,&tdy->accumulation_prev); CHKERRQ(ierr);
    ierr = VecDuplicate(tdy->soln,&tdy->soln_prev); CHKERRQ(ierr);

    ierr = TDyCreateLocalVector(tdy,&tdy->soln_loc); CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating Jacobian matrix\n");
  if (tdy->J == NULL) {
    ierr = TDyDiscretizationCreateJacobianMatrix(tdy->discretization,&tdy->J); CHKERRQ(ierr);
    ierr = TDyDiscretizationCreateJacobianMatrix(tdy->discretization,&tdy->Jpre); CHKERRQ(ierr);
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
