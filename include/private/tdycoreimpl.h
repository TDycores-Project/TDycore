#if !defined(TDYCOREIMPL_H)
#define TDYCOREIMPL_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyregressionimpl.h>
#include <tdycore.h>
#include <tdyio.h>
#include <private/tdytiimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdyoptions.h>
#include <private/tdydmimpl.h>
#include <private/tdydiscretizationimpl.h>

#define VAR_PRESSURE 0
#define VAR_TEMPERATURE 1

// Diagnostic field names
#define DIAG_LIQUID_SATURATION 0
#define DIAG_LIQUID_MASS 1

// This type serves as a "virtual table" containing function pointers that
// define the behavior of the dycore.
typedef struct _TDyOps *TDyOps;
struct _TDyOps {
  // Called by TDyCreate to allocate implementation-specific resources. Returns
  // a pointer to a context.
  PetscErrorCode (*create)(void**);

  // Called by TDyDestroy to free implementation-specific resources.
  PetscErrorCode (*destroy)(void*);

  // Creates a DM for a particular simulation (optional).
  // Arguments:
  //   1. A valid MPI communicator
  //   2. A context pointer storing data specifically for the constructor
  //      function
  //   3. A location for the newly created DM.
  // We pass the dycore as the first argument here because we don't expect
  // the caller to know implementation details.
  PetscErrorCode (*create_dm)(MPI_Comm, void*, DM*);

  // Implements the view operation for the TDy implementation with the given
  // viewer.
  PetscErrorCode (*view)(void*, PetscViewer);

  // Called by TDySetFromOptions -- sets implementation-specific options from
  // command-line arguments (and possibly already-parsed options.
  PetscErrorCode (*set_from_options)(void*, TDyOptions*);

  // Sets up fields on the given DM. This process can involve a PetscFE object
  // or a PetscSection, for example. This is called by TDySetFromOptions just
  // before the DM is distributed across processes.
  PetscErrorCode (*set_dm_fields)(void*, DM);

  PetscInt (*get_num_dm_fields)(void*);

  // Called by TDySetup -- configures the DM for solvers. By the time this
  // function is called, the DM has its field layout defined and has been
  // distributed across processes.
  PetscErrorCode (*setup)(void*, DM, EOS*, MaterialProp*,
                          CharacteristicCurves*, Conditions*);

  // Called by TDyUpdateState -- updates the state maintained by the
  // implementation with provided solution data.
  PetscErrorCode (*update_state)(void*, DM, EOS*, MaterialProp*,
                                 CharacteristicCurves*, PetscInt, PetscReal*);

  // Called by TDyComputeErrorNorms -- computes error norms given a solution
  // vector.
  PetscErrorCode (*compute_error_norms)(void*,DM,Conditions*,Vec,PetscReal*,PetscReal*);

  // Updates diagnostic fields given an appropriate DM defining their layout,
  // and a multi-component diagnostics Vec created from that DM with
  // DMCreateLocalVector.
  PetscErrorCode (*update_diagnostics)(void*, DM, Vec);
};

// This type represents the dycore and all of its settings.
struct _p_TDy {
  PETSCHEADER(struct _TDyOps);

  // Implementation-specific context pointer
  void *context;

  // Flags that indicate where the dycore is in the setup process
  TDySetupFlags setup_flags;

  // Discretization that hold information about DM and grid
  TDyDiscretizationType discretization;

  // Contextual information passed to create_dm (if given).
  void* create_dm_context;

  // I/O subsystem
  TDyIO io;

  // options that determine the behavior(s) of the dycore
  TDyOptions options;

  // boundary conditions and sources/sinks
  Conditions *conditions;

  // equation of state
  EOS eos;

  // material properties
  MaterialProp *matprop;

  // characteristic curves
  CharacteristicCurves *cc;

  // regression testing data
  TDyRegression *regression;

  //-----------------------------
  // Diagnostic field management
  //-----------------------------

  // DM that holds layout for diagnostic fields (saturation, liquid mass, etc)
  DM diag_dm;

  // Vec that stores diagnostic fields
  Vec diag_vec;

  //------------------------------------------------------
  // Solver-specific information (should be factored out)
  //------------------------------------------------------
  Mat J, Jpre;

  TDyTimeIntegrator ti;

  /* For SNES based timestepping */
  PetscReal dtime;
  Vec soln;
  Vec soln_loc;
  Vec soln_prev;
  Vec accumulation_prev;
  Vec residual;

};

PETSC_INTERN PetscErrorCode TDyCreateGlobalVector(TDy,Vec*);
PETSC_INTERN PetscErrorCode TDyCreateLocalVector(TDy,Vec*);
PETSC_INTERN PetscErrorCode TDyNaturalToGlobal(TDy,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyGlobalToLocal(TDy,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyGlobalToNatural(TDy,Vec,Vec);

#endif
