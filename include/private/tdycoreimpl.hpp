#ifndef TDYCORE_TYDCOREIMPL_HPP
#define TDYCORE_TYDCOREIMPL_HPP

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
#define VAR_SALINE_CONCENTRATION 2

// Diagnostic field names
#define DIAG_LIQUID_SATURATION 0
#define DIAG_LIQUID_MASS 1

// This base class defines the interface for all TDycore implementations.
class TDycoreImpl {
  // Default constructor -- called by all subclasses.
  TDycoreImpl() {}

  // Destructor
  virtual ~TDycoreImpl();

  // Dycore implementations cannot be deep copied. We only use pointers to
  // these things.
  TDycoreImpl(const TDycoreImpl&) = delete;
  const TDycoreImpl& operator=(const TDycoreImpl&) = delete;

  // Creates a new DM by calling the function for a particular simulation, defined on the given MPI
  // communicator.
  DM create_dm(MPI_Comm comm) const;

  // Implements the view operation for the TDy implementation with the given
  // viewer.
  virtual void view(PetscViewer viewer) const = 0;

  // Called by TDySetFromOptions -- sets implementation-specific options from
  // command-line arguments (and possibly already-parsed options).
  virtual void set_from_options(TDyOptions& options) const = 0;

  // Sets up fields on the given DM. This process can involve a PetscFE object
  // or a PetscSection, for example. This is called by TDySetFromOptions just
  // before the DM is distributed across processes.
  virtual void set_dm_fields(DM dm) const = 0;

  // Returns implementation-specific number of DOFs
  virtual PetscInt num_dm_fields() const = 0;

  // Called by TDySetup -- configures the DM for solvers. By the time this
  // function is called, the DM has its field layout defined and has been
  // distributed across processes.
  virtual setup(TDyDiscretizationType*, EOS*, MaterialProp*,
                CharacteristicCurves*, Conditions*) = 0;

  // Called by TDyUpdateState -- updates the state maintained by the
  // implementation with provided solution data.
  virtual update_state(DM, EOS*, MaterialProp*,
                       CharacteristicCurves*, PetscInt, PetscReal*) = 0;

  // Called by TDyComputeErrorNorms -- computes error norms given a solution
  // vector.
  virtual compute_error_norms(DM,Conditions*,Vec,PetscReal*,PetscReal*) const = 0;

  // Updates diagnostic fields given an appropriate DM defining their layout,
  // and a multi-component diagnostics Vec created from that DM with
  // DMCreateLocalVector.
  virtual void update_diagnostics(DM, Vec) const = 0;
};

extern "C" {

// This type represents the dycore and all of its settings.
struct _p_TDy {
  PETSCHEADER(struct _TDyOps);

  // Specific implementation.
  TDycoreImpl *impl;

  // Flags that indicate where the dycore is in the setup process
  TDySetupFlags setup_flags;

  // Discretization that hold information about DM and grid
  TDyDiscretizationType *discretization;

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
PETSC_INTERN PetscErrorCode TDyTimeCut(TDy);

} // extern "C"

#endif
