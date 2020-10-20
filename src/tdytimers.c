#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdymeshimpl.h>

// Timers registry.
khash_t(TDY_TIMER_MAP)* TDY_TIMERS = NULL;

// Profiling stages registry.
khash_t(TDY_PROFILING_STAGE_MAP)* TDY_PROFILING_STAGES = NULL;

// Are timers enabled?
static PetscBool timersEnabled_ = PETSC_FALSE;

// Timing metadata used by tdyperfplot and other tools.
typedef struct {
  TDyMethod method;
  TDyMode mode;
  int num_cells;
} TimingMetadata;
static TimingMetadata metadata_;

PetscErrorCode TDyInitTimers() {
  // Register timers table.
  if (TDY_TIMERS == NULL)
    TDY_TIMERS = kh_init(TDY_TIMER_MAP);

  // Register profiling stages table.
  if (TDY_PROFILING_STAGES == NULL)
    TDY_PROFILING_STAGES = kh_init(TDY_PROFILING_STAGE_MAP);

  // Register some logging stages.
  PetscErrorCode ierr;
  ierr = TDyAddProfilingStage("TDycore Setup"); CHKERRQ(ierr);
  ierr = TDyAddProfilingStage("TDycore Stepping"); CHKERRQ(ierr);
  ierr = TDyAddProfilingStage("TDycore I/O"); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode TDyEnableTimers() {
  timersEnabled_ = PETSC_TRUE;
  PetscLogDefaultBegin();
  return 0;
}

PetscErrorCode TDyAddProfilingStage(const char* name) {
  khiter_t iter = kh_get(TDY_PROFILING_STAGE_MAP, TDY_PROFILING_STAGES, name);
  PetscLogStage stage;
  if (iter == kh_end(TDY_PROFILING_STAGES)) {
    PetscLogStageRegister(name, &stage);
    int retval;
    iter = kh_put(TDY_PROFILING_STAGE_MAP, TDY_PROFILING_STAGES, name, &retval);
    kh_val(TDY_PROFILING_STAGES, iter) = stage;
  }
  return 0;
}

PetscErrorCode TDyWriteTimingProfile(const char* filename) {
  if (timersEnabled_) {
    PetscViewer log;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "tdycore_profile.csv", &log);
    PetscViewerFormat format = PETSC_VIEWER_ASCII_CSV;
    PetscViewerPushFormat(log, format);
    PetscLogView(log);
    PetscViewerDestroy(&log);

    // Add a footer to the profile CSV that contains useful metadata.
    const char* method_name;
    if (metadata_.method == TPF) {
      method_name = "TPF";
    } else if (metadata_.method == MPFA_O) {
      method_name = "MPFA_O";
    } else if (metadata_.method == MPFA_O_DAE) {
      method_name = "MPFA_O_DAE";
    } else if (metadata_.method == MPFA_O_TRANSIENTVAR) {
      method_name = "MPFA_O_TRANSIENTVAR";
    } else if (metadata_.method == BDM) {
      method_name = "BDM";
    } else { // (metadata_.method == BDM)
      method_name = "WY";
    }
    const char* mode_name;
    if (metadata_.mode == RICHARDS) {
      mode_name = "RICHARDS";
    } else { // (metadata_.mode == TH)
      mode_name = "TH";
    }
    FILE* f = fopen("tdycore_profile.csv", "a");
    fprintf(f, "METADATA\n");
    fprintf(f, "Method,Mode,NumCells\n");
    fprintf(f, "%s,%s,%d", method_name, mode_name, metadata_.num_cells);
    fclose(f);
    return 0;
  }
  return 0;
}

PetscErrorCode TDySetTimingMetadata(TDy tdy) {
  metadata_.method = tdy->method;
  metadata_.mode = tdy->mode;
  metadata_.num_cells = tdy->mesh->num_cells;
  return 0;
}

void TDyDestroyTimers() {
  // Free the timers registry and the profiling stages registry.
  if (TDY_TIMERS != NULL)
    kh_destroy(TDY_TIMER_MAP, TDY_TIMERS);
  if (TDY_PROFILING_STAGES != NULL)
    kh_destroy(TDY_PROFILING_STAGE_MAP, TDY_PROFILING_STAGES);
}
