#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdympfaoimpl.h>
#include <private/tdymeshimpl.h>

// Timers registry (maps timer names to PetscLogEvents).
khash_t(TDY_TIMER_MAP)* TDY_TIMERS = NULL;

// Profiling stages registry (maps stage names to PetscLogStages).
khash_t(TDY_PROFILING_STAGE_MAP)* TDY_PROFILING_STAGES = NULL;

// Timing metadata used by tdyperfplot and other tools.
typedef struct {
  TDyMode mode;
  TDyDiscretization discretization;
  int num_cells;
  int num_proc;
} TimingMetadata;

// Profiling metadata registry (maps TDy objects (pointers) to timing metadata).
KHASH_MAP_INIT_INT64(TDY_PROFILING_MD_MAP, TimingMetadata*)
khash_t(TDY_PROFILING_MD_MAP)* TDY_PROFILING_METADATA = NULL;

// Are timers enabled?
static PetscBool timersEnabled_ = PETSC_FALSE;

// This function is called at finalization time.
static void ShutDownTimers() {
  // Dump timing information before we leave.
  TDyWriteTimingProfile("tdycore_profile.csv");

  TDyDestroyTimers();
}

PetscErrorCode TDyInitTimers() {
  // Register timers table.
  if (TDY_TIMERS == NULL) {
    TDY_TIMERS = kh_init(TDY_TIMER_MAP);
  }

  // Register profiling stages table.
  if (TDY_PROFILING_STAGES == NULL) {
    TDY_PROFILING_STAGES = kh_init(TDY_PROFILING_STAGE_MAP);
  }

  // Register profiling metadata table.
  if (TDY_PROFILING_METADATA == NULL) {
    TDY_PROFILING_METADATA = kh_init(TDY_PROFILING_MD_MAP);
  }

  // Register some logging stages.
  PetscErrorCode ierr;
  ierr = TDyAddProfilingStage("TDycore Setup"); CHKERRQ(ierr);
  ierr = TDyAddProfilingStage("TDycore Stepping"); CHKERRQ(ierr);
  ierr = TDyAddProfilingStage("TDycore I/O"); CHKERRQ(ierr);

  // Register our shutdown function.
  TDyOnFinalize(ShutDownTimers);

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

PetscErrorCode TDySetTimingMetadata(TDy tdy) {
  if (timersEnabled_) {
    // Convert the tdy pointer to a 64-bit integer address so we can use it as
    // a key in the metadata table.
    khint64_t tdy_addr = (khint64_t)tdy;
    khiter_t iter = kh_get(TDY_PROFILING_MD_MAP, TDY_PROFILING_METADATA, tdy_addr);
    TimingMetadata* md;
    PetscNew(&md);
    if (iter == kh_end(TDY_PROFILING_METADATA)) {
      int retval;
      iter = kh_put(TDY_PROFILING_MD_MAP, TDY_PROFILING_METADATA, tdy_addr, &retval);
      kh_val(TDY_PROFILING_METADATA, iter) = md;
    }
    md->mode = tdy->options.mode;
    md->discretization = tdy->options.discretization;
    TDyMPFAO *mpfao = tdy->context;
    TDyMesh *mesh = mpfao->mesh;
    if (tdy->dm != NULL) {
      PetscInt cStart, cEnd;
      DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd);
      md->num_cells = mesh->num_cells_local;
    } else {
      md->num_cells = 0;
    }
    MPI_Comm_size(PETSC_COMM_WORLD, &(md->num_proc));
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

    // Add a footer to the profile CSV that contains metadata.
    // TODO: For now, we only support profiling for one TDy instance. It's not
    // TODO: entirely clear to me how best to accommodate multiple instances
    // TODO: in a way that is easy to understand. So we just retrieve metadata
    // TODO: for the first TDy we encounter.

    // Fetch the metadata for the first TDy instance.
    khiter_t md_iter = kh_begin(TDY_PROFILING_METADATA);
    while (!kh_exist(TDY_PROFILING_METADATA, md_iter)) ++md_iter;
    if (md_iter == kh_end(TDY_PROFILING_METADATA)) {
      // No metadata was recorded. We're finished.
      return 0;
    }

    TimingMetadata* md = kh_val(TDY_PROFILING_METADATA, md_iter);

    // Now write the footer.
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      // Count up the total number of cells in the simulation.
      int num_cells;
      MPI_Reduce(&(md->num_cells), &num_cells, 1, MPI_INT, MPI_SUM,
                 0, PETSC_COMM_WORLD);
      const char* mode_name = TDyModes[md->mode];
      const char* disc_name = TDyDiscretizations[md->discretization];
      FILE* f = fopen("tdycore_profile.csv", "a");
      fprintf(f, "METADATA\n");
      fprintf(f, "Mode,Discretization,NumProc,NumCells\n");
      fprintf(f, "%s,%s,%d,%d", mode_name, disc_name, md->num_proc, num_cells);
      fclose(f);
    } else { // rank > 0
      MPI_Reduce(&md->num_cells, NULL, 1, MPI_INT, MPI_SUM,
                 0, PETSC_COMM_WORLD);
    }
    return 0;
  }
  return 0;
}

void TDyDestroyTimers() {
  // Free the various registries.
  if (TDY_TIMERS != NULL) {
    kh_destroy(TDY_TIMER_MAP, TDY_TIMERS);
  }
  if (TDY_PROFILING_STAGES != NULL) {
    kh_destroy(TDY_PROFILING_STAGE_MAP, TDY_PROFILING_STAGES);
  }
  TimingMetadata* val;
  kh_foreach_value(TDY_PROFILING_METADATA, val, PetscFree(val));
  if (TDY_PROFILING_METADATA != NULL) {
    kh_destroy(TDY_PROFILING_MD_MAP, TDY_PROFILING_METADATA);
  }
}
