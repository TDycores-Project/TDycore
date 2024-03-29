#ifndef TDYTIMERS_H
#define TDYTIMERS_H

#include <petsc.h>

PETSC_EXTERN PetscClassId TDY_CLASSID;

// Initializes the timers subsystem.
PETSC_EXTERN PetscErrorCode TDyInitTimers(void);

// Enables timers.
PETSC_EXTERN PetscErrorCode TDyEnableTimers(void);

// Timer (PetscLogEvent) macros for profiling.
// These macros make it easier to create/start/stop timers for profiling parts
// of TDycore. The PetscEventLog machinery is a bit cumbersome when applied to
// code bases larger than small examples. Here we attempt to automatically
// manage timers in a global registry, retrievable by name.

// To accomplish this, we use a hash map with string keys and PetscLogEvent
// values. PETSc ships with khash, so let's appropriate it.
#include <petsc/private/khash/khash.h>
KHASH_MAP_INIT_STR(TDY_TIMER_MAP, PetscLogEvent)
PETSC_EXTERN khash_t(TDY_TIMER_MAP)* TDY_TIMERS;

// t = TDyGetTimer(name): creates or returns a timer (PetscLogEvent).
static inline PetscLogEvent TDyGetTimer(const char* name) {
  khiter_t iter = kh_get(TDY_TIMER_MAP, TDY_TIMERS, name);
  PetscLogEvent timer;
  if (iter == kh_end(TDY_TIMERS)) {
    PetscLogEventRegister(name, TDY_CLASSID, &timer);
    int retval;
    iter = kh_put(TDY_TIMER_MAP, TDY_TIMERS, name, &retval);
    kh_val(TDY_TIMERS, iter) = timer;
  } else {
    timer = kh_val(TDY_TIMERS, iter);
  }
  return timer;
}

// TDyStartTimer(timer): declares and starts the given timer.
#define TDyStartTimer(timer) \
  PetscLogEventBegin(timer, 0, 0, 0, 0);

// TDyStopTimer(name): stops a timer started by TDyStartTimer(name) in the
// same function.
#define TDyStopTimer(timer) \
  PetscLogEventEnd(timer, 0, 0, 0, 0);

// TDY_START_FUNCTION_TIMER: call this at the beginning of a function to start
// a timer named after the function itself.
#define TDY_START_FUNCTION_TIMER() \
  PetscLogEvent TDY_FUNC_TIMER = TDyGetTimer(__func__); \
  TDyStartTimer(TDY_FUNC_TIMER)

// TDY_STOP_FUNCTION_TIMER: call this at each exit point in a function to stop
// a timer started by TDY_START_FUNCTION_TIMER.
#define TDY_STOP_FUNCTION_TIMER() \
  TDyStopTimer(TDY_FUNC_TIMER)

// Here we define a registry for profiling stages.
KHASH_MAP_INIT_STR(TDY_PROFILING_STAGE_MAP, PetscLogStage)
PETSC_EXTERN khash_t(TDY_PROFILING_STAGE_MAP)* TDY_PROFILING_STAGES;

// TDyAddProfilingStage(name): creates a profiling stage (PetscLogStage).
PETSC_EXTERN PetscErrorCode TDyAddProfilingStage(const char* name);

// TDyEnterProfilingstage(name): enters the profiling stage with the given name.
// Has no effect if the given stage name is invalid.
#define TDyEnterProfilingStage(name) \
  { \
    khiter_t iter = kh_get(TDY_PROFILING_STAGE_MAP, TDY_PROFILING_STAGES, name); \
    PetscLogStage stage; \
    if (iter != kh_end(TDY_PROFILING_STAGES)) { \
      stage = kh_val(TDY_PROFILING_STAGES, iter); \
      PetscLogStagePush(stage); \
    } \
  }

// TDyExitProfilingstage(name): exits the profiling stage with the given name.
// (The name isn't actually used, but it can be useful to explicate it.)
#define TDyExitProfilingStage(name) \
  { \
    khiter_t iter = kh_get(TDY_PROFILING_STAGE_MAP, TDY_PROFILING_STAGES, name); \
    if (iter != kh_end(TDY_PROFILING_STAGES)) \
      PetscLogStagePop(); \
  }

typedef struct _p_TDy *TDy;

// Stores metadata for the given dycore in the timers subsystem for
// profiling.
PETSC_EXTERN PetscErrorCode TDySetTimingMetadata(TDy tdy);

// Writes a timing profile report to the given filename.
PETSC_EXTERN PetscErrorCode TDyWriteTimingProfile(const char* filename);

// Breaks down the timers subsystem at finalization.
void TDyDestroyTimers(void);

#endif
