#ifndef TDYTIMERS_H
#define TDYTIMERS_H

#include <petsc.h>

// Timer (PetscLogEvent) macros for profiling.
// These macros make it easier to create/start/stop timers for profiling parts
// of TDycore. The PetscEventLog machinery is a bit cumbersome when applied to
// code bases larger than small examples. Here we attempt to automatically
// manage timers in a global registry, retrievable by name.

// To accomplish this, we use a hash map with string keys and PetscLogEvent
// values. PETSc ships with khash, so let's appropriate it.
#include <petsc/private/kernels/khash.h>
KHASH_MAP_INIT_STR(TDY_TIMER_MAP, PetscLogEvent)
static khash_t(TDY_TIMER_MAP)* TDY_TIMERS = NULL;

// TDY_START_TIMER(name): declares and starts a timer.
#define TDY_START_TIMER(name) \
  khiter_t name##_iter = kh_get(TDY_TIMER_MAP, TDY_TIMERS, #name); \
  PetscLogEvent name##_timer; \
  if (name##_iter == kh_end(TDY_TIMER_MAP)) { \
    PetscLogEventRegister(#name, TDY_CLASSID, &name##_timer); \
    int retval; \
    name##_iter = kh_put(TDY_TIMER_MAP, TDY_TIMERS, &retval); \
    kh_val(TDY_TIMERS, name##_iter) = name##_timer; \
  } else { \
    name##_timer = kh_val(TDY_TIMERS, name##_iter); \
  } \
  PetscLogEventBegin(name##_timer, 0, 0, 0, 0);

// TDY_STOP_TIMER(name): stops a timer started by TDY_STOP_TIMER(name) in the
// same function.
#define TDY_STOP_TIMER(name) \
  PetscLogEventEnd(name##_timer, 0, 0, 0, 0);

// TDY_START_FUNCTION_TIMER: call this at the beginning of a function to start
// a timer named after the function itself.
#define TDY_START_FUNCTION_TIMER() \
  TDY_START_TIMER(__func__)

// TDY_STOP_FUNCTION_TIMER: call this at each exit point in a function to stop
// a timer started by TDY_START_FUNCTION_TIMER.
#define TDY_STOP_FUNCTION_TIMER() \
  TDY_STOP_TIMER(__func__)

// Initialize timer machinery.
PETSC_EXTERN PetscErrorCode TDyInitTimers(void);

// Destroy timer machinery, freeing resources.
PETSC_EXTERN PetscErrorCode TDyDestroyTimers(void);

#endif
