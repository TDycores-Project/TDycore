#ifndef TDYTIMERS_H
#define TDYTIMERS_H

#include <petsc.h>

PETSC_EXTERN PetscClassId TDY_CLASSID;

// Timer (PetscLogEvent) macros for profiling.
// These macros make it easier to create/start/stop timers for profiling parts
// of TDycore. The PetscEventLog machinery is a bit cumbersome when applied to
// code bases larger than small examples. Here we attempt to automatically
// manage timers in a global registry, retrievable by name.

// To accomplish this, we use a hash map with string keys and PetscLogEvent
// values. PETSc ships with khash, so let's appropriate it.
#include <petsc/private/kernels/khash.h>
KHASH_MAP_INIT_STR(TDY_TIMER_MAP, PetscLogEvent)
extern khash_t(TDY_TIMER_MAP)* TDY_TIMERS;

// t = TDY_GET_TIMER(name): creates or returns a timer (PetscLogEvent).
PETSC_STATIC_INLINE PetscLogEvent TDY_GET_TIMER(const char* name) {
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

// TDY_START_TIMER(timer): declares and starts the given timer.
#define TDY_START_TIMER(timer) \
  PetscLogEventBegin(timer, 0, 0, 0, 0);

// TDY_STOP_TIMER(name): stops a timer started by TDY_STOP_TIMER(name) in the
// same function.
#define TDY_STOP_TIMER(timer) \
  PetscLogEventEnd(timer, 0, 0, 0, 0);

// TDY_START_FUNCTION_TIMER: call this at the beginning of a function to start
// a timer named after the function itself.
#define TDY_START_FUNCTION_TIMER() \
  PetscLogEvent TDY_FUNC_TIMER = TDY_GET_TIMER(__func__); \
  TDY_START_TIMER(TDY_FUNC_TIMER)

// TDY_STOP_FUNCTION_TIMER: call this at each exit point in a function to stop
// a timer started by TDY_START_FUNCTION_TIMER.
#define TDY_STOP_FUNCTION_TIMER() \
  TDY_STOP_TIMER(TDY_FUNC_TIMER)

// Initialize timer machinery.
PETSC_EXTERN PetscErrorCode TDyInitTimers(void);

// Destroy timer machinery, freeing resources.
PETSC_EXTERN PetscErrorCode TDyDestroyTimers(void);

#endif
