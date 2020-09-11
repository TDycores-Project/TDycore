#include "tdytimers.h"

PetscErrorCode TDyInitTimers()
{
  TDY_TIMERS = kh_init(TDY_TIMER_MAP);
  return 0;
}

PetscErrorCode TDyDestroyTimers()
{
  if (TDY_TIMERS != NULL)
    kh_destroy(TDY_TIMER_MAP, TDY_TIMERS);
  return 0;
}

