#include "tdytimers.h"

khash_t(TDY_TIMER_MAP)* TDY_TIMERS = NULL;

PetscErrorCode TDyInitTimers()
{
  if (TDY_TIMERS == NULL)
    TDY_TIMERS = kh_init(TDY_TIMER_MAP);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyDestroyTimers()
{
  if (TDY_TIMERS != NULL)
    kh_destroy(TDY_TIMER_MAP, TDY_TIMERS);
  PetscFunctionReturn(0);
}

