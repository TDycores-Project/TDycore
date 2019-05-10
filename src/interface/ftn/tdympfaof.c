#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
#include <tdycore.h>
#include <petsc/private/f90impl.h>

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tdympfaorecovervelocity_         TDYMPFAORECOVERVELOCITY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdympfaorecovervelocity_         tdympfaorecovervelocity
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  tdympfaorecovervelocity_(TDy tdy, Vec U, int *__ierr){
*__ierr = TDyMPFAORecoverVelocity(
  (TDy)PetscToPointer((tdy) ),
  (Vec)PetscToPointer((U) ));
}
#if defined(__cplusplus)
}
#endif
