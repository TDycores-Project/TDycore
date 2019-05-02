#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
#include <tdycore.h>
#include <petsc/private/f90impl.h>

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))

#include "tdycore.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tdycreate_                  TDYCREATE
#define tdysetdiscretizationmethod_ TDYSETDISCRETIZATIONMETHOD
#define tdysetfromoption_           TDYSETFROMOPTIONS
#define tdycomputesystem_           TDYCOMPUTESYSTEM
//#define tdysetpermeabilitytensor_   TDYSETPERMEABILITYTENSOR
//#define tdysetforcingfunction_      TDYSETFORCINGFUNCTION
//#define tdysetdirichletfunction_    TDYSETDIRICHLETFUNCTION
//#define tdysetdirichletflux_        TDYSETDIRICHLETFLUX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdycreate_                  tdycreate
#define tdysetdiscretizationmethod_ tdysetdiscretizationmethod
#define tdysetfromoptions_          tdysetfromoptions
#define tdycomputesystem_           tdycomputesystem
//#define tdysetpermeabilitytensor_   tdysetpermeabilitytensor
//#define tdysetforcingfunction_      tdysetforcingfunction
//#define tdysetdirichletfunction_    tdysetdirichletfunction
//#define tdysetdirichletflux_        tdysetdirichletflux
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  tdycreate_(DM dm,TDy *_tdy, int *__ierr){
*__ierr = TDyCreate((DM)PetscToPointer((dm)), _tdy);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  tdysetdiscretizationmethod_(TDy tdy, PetscInt *method, int *__ierr){
*__ierr = TDySetDiscretizationMethod((TDy)PetscToPointer((tdy)), *method);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  tdysetfromoptions_(TDy tdy, int *__ierr){
*__ierr = TDySetFromOptions((TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  tdycomputesystem_(TDy tdy, Mat K, Vec F, int *__ierr){
*__ierr = TDyComputeSystem(
  (TDy)PetscToPointer((tdy) ),
  (Mat)PetscToPointer((K) ),
  (Vec)PetscToPointer((F) ));
}
#if defined(__cplusplus)
}
#endif
