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
#define tdysetpermeabilityfunction_   TDYSETPERMEABILITYFUNCTION
#define tdysetforcingfunction2_      TDYSETFORCINGFUNCTION2
#define tdysetdirichletvaluefunction_    TDYSETDIRICHLETVALUEFUNCTION
//#define tdysetdirichletflux_        TDYSETDIRICHLETFLUX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdycreate_                  tdycreate
#define tdysetdiscretizationmethod_ tdysetdiscretizationmethod
#define tdysetfromoptions_          tdysetfromoptions
#define tdycomputesystem_           tdycomputesystem
#define tdysetpermeabilityfunction_   tdysetpermeabilityfunction
#define tdysetforcingfunction2_      tdysetforcingfunction2
#define tdysetdirichletvaluefunction_    tdysetdirichletvaluefunction
//#define tdysetdirichletflux_        tdysetdirichletflux
#endif

static struct {
  PetscFortranCallbackId permeability;
  PetscFortranCallbackId forcing;
  PetscFortranCallbackId dirichletvalue;
  PetscFortranCallbackId dirichletflux;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId function_pgiptr;
#endif
} _cb;


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

static PetscErrorCode ourtdypermeabilityfunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.permeability,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void PETSC_STDCALL tdysetpermeabilityfunction_(TDy *tdy, void (PETSC_STDCALL *func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.permeability,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetPermeabilityFunction(*tdy,ourtdypermeabilityfunction,NULL);
}

static PetscErrorCode ourtdyforcingfunction2(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.forcing,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void PETSC_STDCALL tdysetforcingfunction2_(TDy *tdy, void (PETSC_STDCALL *func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.forcing,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetForcingFunction2(*tdy,ourtdyforcingfunction2,NULL);
}

static PetscErrorCode ourtdydirichletvaluefunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.dirichletvalue,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void PETSC_STDCALL tdysetdirichletvaluefunction_(TDy *tdy, void (PETSC_STDCALL *func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.dirichletvalue,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetDirichletValueFunction(*tdy,ourtdydirichletvaluefunction,NULL);
}
