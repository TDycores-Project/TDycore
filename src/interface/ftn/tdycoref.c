#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
#include <tdycore.h>
#include <petsc/private/f90impl.h>

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))

#include "tdycore.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tdycreate_                                  TDYCREATE
#define tdysetdiscretizationmethod_                 TDYSETDISCRETIZATIONMETHOD
#define tdysetup_                                   TDYSETUP
#define tdysetwaterdensitytype_                     TDYWATERDENSITYTYPE
#define tdysetifunction_                            TDYSETIFUNCTION
#define tdysetijacobian_                            TDYSETIJACOBIAN
#define tdysetsnesfunction_                         TDYSETSNESFUNCTION
#define tdysetsnesjacobian_                         TDYSETSNESJACOBIAN
#define tdysetdtimeforsnessolver_                   TDYSETDTIMEFORSNESSOLVER
#define tdysetinitialsolutionforsnessolver_         TDYSETINITIALSOLUTIONFORSNESSOLVER
#define tdypresolvesnessolver_                      TDYSETPRESOLVESNESSOLVER
#define tdypostsolvesnessolver_                     TDYSETPOSTSOLVESNESSOLVER
#define tdysetfromoption_                           TDYSETFROMOPTIONS
#define tdycomputesystem_                           TDYCOMPUTESYSTEM
#define tdycomputeerrornorms_                       TDYCOMPUTEERRORNORMS
#define tdysetporosityfunction_                     TDYSETPOROSITYFUNCTION
#define tdysetpermeabilityfunction_                 TDYSETPERMEABILITYFUNCTION
#define tdysetresidualsaturationfunction_           TDYSETRESIDUALSATURATIONFUNCTION
#define tdysetforcingfunction_                      TDYSETFORCINGFUNCTION
#define tdysetdirichletvaluefunction_               TDYSETDIRICHLETVALUEFUNCTION
#define tdysetdirichletfluxfunction_                TDYSETDIRICHLETFLUXFUNCTION
#define tdysetporosityvalueslocal0_                 TDYSETPOROSITYVALUESLOCAL0
#define tdysetporosityvalueslocal11_                TDYSETPOROSITYVALUESLOCAL11
#define tdysetblockpermeabilityueslocal0_           TDYSETBLOCKPERMEABILITYVALUESLOCAL0
#define tdysetblockpermeabilityvalueslocal11_       TDYSETBLOCKPERMEABILITYVALUESLOCAL11
#define tdysetresidualsaturationvalueslocal0_       TDYSETRESIDUALSATURATIONVALUESLOCAL0
#define tdysetresidualsaturationvalueslocal11_      TDYSETRESIDUALSATURATIONVALUESLOCAL11
#define tdysetresidualsaturationvalueslocal0_       TDYSETRESIDUALSATURATIONVALUESLOCAL0
#define tdysetresidualsaturationvalueslocal11_      TDYSETRESIDUALSATURATIONVALUESLOCAL11
#define tdysetmaterialpropertymvalueslocal0_        TDYSETMATERIALPROPERTYMVALUESLOCAL0
#define tdysetmaterialpropertymvalueslocal11_       TDYSETMATERIALPROPERTYMVALUESLOCAL11
#define tdysetmaterialpropertynvalueslocal0_        TDYSETMATERIALPROPERTYNVALUESLOCAL0
#define tdysetmaterialpropertynvalueslocal11_       TDYSETMATERIALPROPERTYNVALUESLOCAL11
#define tdysetmaterialpropertyalphavalueslocal0_    TDYSETMATERIALPROPERTYALPHAVALUESLOCAL0
#define tdysetmaterialpropertyalphavalueslocal11_   TDYSETMATERIALPROPERTYALPHAVALUESLOCAL11
#define tdysetsourcesinkvalueslocal0_               TDYSETSOURCESINKVALUESLOCAL0
#define tdysetsourcesinkvalueslocal11_              TDYSETSOURCESINKVALUESLOCAL11
#define tdygetsatruationvalueslocal_                TDYGETSATURATIONVALUESLOCAL
#define tdygetliquidmassvalueslocal_                TDYGETLIQUIDMASSVALUESLOCAL
#define tdygetmaterialpropertymvalueslocal_         TDYGETMATERIALPROPERTYMVALUESLOCAL
#define tdygetmaterialpropertyalphavalueslocal_     TDYGETMATERIALPROPERTYALPHAVALUESLOCAL
#define tdygetporosityalueslocal_                   TDYGETPOROSITYVALUESLOCAL
#define tdygetblockpermeabilityueslocal_            TDYGETBLOCKPERMEABILITYVALUESLOCAL
#define tdygetnumcellslocal_                        TDYGETNUMCELLSLOCAL
#define tdygetcellnaturalidslocal_                  TDYGETCELLNATURALIDSLOCAL
#define tdyupdatestate_                             TDYUPDATESTATE
#define tdyoutputregression_                        TDYOUTPUTREGRESSION
#define tdydestroy_                                 TDYDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdycreate_                                  tdycreate
#define tdysetdiscretizationmethod_                 tdysetdiscretizationmethod
#define tdysetup_                                   tdysetup
#define tdysetwaterdensitytype_                     tdysetwaterdensitytype
#define tdysetifunction_                            tdysetifunction
#define tdysetijacobian_                            tdysetijacobian
#define tdysetsnesfunction_                         tdysetsnesfunction
#define tdysetsnesjacobian_                         tdysetsnesjacobian
#define tdysetdtimeforsnessolver_                   tdysetdtimeforsnessolver
#define tdysetinitialsolutionforsnessolver_         tdysetinitialsolutionforsnessolver
#define tdypresolvesnessolver_                      tdypresolvesnessolver
#define tdypostsolvesnessolver_                     tdypostsolvesnessolver
#define tdysetfromoptions_                          tdysetfromoptions
#define tdycomputesystem_                           tdycomputesystem
#define tdycomputeerrornorms_                       tdycomputeerrornorms
#define tdysetporosityfunction_                     tdysetporosityfunction
#define tdysetpermeabilityfunction_                 tdysetpermeabilityfunction
#define tdysetresidualsaturationfunction_           tdysetresidualsaturationfunction
#define tdysetforcingfunction_                      tdysetforcingfunction
#define tdysetdirichletvaluefunction_               tdysetdirichletvaluefunction
#define tdysetdirichletfluxfunction_                tdysetdirichletfluxfunction
#define tdysetporosityvalueslocal0_                 tdysetporosityvalueslocal0
#define tdysetporosityvalueslocal11_                tdysetporosityvalueslocal11
#define tdysetblockpermeabilityueslocal0_           tdysetblockpermeabilityvalueslocal0
#define tdysetblockpermeabilityvalueslocal11_       tdysetblockpermeabilityvalueslocal11
#define tdysetresidualsaturationvalueslocal0_       tdysetresidualsaturationvalueslocal0
#define tdysetresidualsaturationvalueslocal11_      tdysetresidualsaturationvalueslocal11
#define tdysetmaterialpropertymvalueslocal0_        tdysetmaterialpropertymvalueslocal0
#define tdysetmaterialpropertymvalueslocal11_       tdysetmaterialpropertymvalueslocal11
#define tdysetmaterialpropertynvalueslocal0_        tdysetmaterialpropertynvalueslocal0
#define tdysetmaterialpropertynvalueslocal11_       tdysetmaterialpropertynvalueslocal11
#define tdysetmaterialpropertyalphavalueslocal0_    tdysetmaterialpropertyalphavalueslocal0
#define tdysetmaterialpropertyalphavalueslocal11_   tdysetmaterialpropertyalphavalueslocal11
#define tdysetsourcesinkvalueslocal0_               tdysetsourcesinkvalueslocal0
#define tdysetsourcesinkvalueslocal11_              tdysetsourcesinkvalueslocal11
#define tdygetsatruationvalueslocal_                tdygetsaturationvalueslocal
#define tdygetliquidmassvalueslocal_                tdygetliquidmassvalueslocal
#define tdygetmaterialpropertymvalueslocal_         tdygetmaterialpropertymvalueslocal
#define tdygetmaterialpropertyalphavalueslocal_     tdygetmaterialpropertyalphavalueslocal
#define tdygetporosityvalueslocal_                  tdygetporosityvalueslocal
#define tdygetblockpermeabilityueslocal_            tdygetblockpermeabilityvalueslocal
#define tdygetnumcellslocal_                        tdygetnumcellslocal
#define tdygetcellnaturalidslocal_                  tdygetcellnaturalidslocal
#define tdyupdatestate_                             tdyupdatestate
#define tdyoutputregression_                        tdyoutputregression
#define tdydestroy_                                 tdydestroy
#endif

static struct {
  PetscFortranCallbackId porosity;
  PetscFortranCallbackId permeability;
  PetscFortranCallbackId residualsaturation;
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
PETSC_EXTERN void  tdycreate_(DM dm,TDy *_tdy, int *__ierr){
*__ierr = TDyCreate((DM)PetscToPointer((dm)), _tdy);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetdiscretizationmethod_(TDy tdy, PetscInt *method, int *__ierr){
*__ierr = TDySetDiscretizationMethod((TDy)PetscToPointer((tdy)), *method);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetup_(TDy tdy, int *__ierr){
*__ierr = TDySetup((TDy)PetscToPointer((tdy)) );
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetwaterdensitytype_(TDy tdy, PetscInt *method, int *__ierr){
*__ierr = TDySetWaterDensityType((TDy)PetscToPointer((tdy)), *method);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetifunction_(TS ts, TDy tdy, int *__ierr){
*__ierr = TDySetIFunction(
  (TS)PetscToPointer((ts)),
  (TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetijacobian_(TS ts, TDy tdy, int *__ierr){
*__ierr = TDySetIJacobian(
  (TS)PetscToPointer((ts)),
  (TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetsnesfunction_(SNES snes, TDy tdy, int *__ierr){
*__ierr = TDySetSNESFunction(
  (SNES)PetscToPointer((snes)),
  (TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetsnesjacobian_(SNES snes, TDy tdy, int *__ierr){
*__ierr = TDySetSNESJacobian(
  (SNES)PetscToPointer((snes)),
  (TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetdtimeforsnessolver_(TDy tdy, PetscReal *dtime, int *__ierr){
*__ierr = TDySetDtimeForSNESSolver((TDy)PetscToPointer(tdy), *dtime);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetinitialsolutionforsnessolver_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDySetInitialSolutionForSNESSolver(
  (TDy)PetscToPointer(tdy),
  (Vec)PetscToPointer(solution));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdypresolvesnessolver_(TDy tdy, int *__ierr){
*__ierr = TDyPreSolveSNESSolver((TDy)PetscToPointer(tdy));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdypostsolvesnessolver_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDyPostSolveSNESSolver(
  (TDy)PetscToPointer(tdy),
  (Vec)PetscToPointer(solution));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetfromoptions_(TDy tdy, int *__ierr){
*__ierr = TDySetFromOptions((TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdycomputesystem_(TDy tdy, Mat K, Vec F, int *__ierr){
*__ierr = TDyComputeSystem(
  (TDy)PetscToPointer((tdy) ),
  (Mat)PetscToPointer((K) ),
  (Vec)PetscToPointer((F) ));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdycomputeerrornorms_(TDy tdy, Vec U, PetscReal *normp, PetscReal *normv,  int *__ierr){
*__ierr = TDyComputeErrorNorms(
  (TDy)PetscToPointer((tdy) ),
  (Vec)PetscToPointer((U) ),
  normp, normv);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdyoutputregression_(TDy tdy, Vec U, int *__ierr){
*__ierr = TDyOutputRegression(
  (TDy)PetscToPointer((tdy) ),
  (Vec)PetscToPointer((U) ));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdydestroy_(TDy *_tdy, int *__ierr){
*__ierr = TDyDestroy(_tdy);
}
#if defined(__cplusplus)
}
#endif

static PetscErrorCode ourtdyporosityfunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.porosity,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

static PetscErrorCode ourtdypermeabilityfunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.permeability,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void tdysetporosityfunction_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.porosity,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetPorosityFunction(*tdy,ourtdyporosityfunction,NULL);
}

PETSC_EXTERN void tdysetpermeabilityfunction_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.permeability,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetPermeabilityFunction(*tdy,ourtdypermeabilityfunction,NULL);
}

static PetscErrorCode ourtdysetresidualfunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.residualsaturation,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}


PETSC_EXTERN void tdysetresidualsaturationfunction_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.residualsaturation,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetResidualSaturationFunction(*tdy,ourtdysetresidualfunction,NULL);
}

static PetscErrorCode ourtdyforcingfunction2(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.forcing,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void tdysetforcingfunction_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.forcing,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetForcingFunction(*tdy,ourtdyforcingfunction2,NULL);
}

static PetscErrorCode ourtdydirichletvaluefunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.dirichletvalue,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void tdysetdirichletvaluefunction_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.dirichletvalue,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetDirichletValueFunction(*tdy,ourtdydirichletvaluefunction,NULL);
}

static PetscErrorCode ourtdydirichletfluxfunction(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.dirichletflux,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void tdysetdirichletfluxfunction_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.dirichletflux,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetDirichletFluxFunction(*tdy,ourtdydirichletfluxfunction,NULL);
}

PETSC_EXTERN void tdysetporosityvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetPorosityValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetporosityvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetporosityvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetporosityvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetporosityvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetblockpermeabilityvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetBlockPermeabilityValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetblockpermeabilityvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetblockpermeabilityvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetblockpermeabilityvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetblockpermeabilityvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetresidualsaturationvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetResidualSaturationValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetresidualsaturationvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetresidualsaturationvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetresidualsaturationvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetresidualsaturationvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetmaterialpropertymvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetMaterialPropertyMValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetmaterialpropertymvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetmaterialpropertymvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetmaterialpropertymvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetmaterialpropertymvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetmaterialpropertynvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetMaterialPropertyNValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetmaterialpropertynvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetmaterialpropertynvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetmaterialpropertynvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetmaterialpropertynvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetmaterialpropertyalphavalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetMaterialPropertyAlphaValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetmaterialpropertyalphavalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetmaterialpropertyalphavalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetmaterialpropertyalphavalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetmaterialpropertyalphavalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetsourcesinkvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetSourceSinkValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetsourcesinkvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetsourcesinkvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetsourcesinkvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetsourcesinkvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdygetsaturationvalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetSaturationValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetliquidmassvalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetLiquidMassValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetmaterialpropertymvalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetMaterialPropertyMValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetmaterialpropertyalphavalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetMaterialPropertyAlphaValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetporosityvalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetPorosityValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetblockpermeabilityvalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetBlockPermeabilityValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetnumcellslocal_(TDy *tdy,PetscInt *ni, int *ierr )
{
  *ierr = TDyGetNumCellsLocal(*tdy,ni);
}

PETSC_EXTERN void tdygetcellnaturalidslocal_(TDy *tdy,PetscInt *ni, PetscInt y[], int *ierr )
{
  *ierr = TDyGetCellNaturalIDsLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetcellislocal_(TDy *tdy,PetscInt *ni, PetscInt y[], int *ierr )
{
  *ierr = TDyGetCellIsLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdyupdatestate_(TDy *tdy,PetscScalar y[], int *ierr )
{
  *ierr = TDyUpdateState(*tdy,y);
}
