#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
#include <tdycore.h>
#include <petsc/private/f90impl.h>

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tdyinitnoarguments_                            TDYINITNOARGUMENTS
#define tdyfinalize_                                   TDYFINALIZE
#define tdycreate_                                     TDYCREATE
#define tdycreatesetdm_                                TDYSETDM
#define tdysetmode_                                    TDYSETMODE
#define tdysetdiscretizationmethod_                    TDYSETDISCRETIZATIONMETHOD
#define tdysetfromoptions_                             TDYSETFROMOPTIONS
#define tdydriverinitializettdy_                       TDYDRIVERINITIALIZETDY
#define tdydtimeintegratorruntotime_                   TDYTIMEINTEGRATORRUNTOTIME
#define tdydtimeintegratorsettimestep_                 TDYTIMEINTEGRATORSETTIMESTEP
#define tdydtimeintegratoroutputregression_            TDYTIMEINTEGRATOROUTPUTREGRESSION
#define tdysetupnumericalmethods_                      TDYSETUPNUMERICALMETHODS
#define tdysetwaterdensitytype_                        TDYSETWATERDENSITYTYPE
#define tdysetmpfaogmatrixmethod_                      TDYSETMPFAOGMATRIXMETHOD
#define tdysetmpfaoboundaryconditiontype_              TDYSETMPFAOGBOUNDARYCONDITIONTYPE
#define tdysetifunction_                               TDYSETIFUNCTION
#define tdysetijacobian_                               TDYSETIJACOBIAN
#define tdysetsnesfunction_                            TDYSETSNESFUNCTION
#define tdysetsnesjacobian_                            TDYSETSNESJACOBIAN
#define tdycreatevectors_                              TDYCREATEVECTORS
#define tdycreatejacobian_                             TDYCREATEJACOBIAN
#define tdysetdtimeforsnessolver_                      TDYSETDTIMEFORSNESSOLVER
#define tdysetinitialcondition_                        TDYSETINITIALCONDITION
#define tdypresolvesnessolver_                         TDYSETPRESOLVESNESSOLVER
#define _                        TDYSETPOSTSOLVESNESSOLVER
#define tdycomputesystem_                              TDYCOMPUTESYSTEM
#define tdycomputeerrornorms_                          TDYCOMPUTEERRORNORMS
#define tdysetporosityfunction_                        TDYSETPOROSITYFUNCTION
#define tdysetpermeabilityfunction_                    TDYSETPERMEABILITYFUNCTION
#define tdysetresidualsaturationfunction_              TDYSETRESIDUALSATURATIONFUNCTION
#define tdysetforcingfunction_                         TDYSETFORCINGFUNCTION
#define tdysetboundarypressurefn_                      TDYSETBOUNDARYPRESSUREFN
#define tdysetboundaryvelocityfn_                      TDYSETBOUNDARYVELOCITYFN
#define tdysetporosityvalueslocal0_                    TDYSETPOROSITYVALUESLOCAL0
#define tdysetporosityvalueslocal11_                   TDYSETPOROSITYVALUESLOCAL11
#define tdysetblockpermeabilityueslocal0_              TDYSETBLOCKPERMEABILITYVALUESLOCAL0
#define tdysetblockpermeabilityvalueslocal11_          TDYSETBLOCKPERMEABILITYVALUESLOCAL11
#define tdysetresidualsaturationvalueslocal0_          TDYSETRESIDUALSATURATIONVALUESLOCAL0
#define tdysetresidualsaturationvalueslocal11_         TDYSETRESIDUALSATURATIONVALUESLOCAL11
#define tdysetresidualsaturationvalueslocal0_          TDYSETRESIDUALSATURATIONVALUESLOCAL0
#define tdysetresidualsaturationvalueslocal11_         TDYSETRESIDUALSATURATIONVALUESLOCAL11
#define tdysetcharacteristiccurvemualemvaluesLocal0_        TDYSETCCHARACTERISTICCURVENMUALEMALUESLOCAL0
#define tdysetcharacteristiccurvemualemvaluesLocal11_       TDYSETCCHARACTERISTICCURVENMUALEMALUESLOCAL11
#define tdysetcharacteristiccurvenvalueslocal0_        TDYSETCHARACTERISTICCURVENVALUESLOCAL0
#define tdysetcharacteristiccurvenvalueslocal11_       TDYSETCHARACTERISTICCURVENVALUESLOCAL11
#define tdysetcharacteristiccurvevangenuchtenvalueslocal0_    TDYSETCHARACTERISTICCURVEVANGENUCHTENVALUESLOCAL0
#define tdysetcharacteristiccurvevangenuchtenvalueslocal11_   TDYSETCHARACTERISTICCURVEVANGENUCHTENVALUESLOCAL11
#define tdysetsourcesinkvalueslocal0_                  TDYSETSOURCESINKVALUESLOCAL0
#define tdysetsourcesinkvalueslocal11_                 TDYSETSOURCESINKVALUESLOCAL11
#define tdygetsatruationvalueslocal_                   TDYGETSATURATIONVALUESLOCAL
#define tdygetliquidmassvalueslocal_                   TDYGETLIQUIDMASSVALUESLOCAL
#define tdygetcharacteristiccurvemvalueslocal_         TDYGETCHARACTERISTICCURVEMVALUESLOCAL
#define tdysetcharacteristiccurvevangenuchtenvalueslocal_     TDYGETCHARACTERISTICCURVEALPHAVALUESLOCAL
#define tdygetporosityalueslocal_                      TDYGETPOROSITYVALUESLOCAL
#define tdygetblockpermeabilityueslocal_               TDYGETBLOCKPERMEABILITYVALUESLOCAL
#define tdygetnumcellslocal_                           TDYGETNUMCELLSLOCAL
#define tdygetcellnaturalidslocal_                     TDYGETCELLNATURALIDSLOCAL
#define tdyupdatestate_                                TDYUPDATESTATE
#define tdyoutputregression_                           TDYOUTPUTREGRESSION
#define tdydestroy_                                    TDYDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdyinitnoarguments_                            tdyinitnoarguments
#define tdyfinalize_                                   tdyfinalize
#define tdycreate_                                     tdycreate
#define tdysetdm_                                      tdysetdm
#define tdysetmode_                                    tdysetmode
#define tdysetdiscretizationmethod_                    tdysetdiscretizationmethod
#define tdysetfromoptions_                             tdysetfromoptions
#define tdydriverinitializettdy_                       tdydriverinitializetdy
#define tdydtimeintegratorruntotime_                   tdydtimeintegratorruntotime
#define tdydtimeintegratorsettimestep_                 tdydtimeintegratorsettimestep
#define tdydtimeintegratoroutputregression_            tdydtimeintegratoroutputregression
#define tdysetupnumericalmethods_                      tdysetupnumericalmethods
#define tdysetwaterdensitytype_                        tdysetwaterdensitytype
#define tdysetmpfaogmatrixmethod_                      tdysetmpfaogmatrixmethod
#define tdysetmpfaoboundaryconditiontype_              tdysetmpfaoboundaryconditiontype
#define tdysetifunction_                               tdysetifunction
#define tdysetijacobian_                               tdysetijacobian
#define tdysetsnesfunction_                            tdysetsnesfunction
#define tdysetsnesjacobian_                            tdysetsnesjacobian
#define tdycreatevectors_                              tdycreatevectors
#define tdycreatejacobian_                             tdycreatejacobian
#define tdysetdtimeforsnessolver_                      tdysetdtimeforsnessolver
#define tdysetinitialcondition_                        tdysetinitialcondition
#define tdypresolvesnessolver_                         tdypresolvesnessolver
#define tdypostsolve_                                  tdypostsolve
#define tdycomputesystem_                              tdycomputesystem
#define tdycomputeerrornorms_                          tdycomputeerrornorms
#define tdysetporosityfunction_                        tdysetporosityfunction
#define tdysetpermeabilityfunction_                    tdysetpermeabilityfunction
#define tdysetresidualsaturationfunction_              tdysetresidualsaturationfunction
#define tdysetforcingfunction_                         tdysetforcingfunction
#define tdysetboundarypressurefn_                      tdysetboundarypressurefn
#define tdysetboundaryvelocityfn_                      tdysetboundaryvelocityfn
#define tdysetporosityvalueslocal0_                    tdysetporosityvalueslocal0
#define tdysetporosityvalueslocal11_                   tdysetporosityvalueslocal11
#define tdysetblockpermeabilityueslocal0_              tdysetblockpermeabilityvalueslocal0
#define tdysetblockpermeabilityvalueslocal11_          tdysetblockpermeabilityvalueslocal11
#define tdysetresidualsaturationvalueslocal0_          tdysetresidualsaturationvalueslocal0
#define tdysetresidualsaturationvalueslocal11_         tdysetresidualsaturationvalueslocal11
#define tdysetcharacteristiccurvemualemvalueslocal0_        tdysetcharacteristiccurvemualemvalueslocal0
#define tdysetcharacteristiccurvemualemvalueslocal11_       tdysetcharacteristiccurvemualemvalueslocal11
#define tdysetcharacteristiccurvenvalueslocal0_        tdysetcharacteristiccurvenvalueslocal0
#define tdysetcharacteristiccurvenvalueslocal11_       tdysetcharacteristiccurvenvalueslocal11
#define tdysetcharacteristiccurvevangenuchtenvalueslocal0_    tdysetcharacteristiccurvevangenuchtenvalueslocal0
#define tdysetcharacteristiccurvevangenuchtenvalueslocal11_   tdysetcharacteristiccurvevangenuchtenvalueslocal11
#define tdysetsourcesinkvalueslocal0_                  tdysetsourcesinkvalueslocal0
#define tdysetsourcesinkvalueslocal11_                 tdysetsourcesinkvalueslocal11
#define tdygetsatruationvalueslocal_                   tdygetsaturationvalueslocal
#define tdygetliquidmassvalueslocal_                   tdygetliquidmassvalueslocal
#define tdygetcharacteristiccurvemvalueslocal_         tdygetcharacteristiccurvemvalueslocal
#define tdysetcharacteristiccurvevangenuchtenvalueslocal_     tdysetcharacteristiccurvevangenuchtenvalueslocal
#define tdygetporosityvalueslocal_                     tdygetporosityvalueslocal
#define tdygetblockpermeabilityueslocal_               tdygetblockpermeabilityvalueslocal
#define tdygetnumcellslocal_                           tdygetnumcellslocal
#define tdygetcellnaturalidslocal_                     tdygetcellnaturalidslocal
#define tdyupdatestate_                                tdyupdatestate
#define tdyoutputregression_                           tdyoutputregression
#define tdydestroy_                                    tdydestroy
#endif

// Legacy Fortran function pointer registry
static struct {
  PetscFortranCallbackId porosity;
  PetscFortranCallbackId permeability;
  PetscFortranCallbackId residualsaturation;
  PetscFortranCallbackId forcing;
  PetscFortranCallbackId boundary_pressure;
  PetscFortranCallbackId boundary_velocity;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId function_pgiptr;
#endif
} _cb;

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdyinitnoarguments_(int *__ierr){
*__ierr = TDyInitNoArguments();
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdyfinalize_(int *__ierr){
*__ierr = TDyFinalize();
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdycreate_(TDy *_tdy, int *__ierr){
*__ierr = TDyCreate(_tdy);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetmode_(TDy _tdy, PetscInt *mode, int *__ierr){
*__ierr = TDySetMode((TDy)PetscToPointer((_tdy)), *mode);
}
#if defined(__cplusplus)
}
#endif
#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetdiscretizationmethod_(TDy _tdy, PetscInt *method, int *__ierr){
*__ierr = TDySetDiscretizationMethod((TDy)PetscToPointer((_tdy)), *method);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetdm_(TDy _tdy, DM dm, int *__ierr){
*__ierr = TDySetDM((TDy)PetscToPointer((_tdy)), (DM)PetscToPointer((dm)));
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
PETSC_EXTERN void  tdydriverinitializetdy_(TDy tdy, int *__ierr){
*__ierr = TDyDriverInitializeTDy((TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdytimeintegratorruntotime_(TDy tdy, PetscReal *time, int *__ierr){
*__ierr = TDyTimeIntegratorRunToTime((TDy)PetscToPointer((tdy)), *time);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdytimeintegratorsettimestep_(TDy tdy, PetscReal *dtime, int *__ierr){
*__ierr = TDyTimeIntegratorSetTimeStep((TDy)PetscToPointer((tdy)), *dtime);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdytimeintegratoroutputregression_(TDy tdy, int *__ierr){
*__ierr = TDyTimeIntegratorOutputRegression((TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetupnumericalmethods_(TDy _tdy, int *__ierr){
*__ierr = TDySetupNumericalMethods((TDy)PetscToPointer((_tdy)));
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
PETSC_EXTERN void  tdysetmpfaogmatrixmthod_(TDy tdy, PetscInt *method, int *__ierr){
*__ierr = TDySetMPFAOGmatrixMethod((TDy)PetscToPointer((tdy)), *method);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdysetmpfaoboundaryconditiontype_(TDy tdy, PetscInt *bctype, int *__ierr){
*__ierr = TDySetMPFAOBoundaryConditionType((TDy)PetscToPointer((tdy)), *bctype);
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
PETSC_EXTERN void  tdysetinitialcondition_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDySetInitialCondition(
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
PETSC_EXTERN void  tdypostsolve_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDyPostSolve(
  (TDy)PetscToPointer(tdy),
  (Vec)PetscToPointer(solution));
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
PETSC_EXTERN void  tdycreatevectors_(TDy tdy, int *__ierr){
*__ierr = TDyCreateVectors(
  (TDy)PetscToPointer((tdy)));
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdycreatejacobian_(TDy tdy, int *__ierr){
*__ierr = TDyCreateJacobian(
  (TDy)PetscToPointer((tdy)));
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

static PetscErrorCode ourtdyboundarypressurefn(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.boundary_pressure,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void tdysetboundarypressurefn_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.boundary_pressure,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetBoundaryPressureFn(*tdy,ourtdyboundarypressurefn,NULL);
}

static PetscErrorCode ourtdyboundaryvelocityfn(TDy tdy,PetscReal *x,PetscReal *f,void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void* ptr;
  PetscObjectGetFortranCallback((PetscObject)tdy,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function_pgiptr,NULL,&ptr);
#endif
  PetscObjectUseFortranCallback(tdy,_cb.boundary_velocity,(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode* PETSC_F90_2PTR_PROTO_NOVAR),(&tdy,x,f,_ctx,&ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void tdysetboundaryvelocityfn_(TDy *tdy, PetscErrorCode (*func)(TDy*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy ,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.boundary_velocity,(PetscVoidFunction)func,ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tdy,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function_pgiptr,NULL,ptr);if (*ierr) return;
#endif
  *ierr = TDySetBoundaryVelocityFn(*tdy,ourtdyboundaryvelocityfn,NULL);
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

PETSC_EXTERN void tdysetcharacteristiccurvemualemvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetCharacteristicCurveMualemValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetcharacteristiccurvemualemvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetcharacteristiccurvemualemvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetcharacteristiccurvemualemvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetcharacteristiccurvemualemvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetcharacteristiccurvenvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  *ierr = TDySetCharacteristicCurveNValuesLocal(*tdy,*ni,ix,y);
}

PETSC_EXTERN void tdysetcharacteristiccurvenvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetcharacteristiccurvenvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetcharacteristiccurvenvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], int *ierr )
{
  tdysetcharacteristiccurvenvalueslocal_(tdy,ni,ix,y,ierr);
}

PETSC_EXTERN void tdysetcharacteristiccurvevangenuchtenvalueslocal_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], PetscScalar z[], int *ierr )
{
  *ierr = TDySetCharacteristicCurveVanGenuchtenValuesLocal(*tdy,*ni,ix,y,z);
}

PETSC_EXTERN void tdysetcharacteristiccurvevangenuchtenvalueslocal0_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], PetscScalar z[], int *ierr )
{
  tdysetcharacteristiccurvevangenuchtenvalueslocal_(tdy,ni,ix,y,z,ierr);
}

PETSC_EXTERN void tdysetcharacteristiccurvevangenuchtenvalueslocal11_(TDy *tdy,PetscInt *ni, PetscInt ix[], PetscScalar y[], PetscScalar z[], int *ierr )
{
  tdysetcharacteristiccurvevangenuchtenvalueslocal_(tdy,ni,ix,y,z,ierr);
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

PETSC_EXTERN void tdygetcharacteristiccurvemvalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetCharacteristicCurveMValuesLocal(*tdy,ni,y);
}

PETSC_EXTERN void tdygetcharacteristiccurvealphavalueslocal_(TDy *tdy,PetscInt *ni, PetscScalar y[], int *ierr )
{
  *ierr = TDyGetCharacteristicCurveAlphaValuesLocal(*tdy,ni,y);
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

PETSC_EXTERN void tdyupdatestate_(TDy *tdy,PetscScalar y[], int ncells, int *ierr )
{
  *ierr = TDyUpdateState(*tdy,y,ncells);
}

