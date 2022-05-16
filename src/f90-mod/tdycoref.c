#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
#include <tdycore.h>
#include <private/tdycoreimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <petsc/private/f90impl.h>

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tdyinitnoarguments_                            TDYINITNOARGUMENTS
#define tdyfinalize_                                   TDYFINALIZE
#define tdycreate_                                     TDYCREATE
#define tdysetmode_                                    TDYSETMODE
#define tdysetdiscretization_                          TDYSETDISCRETIZATION
#define tdysetfromoptions_                             TDYSETFROMOPTIONS
#define tdydriverinitializettdy_                       TDYDRIVERINITIALIZETDY
#define tdydtimeintegratorruntotime_                   TDYTIMEINTEGRATORRUNTOTIME
#define tdydtimeintegratorsettimestep_                 TDYTIMEINTEGRATORSETTIMESTEP
#define tdydtimeintegratoroutputregression_            TDYTIMEINTEGRATOROUTPUTREGRESSION
#define tdysetup_                                      TDYSETUP
#define tdygetdm_                                      TDYGETDM
#define tdyupdatediagnostics_                          TDYUPDATEDIAGNOSTICS
#define tdycreatediagnosticvector_                     TDYCREATEDIAGNOSTICVECTOR
#define tdycreateprognosticvector_                     TDYCREATEPROGGNOSTICVECTOR
#define tdycreatepmatrix_                              TDYCREATEMATRIX
#define tdygetliquidsaturation_                        TDYGETLIQUIDSATURATION
#define tdygetliquidmass_                              TDYGETLIQUIDMASS
#define tdygetliquidpressure_                          TDYGETLIQUIDPRESSURE
#define tdysetwaterdensitytype_                        TDYSETWATERDENSITYTYPE
#define tdympfaosetgmatrixmethod_                      TDYMPFAOSETGMATRIXMETHOD
#define tdympfaosetboundaryconditiontype_              TDYMPFAOSETBOUNDARYCONDITIONTYPE
#define tdysetifunction_                               TDYSETIFUNCTION
#define tdysetijacobian_                               TDYSETIJACOBIAN
#define tdysetsnesfunction_                            TDYSETSNESFUNCTION
#define tdysetsnesjacobian_                            TDYSETSNESJACOBIAN
#define tdycreatevectors_                              TDYCREATEVECTORS
#define tdycreatejacobian_                             TDYCREATEJACOBIAN
#define tdysetdtimeforsnessolver_                      TDYSETDTIMEFORSNESSOLVER
#define tdysetinitialcondition_                        TDYSETINITIALCONDITION
#define tdygetinitialcondition_                        TDYGETINITIALCONDITION
#define tdypresolvesnessolver_                         TDYSETPRESOLVESNESSOLVER
#define tdypostsolve_                                  TDYSETPOSTSOLVE
#define tdycomputeerrornorms_                          TDYCOMPUTEERRORNORMS
#define tdyupdatestate_                                TDYUPDATESTATE
#define tdyoutputregression_                           TDYOUTPUTREGRESSION
#define tdydestroy_                                    TDYDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdyinitnoarguments_                            tdyinitnoarguments
#define tdyfinalize_                                   tdyfinalize
#define tdycreate_                                     tdycreate
#define tdysetmode_                                    tdysetmode
#define tdysetdiscretization_                          tdysetdiscretization
#define tdysetfromoptions_                             tdysetfromoptions
#define tdydriverinitializettdy_                       tdydriverinitializetdy
#define tdydtimeintegratorruntotime_                   tdydtimeintegratorruntotime
#define tdydtimeintegratorsettimestep_                 tdydtimeintegratorsettimestep
#define tdydtimeintegratoroutputregression_            tdydtimeintegratoroutputregression
#define tdysetup_                                      tdysetup
#define tdygetdm_                                      tdygetdm
#define tdyupdatediagnostics_                          tdyupdatediagnostics
#define tdycreatediagnosticvector_                     tdycreatediagnosticvector
#define tdycreateprognosticvector_                     tdycreatedprognosticvector
#define tdycreatematrix_                               tdycreatematrix
#define tdygetliquidsaturation_                        tdygetliquidsaturation
#define tdygetliquidmass_                              tdygetliquidmass
#define tdygetliquidpressure_                          tdygetliquidpressure
#define tdysetwaterdensitytype_                        tdysetwaterdensitytype
#define tdympfaosetgmatrixmethod_                      tdympfaosetgmatrixmethod
#define tdympfaosetboundaryconditiontype_              tdympfaosetboundaryconditiontype
#define tdysetifunction_                               tdysetifunction
#define tdysetijacobian_                               tdysetijacobian
#define tdysetsnesfunction_                            tdysetsnesfunction
#define tdysetsnesjacobian_                            tdysetsnesjacobian
#define tdycreatevectors_                              tdycreatevectors
#define tdycreatejacobian_                             tdycreatejacobian
#define tdysetdtimeforsnessolver_                      tdysetdtimeforsnessolver
#define tdysetinitialcondition_                        tdysetinitialcondition
#define tdygetinitialcondition_                        tdygetinitialcondition
#define tdypresolvesnessolver_                         tdypresolvesnessolver
#define tdypostsolve_                                  tdypostsolve
#define tdycomputeerrornorms_                          tdycomputeerrornorms
#define tdyupdatestate_                                tdyupdatestate
#define tdyoutputregression_                           tdyoutputregression
#define tdydestroy_                                    tdydestroy
#endif

PETSC_EXTERN void  tdyinitnoarguments_(int *__ierr){
*__ierr = TDyInitNoArguments();
}

PETSC_EXTERN void  tdyfinalize_(int *__ierr){
*__ierr = TDyFinalize();
}

PETSC_EXTERN void  tdycreate_(TDy *_tdy, int *__ierr){
*__ierr = TDyCreate(PETSC_COMM_WORLD,_tdy);
}

PETSC_EXTERN void  tdysetmode_(TDy _tdy, PetscInt *mode, int *__ierr){
*__ierr = TDySetMode((TDy)PetscToPointer((_tdy)), *mode);
}

PETSC_EXTERN void  tdysetdiscretization_(TDy _tdy, PetscInt *discretization, int *__ierr){
*__ierr = TDySetDiscretization((TDy)PetscToPointer((_tdy)), *discretization);
}

PETSC_EXTERN void  tdysetfromoptions_(TDy tdy, int *__ierr){
*__ierr = TDySetFromOptions((TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdydriverinitializetdy_(TDy tdy, int *__ierr){
*__ierr = TDyDriverInitializeTDy((TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdytimeintegratorruntotime_(TDy tdy, PetscReal *time, int *__ierr){
*__ierr = TDyTimeIntegratorRunToTime((TDy)PetscToPointer((tdy)), *time);
}

PETSC_EXTERN void  tdytimeintegratorsettimestep_(TDy tdy, PetscReal *dtime, int *__ierr){
*__ierr = TDyTimeIntegratorSetTimeStep((TDy)PetscToPointer((tdy)), *dtime);
}

PETSC_EXTERN void  tdytimeintegratoroutputregression_(TDy tdy, int *__ierr){
*__ierr = TDyTimeIntegratorOutputRegression((TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdysetup_(TDy _tdy, int *__ierr){
*__ierr = TDySetup((TDy)PetscToPointer((_tdy)));
}

PETSC_EXTERN void  tdygetdm_(TDy _tdy, DM *dm, int *__ierr){
*__ierr = TDyGetDM((TDy)PetscToPointer((_tdy)), dm);
}

PETSC_EXTERN void  tdycreatediagnosticvector_(TDy _tdy, Vec *v, int *__ierr){
*__ierr = TDyCreateDiagnosticVector((TDy)PetscToPointer((_tdy)), v);
}

PETSC_EXTERN void  tdycreateprognosticvector_(TDy _tdy, Vec *v, int *__ierr){
*__ierr = TDyCreatePrognosticVector((TDy)PetscToPointer((_tdy)), v);
}

PETSC_EXTERN void  tdycreatematrix_(TDy _tdy, Mat *m, int *__ierr){
*__ierr = TDyCreateMatrix((TDy)PetscToPointer((_tdy)), m);
}

PETSC_EXTERN void  tdyupdatediagnostics_(TDy _tdy, int *__ierr){
*__ierr = TDyUpdateDiagnostics((TDy)PetscToPointer((_tdy)));
}

PETSC_EXTERN void  tdygetliquidsaturation_(TDy _tdy, Vec v, int *__ierr){
*__ierr = TDyGetLiquidSaturation((TDy)PetscToPointer((_tdy)), (Vec)PetscToPointer(v));
}

PETSC_EXTERN void  tdygetliquidmass_(TDy _tdy, Vec v, int *__ierr){
*__ierr = TDyGetLiquidMass((TDy)PetscToPointer((_tdy)), (Vec)PetscToPointer(v));
}

PETSC_EXTERN void  tdygetliquidpressure_(TDy _tdy, Vec v, int *__ierr){
*__ierr = TDyGetLiquidPressure((TDy)PetscToPointer((_tdy)), (Vec)PetscToPointer(v));
}

PETSC_EXTERN void  tdysetwaterdensitytype_(TDy tdy, PetscInt *method, int *__ierr){
*__ierr = TDySetWaterDensityType((TDy)PetscToPointer((tdy)), *method);
}

PETSC_EXTERN void  tdympfaosetgmatrixmethod_(TDy tdy, PetscInt *method, int *__ierr){
*__ierr = TDyMPFAOSetGmatrixMethod((TDy)PetscToPointer((tdy)), *method);
}

PETSC_EXTERN void  tdympfaosetboundaryconditiontype_(TDy tdy, PetscInt *bctype, int *__ierr){
*__ierr = TDyMPFAOSetBoundaryConditionType((TDy)PetscToPointer((tdy)), *bctype);
}

PETSC_EXTERN void  tdysetifunction_(TS ts, TDy tdy, int *__ierr){
*__ierr = TDySetIFunction(
  (TS)PetscToPointer((ts)),
  (TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdysetijacobian_(TS ts, TDy tdy, int *__ierr){
*__ierr = TDySetIJacobian(
  (TS)PetscToPointer((ts)),
  (TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdysetsnesfunction_(SNES snes, TDy tdy, int *__ierr){
*__ierr = TDySetSNESFunction(
  (SNES)PetscToPointer((snes)),
  (TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdysetsnesjacobian_(SNES snes, TDy tdy, int *__ierr){
*__ierr = TDySetSNESJacobian(
  (SNES)PetscToPointer((snes)),
  (TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdysetdtimeforsnessolver_(TDy tdy, PetscReal *dtime, int *__ierr){
*__ierr = TDySetDtimeForSNESSolver((TDy)PetscToPointer(tdy), *dtime);
}

PETSC_EXTERN void  tdysetinitialcondition_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDySetInitialCondition(
  (TDy)PetscToPointer(tdy),
  (Vec)PetscToPointer(solution));
}

PETSC_EXTERN void  tdygetinitialcondition_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDyGetInitialCondition(
  (TDy)PetscToPointer(tdy),
  (Vec)PetscToPointer(solution));
}

PETSC_EXTERN void  tdypresolvesnessolver_(TDy tdy, int *__ierr){
*__ierr = TDyPreSolveSNESSolver((TDy)PetscToPointer(tdy));
}

PETSC_EXTERN void  tdypostsolve_(TDy tdy, Vec solution, int *__ierr){
*__ierr = TDyPostSolve(
  (TDy)PetscToPointer(tdy),
  (Vec)PetscToPointer(solution));
}

PETSC_EXTERN void  tdycomputeerrornorms_(TDy tdy, Vec U, PetscReal *normp, PetscReal *normv,  int *__ierr){
*__ierr = TDyComputeErrorNorms(
  (TDy)PetscToPointer((tdy) ),
  (Vec)PetscToPointer((U) ),
  normp, normv);
}

PETSC_EXTERN void  tdycreatevectors_(TDy tdy, int *__ierr){
*__ierr = TDyCreateVectors(
  (TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdycreatejacobian_(TDy tdy, int *__ierr){
*__ierr = TDyCreateJacobian(
  (TDy)PetscToPointer((tdy)));
}

PETSC_EXTERN void  tdyoutputregression_(TDy tdy, Vec U, int *__ierr){
*__ierr = TDyOutputRegression(
  (TDy)PetscToPointer((tdy) ),
  (Vec)PetscToPointer((U) ));
}

PETSC_EXTERN void  tdydestroy_(TDy *_tdy, int *__ierr){
*__ierr = TDyDestroy(_tdy);
}

PETSC_EXTERN void tdyupdatestate_(TDy *tdy,PetscScalar y[], int ncells, int *ierr )
{
  *ierr = TDyUpdateState(*tdy,y,ncells);
}

//------------------------------------------------------------------------
//                  Fortran 90 spatial functions
//------------------------------------------------------------------------

// This is a context wrapped around an F90 spatial function (identified by an
// integer ID) that allows it to be called within a C spatial function.
typedef struct F90SpatialFunctionWrapper {
  PetscInt dim, id;
} F90SpatialFunctionWrapper;

typedef struct F90IntegerSpatialFunctionWrapper {
  PetscInt dim, id;
} F90IntegerSpatialFunctionWrapper;

// This subroutine is defined on the Fortran side and calls the Fortran-defined
// spatial function with the given ID.
extern void TDyCallF90SpatialFunction(PetscInt, PetscInt, PetscReal*,
                                      PetscReal*, PetscErrorCode*);

extern void TDyCallF90IntegerSpatialFunction(PetscInt, PetscInt, PetscReal*,
                                      PetscInt*, PetscErrorCode*);

// This creates a wrapped F90 function context.
static PetscErrorCode CreateF90SpatialFunctionContext(PetscInt dim,
                                                      PetscInt id,
                                                      void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  F90SpatialFunctionWrapper *wrapper;
  ierr = PetscMalloc(sizeof(F90SpatialFunctionWrapper), &wrapper); CHKERRQ(ierr);
  wrapper->dim = dim;
  wrapper->id = id;
  *context = wrapper;
  PetscFunctionReturn(0);
}

// This creates a wrapped F90 function context.
static PetscErrorCode CreateF90IntegerSpatialFunctionContext(PetscInt dim,
                                                      PetscInt id,
                                                      void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  F90IntegerSpatialFunctionWrapper *wrapper;
  ierr = PetscMalloc(sizeof(F90IntegerSpatialFunctionWrapper), &wrapper); CHKERRQ(ierr);
  wrapper->dim = dim;
  wrapper->id = id;
  *context = wrapper;
  PetscFunctionReturn(0);
}

// This calls the F90 function embedded within its context.
static PetscErrorCode WrappedF90SpatialFunction(void *context, PetscInt n,
                                                PetscReal *x, PetscReal *f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  F90SpatialFunctionWrapper *wrapper = context;
  TDyCallF90SpatialFunction(wrapper->id, n, x, f, &ierr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode WrappedF90IntegerSpatialFunction(void *context, PetscInt n,
                                                PetscReal *x, PetscInt *f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  F90SpatialFunctionWrapper *wrapper = context;
  TDyCallF90IntegerSpatialFunction(wrapper->id, n, x, f, &ierr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// This calls the F90 function embedded within its context, assigning its value
// to the diagonal components of an isotropic tensor
static PetscErrorCode WrappedF90IsotropicTensorFunction(void *context, PetscInt n,
                                                        PetscReal *x, PetscReal *f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  F90SpatialFunctionWrapper *wrapper = context;
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  PetscReal values[n];
  memset(f, 0, n * dim2 * sizeof(PetscReal));
  TDyCallF90SpatialFunction(wrapper->id, n, x, values, &ierr); CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim; ++j) {
      f[dim2*i+j*dim+j] = values[i];
    }
  }
  PetscFunctionReturn(0);
}

// This calls the F90 function embedded within its context, assigning its value
// to the components of a diagonal anisotropic tensor
static PetscErrorCode WrappedF90DiagonalTensorFunction(void *context, PetscInt n,
                                                       PetscReal *x, PetscReal *f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  F90SpatialFunctionWrapper *wrapper = context;
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  PetscReal values[dim*n];
  TDyCallF90SpatialFunction(wrapper->id, n, x, values, &ierr); CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim; ++j) {
      f[dim2*i+j*dim+j] = values[dim*i+j];
    }
  }
  PetscFunctionReturn(0);
}

// Generic destructor for contexts that wrap F90 functions.
static void DestroyContext(void* context) {
  PetscFree(context);
}

//------------------------------------------------------------------------
//                  Material properties and conditions
//------------------------------------------------------------------------

// This macro can be used to expose a Fortran 90 subroutine that assigns a
// spatial function to a material property.
#define WRAP_MATPROP(f90_fn, matprop_fn, wrapper_fn) \
PETSC_EXTERN PetscErrorCode f90_fn(TDy tdy, PetscInt id) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
  PetscInt dim; \
  ierr = DMGetDimension(((tdy->discretization)->tdydm)->dm, &dim); CHKERRQ(ierr); \
  void *context; \
  ierr = CreateF90SpatialFunctionContext(dim, id, &context); CHKERRQ(ierr); \
  ierr = matprop_fn(tdy->matprop, context, wrapper_fn, DestroyContext); \
  CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
} \

WRAP_MATPROP(TDySetPorosityFunctionF90, MaterialPropSetPorosity, WrappedF90SpatialFunction)

WRAP_MATPROP(TDySetIsotropicPermeabilityFunctionF90, MaterialPropSetPermeability, WrappedF90IsotropicTensorFunction)
WRAP_MATPROP(TDySetDiagonalPermeabilityFunctionF90, MaterialPropSetPermeability, WrappedF90DiagonalTensorFunction)
WRAP_MATPROP(TDySetTensorPermeabilityFunctionF90, MaterialPropSetPermeability, WrappedF90SpatialFunction)

WRAP_MATPROP(TDySetIsotropicThermalConductivityFunctionF90, MaterialPropSetThermalConductivity, WrappedF90IsotropicTensorFunction)
WRAP_MATPROP(TDySetDiagonalThermalConductivityFunctionF90, MaterialPropSetThermalConductivity, WrappedF90DiagonalTensorFunction)
WRAP_MATPROP(TDySetTensorThermalConductivityFunctionF90, MaterialPropSetThermalConductivity, WrappedF90SpatialFunction)

WRAP_MATPROP(TDySetResidualSaturationFunctionF90, MaterialPropSetResidualSaturation, WrappedF90SpatialFunction)
WRAP_MATPROP(TDySetSoilDensityFunctionF90, MaterialPropSetSoilDensity, WrappedF90SpatialFunction)
WRAP_MATPROP(TDySetSoilSpecificHeat, MaterialPropSetSoilSpecificHeat, WrappedF90SpatialFunction)

// This macro can be used to expose a Fortran 90 subroutine that assigns a
// spatial function to a boundary condition/source/sink.
#define WRAP_CONDITION(f90_fn, condition_fn, wrapper_fn) \
PETSC_EXTERN PetscErrorCode f90_fn(TDy tdy, PetscInt id) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
  PetscInt dim; \
  ierr = DMGetDimension(((tdy->discretization)->tdydm)->dm, &dim); CHKERRQ(ierr); \
  void *context; \
  ierr = CreateF90SpatialFunctionContext(dim, id, &context); CHKERRQ(ierr); \
  ierr = condition_fn(tdy->conditions, context, wrapper_fn, DestroyContext); \
  CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
} \

#define WRAP_CONDITION_INTEGER(f90_fn, condition_fn, wrapper_fn) \
PETSC_EXTERN PetscErrorCode f90_fn(TDy tdy, PetscInt id) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
  PetscInt dim; \
  ierr = DMGetDimension(((tdy->discretization)->tdydm)->dm, &dim); CHKERRQ(ierr); \
  void *context; \
  ierr = CreateF90IntegerSpatialFunctionContext(dim, id, &context); CHKERRQ(ierr); \
  ierr = condition_fn(tdy->conditions, context, wrapper_fn, DestroyContext); \
  CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
} \

WRAP_CONDITION(TDySetForcingFunctionF90, ConditionsSetForcing, WrappedF90SpatialFunction)
WRAP_CONDITION(TDySetEnergyForcingFunctionF90, ConditionsSetEnergyForcing, WrappedF90SpatialFunction)
WRAP_CONDITION(TDySetBoundaryPressureFunctionF90, ConditionsSetBoundaryPressure, WrappedF90SpatialFunction)
WRAP_CONDITION_INTEGER(TDySetBoundaryPressureTypeFunctionF90, ConditionsSetBoundaryPressureType, WrappedF90IntegerSpatialFunction)
WRAP_CONDITION(TDySetBoundaryTemperatureFunctionF90, ConditionsSetBoundaryTemperature, WrappedF90SpatialFunction)
WRAP_CONDITION(TDySetBoundaryVelocityFunctionF90, ConditionsSetBoundaryVelocity, WrappedF90SpatialFunction)

