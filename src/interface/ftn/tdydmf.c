#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
#include <private/tdydmimpl.h>
#include <petsc/private/f90impl.h>

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))

#include "tdycore.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tdycreatedm_                                TDYCREATEDM
#define tdydistributedm_                            TDYDISTRIBUTEDM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tdycreatedm_                                tdycreatedm
#define tdydistributedm_                            tdydistributedm
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdycreatedm_(DM *dm,int *__ierr){
*__ierr = TDyCreateDM(dm);
}
#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void  tdydistributedm_(DM *dm,int *__ierr){
*__ierr = TDyDistributeDM(dm);
}
#if defined(__cplusplus)
}
#endif

