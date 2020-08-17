#if !defined(TDYRICHARDS_H)
#define TDYRICHARDS_H

#include <petsc.h>
#include <tdycore.h>

PETSC_EXTERN PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                                     PetscBool*,PetscBool*,
                                                     void*);

#endif
