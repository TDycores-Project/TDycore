#if !defined(TDYREGRESSIONIMPL_H)
#define TDYREGRESSIONIMPL_H

#include <petsc.h>
#include "tdycore.h"

typedef struct _TDy_regression TDy_regression;

struct _TDy_regression {
  char filename[PETSC_MAX_PATH_LEN];
  PetscInt num_cells_per_process;
  PetscInt *cells_per_process_natural_ids;
  Vec cells_per_process_vec;
  VecScatter scatter_cells_per_process_gtos;
};

PETSC_EXTERN PetscErrorCode TDyRegressionInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyRegressionOutput(TDy,Vec);

#endif
