#include <private/tdycoreimpl.h>

/* ---------------------------------------------------------------- */

PetscErrorCode ExtractSubGmatrix(TDy tdy, PetscInt cell_id,
                                 PetscInt sub_cell_id, PetscInt dim, PetscReal **Gmatrix) {

  PetscInt i, j;

  PetscFunctionBegin;

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      Gmatrix[i][j] = tdy->subc_Gmatrix[cell_id][sub_cell_id][i][j];
    }
  }

  PetscFunctionReturn(0);
}
