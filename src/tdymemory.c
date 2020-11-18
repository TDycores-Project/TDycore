#include <private/tdymemoryimpl.h>

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_IntegerArray_1D(PetscInt *array_1D, PetscInt ndim_1,
    PetscInt init_value) {

  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    array_1D[i] = init_value;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_IntegerArray_2D(PetscInt **array_2D, PetscInt ndim_1,
                                          PetscInt ndim_2, PetscInt value) {

  PetscInt i,j;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      array_2D[i][j] = value;
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_1D(PetscReal *array_1D, PetscInt ndim_1,
                                       PetscReal value) {

  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    array_1D[i] = value;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_2D(PetscReal **array_2D, PetscInt ndim_1,
                                       PetscInt ndim_2, PetscReal value) {

  PetscInt i,j;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      array_2D[i][j] = value;
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_3D(PetscReal ***array_3D, PetscInt ndim_1,
                                       PetscInt ndim_2, PetscInt ndim_3, PetscReal value) {

  PetscInt i,j,k;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      for (k=0; k<ndim_3; k++) {
        array_3D[i][j][k] = value;
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_4D(PetscReal ****array_4D, PetscInt ndim_1,
                                       PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4, PetscReal value) {

  PetscInt i,j,k,l;
  PetscFunctionBegin;
  for (i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      for (k=0; k<ndim_3; k++) {
        for (l=0; l<ndim_4; l++) {
          array_4D[i][j][k][l] = value;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_IntegerArray_1D(PetscInt **array_1D, PetscInt ndim_1) {

  PetscErrorCode ierr;
  PetscFunctionBegin;
  *array_1D = (PetscInt *)malloc(ndim_1*sizeof(PetscInt));
  ierr = TDyInitialize_IntegerArray_1D(*array_1D, ndim_1, -1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_IntegerArray_2D(PetscInt ***array_2D, PetscInt ndim_1,
                                     PetscInt ndim_2) {

  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *array_2D = (PetscInt **)malloc(ndim_1*sizeof(PetscInt *));
  for(i=0; i<ndim_1; i++) {
    (*array_2D)[i] = (PetscInt *)malloc(ndim_2*sizeof(PetscInt ));
  }
  ierr = TDyInitialize_IntegerArray_2D(*array_2D, ndim_1, ndim_2, -1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_1D(PetscReal **array_1D, PetscInt ndim_1) {

  PetscErrorCode ierr;
  PetscFunctionBegin;
  *array_1D = (PetscReal *)malloc(ndim_1*sizeof(PetscReal ));
  ierr = TDyInitialize_RealArray_1D(*array_1D, ndim_1, 0.0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_2D(PetscReal ***array_2D, PetscInt ndim_1,
                                     PetscInt ndim_2) {

  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *array_2D = (PetscReal **)malloc(ndim_1*sizeof(PetscReal *));
  for(i=0; i<ndim_1; i++) {
    (*array_2D)[i] = (PetscReal *)malloc(ndim_2*sizeof(PetscReal ));
  }
  ierr = TDyInitialize_RealArray_2D(*array_2D, ndim_1, ndim_2, 0.0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_3D(PetscReal ****array_3D, PetscInt ndim_1,
                                     PetscInt ndim_2, PetscInt ndim_3) {

  PetscInt i,j;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *array_3D = (PetscReal ***)malloc(ndim_1*sizeof(PetscReal **));
  for(i=0; i<ndim_1; i++) {
    (*array_3D)[i] = (PetscReal **)malloc(ndim_2*sizeof(PetscReal *));
    for(j=0; j<ndim_2; j++) {
      (*array_3D)[i][j] = (PetscReal *)malloc(ndim_3*sizeof(PetscReal));
    }
  }
  ierr = TDyInitialize_RealArray_3D(*array_3D, ndim_1, ndim_2, ndim_3, 0.0);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_4D(PetscReal *****array_4D, PetscInt ndim_1,
                                     PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4) {

  PetscInt i,j,k;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *array_4D = (PetscReal ****)malloc(ndim_1*sizeof(PetscReal ***));
  for(i=0; i<ndim_1; i++) {
    (*array_4D)[i] = (PetscReal ***)malloc(ndim_2*sizeof(PetscReal **));
    for(j=0; j<ndim_2; j++) {
      (*array_4D)[i][j] = (PetscReal **)malloc(ndim_3*sizeof(PetscReal *));
      for(k=0; k<ndim_3; k++) {
        (*array_4D)[i][j][k] = (PetscReal *)malloc(ndim_4*sizeof(PetscReal));
      }
    }
  }
  ierr = TDyInitialize_RealArray_4D(*array_4D, ndim_1, ndim_2, ndim_3, ndim_4, 0.0);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_IntegerArray_2D(PetscInt **array_2D, PetscInt ndim_1) {

  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    free(array_2D[i]);
  }
  free(array_2D);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_2D(PetscReal **array_2D, PetscInt ndim_1) {

  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    free(array_2D[i]);
  }
  free(array_2D);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_3D(PetscReal ***array_3D, PetscInt ndim_1, PetscInt ndim_2) {

  PetscInt i,j;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for(j=0; j<ndim_2; j++) {
      free(array_3D[i][j]);
    }
    free(array_3D[i]);
  }
  free(array_3D);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_4D(PetscReal ****array_4D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3) {

  PetscInt i,j,k;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for(j=0; j<ndim_2; j++) {
      for(k=0; k<ndim_3; k++) {
        free(array_4D[i][j][k]);
      }
      free(array_4D[i][j]);
    }
    free(array_4D[i]);
  }
  free(array_4D);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyCell_1D(PetscInt ndim_1, TDyCell **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDyCell *)malloc(ndim_1*sizeof(TDyCell));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyVertex_1D(PetscInt ndim_1, TDyVertex **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDyVertex *)malloc(ndim_1*sizeof(TDyVertex));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyEdge_1D(PetscInt ndim_1, TDyEdge **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDyEdge *)malloc(ndim_1*sizeof(TDyEdge));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyFace_1D(PetscInt ndim_1, TDyFace **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDyFace *)malloc(ndim_1*sizeof(TDyFace));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDySubcell_1D(PetscInt ndim_1, TDySubcell **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDySubcell *)malloc(ndim_1*sizeof(TDySubcell));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyVector_1D(PetscInt ndim_1, TDyVector **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDyVector *)malloc(ndim_1*sizeof(TDyVector));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyCoordinate_1D(PetscInt ndim_1, TDyCoordinate **array_1D) {

  PetscFunctionBegin;
  *array_1D = (TDyCoordinate *)malloc(ndim_1*sizeof(TDyCoordinate));
  PetscFunctionReturn(0);
}
