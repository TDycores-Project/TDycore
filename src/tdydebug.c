#include <tdycoreprivate.h>

void PrintMatrix(PetscReal *A,PetscInt nr,PetscInt nc,PetscBool row_major) {
  PetscInt i,j;
  printf("[[");
  for(i=0; i<nr; i++) {
    if(i>0) printf(" [");
    for(j=0; j<nc; j++) {
      if(row_major) {
        printf("%+.4f, ",A[i*nc+j]);
      } else {
        printf("%+.4f, ",A[j*nr+i]);
      }
    }
    printf("]");
    if(i<nr-1) printf(",\n");
  }
  printf("]\n");
}

PetscErrorCode CheckSymmetric(PetscReal *A,PetscInt n) {
  PetscInt i,j;
  PetscErrorCode ierr = 0;
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      if(PetscAbsReal(A[i*n+j]-A[j*n+i]) > 1e-12) {
        printf("Symmetry Error A[%d,%d] = %f, A[%d,%d] = %f\n",i,j,A[i*n+j],j,i,
               A[j*n+i]);
        ierr = 64;
      }
    }
  }
  return ierr;
}
