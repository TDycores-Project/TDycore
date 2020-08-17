#include "tdyout.h"

PetscErrorCode RichardsPrintVec(Vec v,char *prefix, int print_count) {
  char word[32];
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (print_count >= 0)
    sprintf(word,"%s_%d.txt",prefix,print_count);
  else
    sprintf(word,"%s.txt",prefix);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer);
         CHKERRQ(ierr);
  ierr = VecView(v,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
