#include <tdyio.h>

PetscErrorCode TDyIOCreate(TDyIO *_io) {
  TDyIO io;
  PetscFunctionBegin;
  io = (TDyIO)malloc(sizeof(struct TDyIO));
  *_io = io;

  io->io_process = PETSC_FALSE;
  io->print_intermediate = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOPrintVec(Vec v,const char *prefix, int print_count) {
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

PetscErrorCode TDyIODestroy(TDyIO *io) {
  PetscFunctionBegin;
  free(*io);
  io = PETSC_NULL;
  PetscFunctionReturn(0);
}
