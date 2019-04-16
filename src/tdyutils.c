#include "tdyutils.h"
#include <petscblaslapack.h>

/* ---------------------------------------------------------------- */
PetscInt GetNumberOfCellVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,c,q,i,cStart,cEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nq = -1;
  for(c=cStart; c<cEnd; c++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    q = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh cells must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

/* ---------------------------------------------------------------- */
PetscInt GetNumberOfFaceVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,f,q,i,fStart,fEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  nq = -1;
  for(f=fStart; f<fEnd; f++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    q = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh faces must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}
