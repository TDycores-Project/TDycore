program main

#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscdm.h>
#include <finclude/tdycore.h>

  use tdycore
  use petscvec
  use petscdm

  implicit none

  TDy            :: tdy
  DM             :: dm, dmDist
  PetscInt       :: N, rank, method
  PetscBool      :: flg
  PetscInt       :: dim, faces(3), lower(3), upper(3)
  Mat            :: K
  Vec            :: U,F
  PetscErrorCode :: ierr

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  N = 8
  dim = 2;

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-N',N,flg,ierr);CHKERRA(ierr)
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)

  write(*,*)'This is a demo mpfaof90 code'

  faces(:) = N;
  lower(:) = 0.d0;
  upper(:) = 0.d0;
  
  call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces, lower, upper, &
       PETSC_NULL_INTEGER, PETSC_TRUE, dm, ierr); CHKERRA(ierr);
  call DMSetFromOptions(dm, ierr); CHKERRA(ierr);
  call DMPlexDistribute(dm, 1, PETSC_NULL_SF, dmDist, ierr); CHKERRA(ierr);
  if (dmDist /= PETSC_NULL_DM) then
     call DMDestroy(dm, ierr); CHKERRA(ierr);
     dm = dmDist;
  end if
  !call DMViewFromOptions(dm,PETSC_NULL_CHARACTER,'-dm_view',ierr); CHKERRA(ierr)
  

  call TDyCreate(dm, tdy, ierr);CHKERRA(ierr);
  method = 1;
  call TDySetDiscretizationMethod(tdy,method, ierr); CHKERRA(ierr);
  call TDySetFromOptions(tdy, ierr); CHKERRA(ierr);
  
  call DMCreateGlobalVector(dm,U,ierr); CHKERRA(ierr);
  call DMCreateGlobalVector(dm,F,ierr); CHKERRA(ierr);
  call DMCreateMatrix      (dm,K,ierr); CHKERRA(ierr);
  call TDyComputeSystem(tdy,K,F,ierr); CHKERRQ(ierr);

  call PetscFinalize(ierr)
  
end
