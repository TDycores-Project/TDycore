#include "petsc/finclude/petsc.h"

module box_mesh_routines

  use petsc

  implicit none

public 

contains

end module box_mesh_routines

program mixed

  use petsc
  use box_mesh_routines

  implicit none

#include "petsc/finclude/petscdmplex.h"

  DM :: dm, dmi
  Mat :: Jacobian
  PetscInt :: faces(3)
  PetscReal :: lower(3), upper(3)
  PetscErrorCode :: ierr

  PetscInt :: i, j, k
  PetscInt :: pstart, pend
  PetscInt :: cstart, cend
  PetscSection :: section

  PetscBool :: use_cone
  PetscBool :: use_closure

  faces = 2
  lower = 0.d0
  upper = 1.d0

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  call DMPlexCreateBoxMesh(PETSC_COMM_WORLD,3,PETSC_FALSE,faces,lower,upper, &
                           DM_BOUNDARY_NONE,PETSC_TRUE,dmi,ierr)
  call DMView(dmi,PETSC_VIEWER_STDOUT_WORLD,ierr)
#if 1
  call PetscSectionCreate(PETSC_COMM_WORLD,section,ierr)
  call PetscSectionSetNumFields(section,1,ierr)
  call PetscSectionSetFieldName(section,0,"Pressure",ierr)
  call PetscSectionSetFieldComponents(section,0,1,ierr)
  call DMPlexGetChart(dmi,pstart,pend,ierr)
  call PetscSectionSetChart(section,pstart,pend,ierr)
  print *, 'chart: ', pstart, pend
  call DMPlexGetHeightStratum(dmi,0,cstart,cend,ierr)
  do i = cstart, cend-1
    call PetscSectionSetFieldDof(section,i,0,1,ierr)
    call PetscSectionSetDof(section,i,1,ierr)
  enddo
  call PetscSectionSetup(section,ierr)
  call DMSetSection(dmi,section,ierr)
  call PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD,ierr)
  call PetscSectionDestroy(section,ierr)
#endif
  ! star stencil: TRUE, FALSE
  ! box stencil: TRUE, TRUE
  use_cone = PETSC_TRUE
  use_closure = PETSC_FALSE
  call DMSetBasicAdjacency(dmi,use_cone,use_closure,ierr)
  print *, '--'
  call DMCreateMatrix(dmi,Jacobian,ierr)
  call MatView(Jacobian,PETSC_VIEWER_STDOUT_WORLD,ierr)
  call DMDestroy(dmi,ierr)
  call PetscFinalize(ierr)

end program mixed
