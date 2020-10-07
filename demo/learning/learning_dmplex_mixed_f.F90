#include "petsc/finclude/petsc.h"

module routines

  use petsc

  implicit none

public 

contains

subroutine set_cone_values(cone,offset,values)

  implicit none

  PetscInt :: cone(:)
  PetscInt :: offset
  PetscInt :: values(:)

  PetscInt :: num_values
  PetscInt :: i

  num_values = size(values)
  do i = 1, num_values
    cone(offset+i) = values(i) + 15
  enddo
  offset = offset + num_values

end subroutine set_cone_values

subroutine set_vertex_coords(coords,offset,values)

  implicit none

  PetscReal :: coords(:)
  PetscInt :: offset
  PetscReal :: values(:)

  PetscInt :: num_values
  PetscInt :: i

  num_values = size(values)
  do i = 1, num_values
    coords(offset+i) = values(i) 
  enddo
  offset = offset + num_values

end subroutine set_vertex_coords

end module routines

program mixed

  use petsc
  use routines

  implicit none

#include "petsc/finclude/petscdmplex.h"

  DM :: dm, dmi
  PetscErrorCode :: ierr

  PetscInt :: depth = 1
  PetscInt :: num_points(2)
  PetscInt :: cone_size(39)
  PetscInt :: cones(84)
  PetscInt :: cone_orientations(84)
  PetscInt :: offset
  PetscReal :: vertex_coords(72)

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  depth = 1
  num_points = [24,15]
  cone_size = 0
  cone_size(1:15) = [5,4,6,6,6,8,4,4,5,5,8,8,5,5,5]
  cones = 0
  offset = 0
  call set_cone_values(cones,offset,[4,5,6,2,1])
  call set_cone_values(cones,offset,[4,3,5,1])
  call set_cone_values(cones,offset,[2,7,6,4,9,5])
  call set_cone_values(cones,offset,[8,7,2,10,9,4])
  call set_cone_values(cones,offset,[10,9,4,21,14,11])
  call set_cone_values(cones,offset,[19,9,5,12,17,7,6,16])
  call set_cone_values(cones,offset,[5,13,14,15])
  call set_cone_values(cones,offset,[5,14,9,15])
  call set_cone_values(cones,offset,[5,9,19,12,15])
  call set_cone_values(cones,offset,[13,5,12,22,15])
  call set_cone_values(cones,offset,[20,10,9,19,18,8,7,17])
  call set_cone_values(cones,offset,[24,21,14,23,20,10,9,19])
  call set_cone_values(cones,offset,[23,19,9,14,15])
  call set_cone_values(cones,offset,[22,12,19,23,15])
  call set_cone_values(cones,offset,[22,23,14,13,15])
  cones = cones - 1 ! convert to zero-based indexing
  cone_orientations = 0

  vertex_coords = 0
  offset = 0
  call set_vertex_coords(vertex_coords,offset,[5.0d0,5.0d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[5.0d0,2.5d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[5.0d0,5.0d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[5.0d0,2.5d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,5.0d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,5.0d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,2.5d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,0.0d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,2.5d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,0.0d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[5.0d0,2.5d0,0.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,5.0d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,5.0d0,0.0d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,2.5d0,0.0d0])
  call set_vertex_coords(vertex_coords,offset,[1.25d0,3.75d0,1.25d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,5.0d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,2.5d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,0.0d0,5.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,2.5d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,0.0d0,2.5d0])
  call set_vertex_coords(vertex_coords,offset,[2.5d0,0.0d0,0.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,5.0d0,0.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,2.5d0,0.0d0])
  call set_vertex_coords(vertex_coords,offset,[0.0d0,0.0d0,0.0d0])

  print *, 'Hello World!'
  call DMPlexCreate(PETSC_COMM_WORLD,dm,ierr)
  call DMSetDimension(dm,3,ierr); CHKERRA(ierr)
  call DMPlexCreateFromDAG(dm,depth,num_points,cone_size,cones, &
                           cone_orientations,vertex_coords,ierr) 
  CHKERRA(ierr)
  call DMPlexInterpolate(dm,dmi,ierr)
  call DMPlexCopyCoordinates(dm,dmi,ierr)
!  call DMView(dm,PETSC_VIEWER_STDOUT_WORLD,ierr)
  call DMDestroy(dm,ierr)
  call DMView(dmi,PETSC_VIEWER_STDOUT_WORLD,ierr)
  call DMDestroy(dmi,ierr)
  call PetscFinalize(ierr)

end program mixed
