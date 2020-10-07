#include "petsc/finclude/petsc.h"

module routines

  use petsc

  implicit none

public 

contains

#define EIGHT
!#define TWO

subroutine set_cone_values(cone,offset,values)

  implicit none

  PetscInt :: cone(:)
  PetscInt :: offset
  PetscInt :: values(:)

  PetscInt :: num_values
  PetscInt :: i
  PetscInt :: mapping(8)

!  mapping = [1,2,3,4,5,6,7,8]
  mapping = [1,4,3,2,5,6,7,8]

  num_values = size(values)
  do i = 1, num_values
#if defined(EIGHT)
    cone(offset+i) = values(mapping(i)) + 8
#elif defined(TWO)
    cone(offset+i) = values(mapping(i)) + 2
#else
    cone(offset+i) = values(mapping(i)) + 1
#endif
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
  Mat :: Jacobian
  PetscErrorCode :: ierr

  PetscInt :: depth = 1
  PetscInt :: num_points(2)
#if defined(EIGHT)
  PetscInt :: cone_size(35)
  PetscInt :: cones(64)
  PetscInt :: cone_orientations(64)
  PetscReal :: vertex_coords(81)
#elif defined(TWO)
  PetscInt :: cone_size(14)
  PetscInt :: cones(16)
  PetscInt :: cone_orientations(16)
  PetscReal :: vertex_coords(36)
#else
  PetscInt :: cone_size(9)
  PetscInt :: cones(8)
  PetscInt :: cone_orientations(8)
  PetscReal :: vertex_coords(24)
#endif
  PetscInt :: i, j, k
  PetscInt :: pstart, pend
  PetscInt :: cstart, cend
  PetscInt :: fstart, fend
  PetscInt :: vstart, vend
  PetscInt :: offset
  PetscInt, pointer :: points(:)
  PetscInt, pointer :: cone(:)
  PetscInt, pointer :: support(:)
  PetscInt :: reverse_mapping(8)
  PetscInt :: icount
  PetscInt :: vertices(8)

  reverse_mapping = [1,4,3,2,5,6,7,8]

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  depth = 1
  cone_size = 0
  cones = 0
#if defined(EIGHT)
  num_points = [27,8]
  cone_size(1:8) = [8,8,8,8,8,8,8,8]
#elif defined(TWO)
  num_points = [12,2]
  cone_size(1:2) = [8,8]
#else
  num_points = [8,1]
  cone_size(1:1) = [8]
#endif
  offset = 0
#if defined(EIGHT)
  call set_cone_values(cones,offset,[1,2,5,4,10,11,14,13])
  call set_cone_values(cones,offset,[2,3,6,5,11,12,15,14])
  call set_cone_values(cones,offset,[4,5,8,7,13,14,17,16])
  call set_cone_values(cones,offset,[5,6,9,8,14,15,18,17])
  call set_cone_values(cones,offset,[10,11,14,13,19,20,23,22])
  call set_cone_values(cones,offset,[11,12,15,14,20,21,24,23])
  call set_cone_values(cones,offset,[13,14,17,16,22,23,26,25])
  call set_cone_values(cones,offset,[14,15,18,17,23,24,27,26])
#elif defined(TWO)
  call set_cone_values(cones,offset,[1,2,5,4,7,8,11,10])
  call set_cone_values(cones,offset,[2,3,6,5,8,9,12,11])
#else
  call set_cone_values(cones,offset,[1,2,4,3,5,6,8,7])
#endif
  cones = cones - 1 ! convert to zero-based indexing
  cone_orientations = 0

  vertex_coords = 0
  offset = 0
#if defined(EIGHT)
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,2.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,2.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,2.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,1.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,2.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,2.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,2.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,0.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,1.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,2.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,2.d0,2.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,2.d0,2.d0])
#elif defined(TWO)
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[2.d0,1.d0,1.d0])
#else
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,0.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,0.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[0.d0,1.d0,1.d0])
  call set_vertex_coords(vertex_coords,offset,[1.d0,1.d0,1.d0])
#endif

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
  ! get bounds of cell ids
  call DMPlexGetHeightStratum(dmi,0,cstart,cend,ierr)
  ! get bounds of face ids
  call DMPlexGetHeightStratum(dmi,1,fstart,fend,ierr)
  ! get bounds of vertex ids
  call DMPlexGetDepthStratum(dmi,0,vstart,vend,ierr)
  print *, 'cstart, cend: ', cstart, cend
  do i = cstart, cend-1
    write(*,'("pt(",i3,")")') i+1
    ! find neighboring cells
    write(*,'(" neighboring cells:")',advance="no")
    ! DMPlexGetCone() provides a list of face ids for the cell
    call DMPlexGetCone(dmi,i,cone,ierr)
    do j = 1, size(cone)
      if (cone(j) >= fstart .and. cone(j) < fend) then
        ! DMPlexGetSupport() provides a list of shared cell ids for the face
        call DMPlexGetSupport(dmi,cone(j),support,ierr)
        ! print the neighboring cells, skipping the current cell id
        do k = 1, size(support)
          if (support(k) /= i) then 
            write(*,'(x,i3)',advance="no") support(k)+1
          endif
        enddo        
        call DMPlexRestoreSupport(dmi,cone(j),support,ierr)
      endif
    enddo
    call DMPlexRestoreCone(dmi,i,cone,ierr)
    write(*,*) ''
    ! DMPlexGetTransitiveClosure() returns all face, edge and vertex ids 
    ! for the cell
    call DMPlexGetTransitiveClosure(dmi,i,PETSC_TRUE,points,ierr)
!    print *, i, points
    icount = 0
    do j = 1, size(points), 2 ! only need every other as the second integer
                              ! defines the orientation
      ! only add those ids that are vertices
      if (points(j) >= vstart .and. points(j) < vend) then
        icount = icount + 1
        vertices(icount) = points(j)+1
      endif
    enddo
    call DMPlexRestoreTransitiveClosure(dmi,i,PETSC_TRUE,points,ierr)
    write(*,'(" vertices:")',advance="no")
    do j = 1, icount
      ! reverse map to get back to PFLOTRAN/ExodusII ordering
      ! subtract 8 to get the true vertex id
      write(*,'(x,i3)',advance="no") vertices(reverse_mapping(j))-8
    enddo
    write(*,*) ''
    
  enddo
  call DMPlexGetChart(dmi,pstart,pend,ierr)
  print *, 'chart: ', pstart, pend
  call DMCreateMatrix(dmi,Jacobian,ierr)
  call MatView(Jacobian,PETSC_VIEWER_STDOUT_WORLD,ierr)
  call DMDestroy(dmi,ierr)
  call PetscFinalize(ierr)

end program mixed
