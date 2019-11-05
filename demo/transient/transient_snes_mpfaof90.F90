module f90module

  use tdycore
#include <petsc/finclude/petsc.h>
#include <finclude/tdycore.h>

contains

   subroutine PorosityFunction(tdy,x,theta,dummy,ierr)
     implicit none
     TDy                    :: tdy
     PetscReal, intent(in) :: x
     PetscReal, intent(out):: theta
     integer                :: dummy(*)
     PetscErrorCode :: ierr

     theta = 0.115d0
     ierr = 0
   end subroutine PorosityFunction

   subroutine Permeability(K)
     implicit none
     PetscReal, intent(out) :: K(9)
      K(1) = 1.0d-10; K(2) = 0.0    ; K(3) = 0.0    ;
      K(4) = 0.0    ; K(5) = 1.0d-10; K(6) = 0.0    ;
      K(7) = 0.0    ; K(8) = 0.0    ; K(9) = 1.0d-10;
   end subroutine

   subroutine PermeabilityFunction(tdy,x,K,dummy,ierr)
      implicit none
      TDy                    :: tdy
      PetscReal, intent(in)  :: x(2)
      PetscReal, intent(out) :: K(9)
      integer                :: dummy(*)
      PetscErrorCode         :: ierr

      call Permeability(K)

      ierr = 0
  end subroutine PermeabilityFunction

end module f90module

program main

#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscdm.h>
#include <finclude/tdycore.h>

  use tdycore
  use petscvec
  use petscdm
  use petscksp
  use petscsnes
  use f90module

implicit none

  TDy            :: tdy
  DM             :: dm, dmDist
  Vec            :: U
  !TS             :: ts
  SNES           :: snes
  PetscInt       :: rank, successful_exit_code
  PetscBool      :: flg
  PetscInt       :: dim, faces(3)
  PetscReal      :: lower(3), upper(3)
  PetscErrorCode :: ierr
  PetscInt       :: nx, ny, nz
  PetscInt, pointer :: index(:)
  PetscReal, pointer :: residualSat(:), blockPerm(:), liquid_sat(:), liquid_mass(:)
  PetscReal ::  perm(9)
  PetscInt :: c, cStart, cEnd, j, nvalues,g, max_steps, step
  PetscReal :: dtime, mass_pre, mass_post

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr);
  CHKERRA(ierr);

  nx = 1; ny = 1; nz = 15;
  dim = 3
  successful_exit_code= 0
  max_steps = 2

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-max_steps',max_steps,flg,ierr);
  CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-successful_exit_code',successful_exit_code,flg,ierr);
  CHKERRA(ierr)
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);
  CHKERRA(ierr)

  faces(1) = nx; faces(2) = ny; faces(3) = nz;
  lower(:) = 0.d0;
  upper(:) = 1.d0;

  call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces, lower, upper, &
       PETSC_NULL_INTEGER, PETSC_TRUE, dm, ierr);
  CHKERRA(ierr);
  call DMPlexDistribute(dm, 1, PETSC_NULL_SF, dmDist, ierr);
  CHKERRA(ierr);
  if (dmDist /= PETSC_NULL_DM) then
     call DMDestroy(dm, ierr);
     CHKERRA(ierr);
     dm = dmDist;
  end if
  call DMSetUp(dm,ierr);
  CHKERRA(ierr)

  call DMSetFromOptions(dm, ierr);
  CHKERRA(ierr);

  call TDyCreate(dm, tdy, ierr);
  CHKERRA(ierr);

  call DMPlexGetHeightStratum(dm,0,cStart,cEnd,ierr);
  CHKERRA(ierr);

  allocate(blockPerm((cEnd-cStart)*dim*dim));
  allocate(residualSat(cEnd-cStart));
  allocate(liquid_mass(cEnd-cStart));
  allocate(liquid_sat(cEnd-cStart));
  allocate(index(cEnd-cStart));

  call Permeability(perm);

  do c=1,cEnd-cStart+1
    index(c) = c-1;
    residualSat(c) = 0.115d0;
    do j = 1,dim*dim
      blockPerm((c-1)*dim*dim+j) = perm(j)
    enddo
  enddo

  call TDySetPorosityFunction(tdy,PorosityFunction,0,ierr);
  CHKERRA(ierr);

  call TDySetPermeabilityFunction(tdy,PermeabilityFunction,0,ierr);
  CHKERRA(ierr);

  call TDySetBlockPermeabilityValuesLocal(tdy,cEnd-cStart,index,blockPerm,ierr);
  CHKERRA(ierr);

  call TDySetResidualSaturationValuesLocal(tdy,cEnd-cStart,index,residualSat,ierr);
  CHKERRA(ierr);

  call TDySetDiscretizationMethod(tdy,MPFA_O,ierr);
  CHKERRA(ierr);

  call TDySetFromOptions(tdy,ierr);
  CHKERRA(ierr);

  ! Set initial condition
  call DMCreateGlobalVector(dm,U,ierr);
  CHKERRA(ierr);
  CHKERRA(ierr);
  call VecSet(U,91325.d0,ierr);
  CHKERRA(ierr);

  call SNESCreate(PETSC_COMM_WORLD,snes,ierr);
  CHKERRA(ierr);

  call TDySetSNESFunction(snes,tdy,ierr);
  CHKERRA(ierr);

  call TDySetSNESJacobian(snes,tdy,ierr);
  CHKERRA(ierr);

  call SNESSetFromOptions(snes,ierr);
  CHKERRA(ierr);

  call TDySetInitialSolutionForSNESSolver(tdy,U,ierr);
  CHKERRA(ierr);

  dtime = 1800.d0
  call TDySetDtimeForSNESSolver(tdy,dtime,ierr);
  CHKERRA(ierr);

  do step = 1,max_steps
    call TDyPreSolveSNESSolver(tdy,ierr);
    CHKERRA(ierr);

    call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
    CHKERRA(ierr);
    mass_pre = 0.d0
    do g = 1,nvalues
    mass_pre = mass_pre + liquid_mass(g)
    enddo

    call SNESSolve(snes,PETSC_NULL_VEC,U,ierr);
    CHKERRA(ierr);

    call TDyPostSolveSNESSolver(tdy,U,ierr);
    CHKERRA(ierr);

    call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
    CHKERRA(ierr);
    mass_post = 0.d0
    do g = 1,nvalues
    mass_post = mass_post + liquid_mass(g)
    enddo
    write(*,*)'Liquid mass pre,post,diff ',step,mass_pre,mass_post,mass_pre-mass_post
  end do

  call TDyGetSaturationValuesLocal(tdy,nvalues,liquid_sat,ierr)
  CHKERRA(ierr);

  call TDyOutputRegression(tdy,U,ierr);
  CHKERRA(ierr);

  call TDyDestroy(tdy,ierr);
  CHKERRA(ierr);

  call DMDestroy(dm,ierr);
  CHKERRA(ierr);

  call PetscFinalize(ierr);
  CHKERRA(ierr);

  deallocate(blockPerm)
  deallocate(residualSat)
  deallocate(index)

  call exit(successful_exit_code)

end program
