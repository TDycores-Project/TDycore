module mpfaof90mod

  use tdycore
#include <petsc/finclude/petsc.h>
#include <finclude/tdycore.h>

  ! Module variables
  PetscInt :: dim = 3

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

  subroutine PressureFunction(tdy,x,pressure,dummy,ierr)
     implicit none

      TDy                    :: tdy
      PetscReal, intent(in)  :: x(2)
      PetscReal, intent(out) :: pressure
      integer                :: dummy(*)
      PetscErrorCode         :: ierr

      pressure = 100000.d0

      ierr = 0
    end subroutine PressureFunction

  subroutine CreateDM(comm, dm, ierr)
    use petscdm
    implicit none

    MPI_Comm :: comm
    DM :: dm
    PetscErrorCode :: ierr

    PetscInt  :: faces(3), nx, ny, nz
    PetscReal :: lower(3), upper(3)

    nx = 3; ny = 3; nz = 3;
    faces(1) = nx; faces(2) = ny; faces(3) = nz;
    lower(:) = 0.d0;
    upper(:) = 1.d0;

    call DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper, &
       PETSC_NULL_INTEGER, PETSC_TRUE, dm, ierr);
    CHKERRA(ierr)
  end subroutine CreateDM
end module mpfaof90mod

program main

#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscdm.h>
#include <finclude/tdycore.h>

  use tdycore
  use petscvec
  use petscdm
  use petscksp
  use petscts
  use mpfaof90mod

implicit none

  TDy            :: tdy
  DM             :: dm
  Vec            :: U
  TS             :: ts
  PetscInt       :: rank, successful_exit_code
  PetscBool      :: flg
  PetscErrorCode :: ierr
  PetscInt, pointer :: index(:)
  PetscReal, pointer :: residualSat(:), blockPerm(:), liquid_sat(:), liquid_mass(:)
  PetscReal, pointer :: p_loc(:)
  PetscReal ::  perm(9)
  PetscInt :: c, cStart, cEnd, j, nvalues

  call TDyInit(ierr);
  CHKERRA(ierr);
  call TDyCreate(tdy, ierr);
  CHKERRA(ierr);
  call TDySetMode(tdy,RICHARDS,ierr);
  CHKERRA(ierr);
  call TDySetDiscretization(tdy,MPFA_O,ierr);
  CHKERRA(ierr);

  successful_exit_code= 0

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-successful_exit_code',successful_exit_code,flg,ierr);
  CHKERRA(ierr)
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);
  CHKERRA(ierr)

  ! Set a constructor for a DM.
  call TDySetDMConstructor(tdy, CreateDM,ierr);
  CHKERRA(ierr);

  call TDySetFromOptions(tdy,ierr);
  CHKERRA(ierr);

  call TDyGetDM(tdy, dm, ierr);
  call DMPlexGetHeightStratum(dm,0,cStart,cEnd,ierr);
  CHKERRA(ierr);

  allocate(blockPerm((cEnd-cStart)*dim*dim));
  allocate(residualSat(cEnd-cStart));
  allocate(liquid_mass(cEnd-cStart));
  allocate(liquid_sat(cEnd-cStart));
  allocate(index(cEnd-cStart));

  call Permeability(perm);

  do c=1,cEnd-cStart
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

  call TDySetBoundaryPressureFn(tdy,PressureFunction,0,ierr);
  CHKERRA(ierr);

  call TDySetBlockPermeabilityValuesLocal(tdy,cEnd-cStart,index,blockPerm,ierr);
  CHKERRA(ierr);

  call TDySetResidualSaturationValuesLocal(tdy,cEnd-cStart,index,residualSat,ierr);
  CHKERRA(ierr);

  call TDySetup(tdy,ierr);
  CHKERRA(ierr);

  ! Set initial condition
  call DMCreateGlobalVector(dm,U,ierr);
  CHKERRA(ierr);
  CHKERRA(ierr);
  call VecSet(U,91325.d0,ierr);
  CHKERRA(ierr);

  call TSCreate(PETSC_COMM_WORLD,ts,ierr);
  CHKERRA(ierr);

  call TSSetEquationType(ts,TS_EQ_IMPLICIT,ierr);
  CHKERRA(ierr);

  call TSSetType(ts,TSBEULER,ierr);
  CHKERRA(ierr);

  call TDySetIFunction(ts,tdy,ierr);
  CHKERRA(ierr);

  call TDySetIJacobian(ts,tdy,ierr);
  CHKERRA(ierr);

  call TSSetDM(ts,dm,ierr);
  CHKERRA(ierr);

  call TSSetSolution(ts,U,ierr);
  CHKERRA(ierr);

  call TSSetMaxSteps(ts,1,ierr);
  CHKERRA(ierr);

  call TSSetMaxTime(ts,1.d0,ierr);
  CHKERRA(ierr);

  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr);
  CHKERRA(ierr);

  call TSSetFromOptions(ts,ierr);
  CHKERRA(ierr);

  call TSSetUp(ts,ierr);
  CHKERRA(ierr);

  call VecSet(U,90325.d0,ierr);

  call VecGetArrayF90(U,p_loc,ierr);
  CHKERRA(ierr);

  call VecRestoreArrayF90(U,p_loc,ierr);
  CHKERRA(ierr);

  call TSSolve(ts,U,ierr);
  CHKERRA(ierr);

  call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
  CHKERRA(ierr);

  call TDyGetSaturationValuesLocal(tdy,nvalues,liquid_sat,ierr)
  CHKERRA(ierr);

  call TDyOutputRegression(tdy,U,ierr);
  CHKERRA(ierr);

  call TDyDestroy(tdy,ierr);
  CHKERRA(ierr);

  call TDyFinalize(ierr);
  CHKERRA(ierr);

  deallocate(blockPerm)
  deallocate(residualSat)
  deallocate(index)

  call exit(successful_exit_code)

end program
