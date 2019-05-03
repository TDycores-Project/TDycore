module f90module

  use tdycore
#include <petsc/finclude/petsc.h>
#include <finclude/tdycore.h>
  type userctx
     PetscInt :: a
  end type userctx

contains

  subroutine PermeabilityFunction(tdy,x,K,user,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: K(4)
    type(userctx)          :: user
    PetscErrorCode         :: ierr

    K(1) = 5.d0; K(2) = 1.d0
    K(3) = 1.d0; K(4) = 2.d0

    ierr = 0;
    
  end subroutine PermeabilityFunction
  
  subroutine PressureFunction(tdy,x,f,user,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: f
    type(userctx)          :: user
    PetscErrorCode         :: ierr

    !(*f)  = PetscPowReal(1-x[0],4);
    !(*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
    !(*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
    f = (1-x(1))**4.d0 + (1.d0-x(1))*(1.d0-x(2))**3.d0 + cos(1.d0-x(1))*sin(1.d0-x(2));

    ierr = 0;

  end subroutine PressureFunction

  subroutine ForcingFunction(tdy,x,f,user,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: f
    type(userctx)          :: user
    PetscErrorCode         :: ierr

    PetscReal :: K(4)
    
    call PermeabilityFunction(tdy,x,K,user,ierr);
    
    !(*f)  = -K[0]*(12*PetscPowReal(1-x[0],2)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
    !(*f) += -K[1]*( 3*PetscPowReal(1-x[1],2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
    !(*f) += -K[2]*( 3*PetscPowReal(1-x[1],2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
    !(*f) += -K[3]*(-6*(1-x[0])*(x[1]-1)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
    f = &
         -K(1)*(12.d0*((1.d0-x(1))**2.d0) + sin(x(2)-1.d0)*cos(x(1)-1.d0)) + &
         -K(2)*( 3.d0*((1.d0-x(2))**2.d0) + sin(x(1)-1.d0)*cos(x(2)-1.d0)) + &
         -K(3)*( 3.d0*((1.d0-x(2))**2.d0) + sin(x(1)-1.d0)*cos(x(2)-1.d0)) + &
         -K(4)*(-6.d0*(1.d0-x(1))*(x(2)-1.d0) + sin(x(2)-1.d0)*cos(x(1)-1.d0));

    ierr = 0;

  end subroutine ForcingFunction

end module f90module

program main

#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscdm.h>
#include <finclude/tdycore.h>

  use tdycore
  use petscvec
  use petscdm
  use petscksp
  use f90module

  implicit none

  TDy            :: tdy
  DM             :: dm, dmDist
  PetscInt       :: N, rank, method
  PetscBool      :: flg
  PetscInt       :: dim, faces(3)
  PetscReal      :: lower(3), upper(3)
  Mat            :: K
  Vec            :: U,F
  type (userctx) :: user
  KSP            :: ksp
  PetscErrorCode :: ierr

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  N = 8
  dim = 2;

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-N',N,flg,ierr);
  CHKERRA(ierr)
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);
  CHKERRA(ierr)

  faces(:) = N;
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
  call DMSetUp(dm,ierr); CHKERRA(ierr)
  call DMSetFromOptions(dm, ierr);
  CHKERRA(ierr);
  !call DMViewFromOptions(dm,PETSC_NULL_CHARACTER,'-dm_view',ierr); CHKERRA(ierr)

  call TDyCreate(dm, tdy, ierr);
  CHKERRA(ierr);

  call TDySetPermeabilityFunction(tdy,PermeabilityFunction,user,ierr); CHKERRA(ierr);
  call TDySetDirichletValueFunction(tdy,PressureFunction,user,ierr); CHKERRA(ierr);
  call TDySetForcingFunction2(tdy,ForcingFunction,user,ierr); CHKERRA(ierr);

  method = 1;
  call TDySetDiscretizationMethod(tdy,method, ierr);
  CHKERRA(ierr);
  call TDySetFromOptions(tdy, ierr);
  CHKERRA(ierr);


  call DMCreateGlobalVector(dm,U,ierr);
  CHKERRA(ierr);
  call DMCreateGlobalVector(dm,F,ierr);
  CHKERRA(ierr);
  call DMCreateMatrix      (dm,K,ierr);
  CHKERRA(ierr);
  call TDyComputeSystem(tdy,K,F,ierr); CHKERRQ(ierr);

  !Solve system
  call KSPCreate(PETSC_COMM_WORLD,ksp,ierr);
  CHKERRQ(ierr);

  call KSPSetOperators(ksp,K,K,ierr);
  CHKERRQ(ierr);
  call KSPSetFromOptions(ksp,ierr);
  CHKERRQ(ierr);

  call KSPSetUp(ksp,ierr);
  CHKERRQ(ierr);

  call KSPSolve(ksp,F,U,ierr);
  CHKERRQ(ierr);


  call PetscFinalize(ierr)
  
end
