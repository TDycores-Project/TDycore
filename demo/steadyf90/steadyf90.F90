module steadyf90mod

  use tdycore
#include <petsc/finclude/petsc.h>
#include <finclude/tdycore.h>

contains

  subroutine PermeabilityFunction(tdy,x,K,dummy,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: K(4)
    integer                :: dummy(*)
    PetscErrorCode         :: ierr

    K(1) = 5.d0; K(2) = 1.d0
    K(3) = 1.d0; K(4) = 2.d0

    ierr = 0;
    
  end subroutine PermeabilityFunction
  
  subroutine PressureFunction(tdy,x,f,dummy,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: f
    integer                :: dummy(*)
    PetscErrorCode         :: ierr

    !(*f)  = PetscPowReal(1-x[0],4);
    !(*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
    !(*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
    f = (1-x(1))**4.d0 + (1.d0-x(1))*(1.d0-x(2))**3.d0 + cos(1.d0-x(1))*sin(1.d0-x(2));

    ierr = 0;

  end subroutine PressureFunction

  subroutine VelocityFunction(tdy,x,v,dummy,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: v(2)
    integer                :: dummy(*)
    PetscErrorCode         :: ierr

    PetscReal :: vx, vy
    PetscReal :: K(4)
    integer   :: dummy2(1)


    call PermeabilityFunction(tdy,x,K,dummy2,ierr);

    !vx  = -4*PetscPowReal(1-x[0],3);
    !vx += -  PetscPowReal(1-x[1],3);
    !vx += +PetscSinReal(x[1]-1)*PetscSinReal(x[0]-1);
    !vy  = -3*PetscPowReal(1-x[1],2)*(1-x[0]);
    !vy += -PetscCosReal(x[0]-1)*PetscCosReal(x[1]-1);

    vx = &
      -4.d0*((1.d0-x(1))**3.d0) &
      -     ((1.d0-x(2))**3.d0) &
      + sin((x(2)-1.d0)) * sin(x(1)-1.d0);
    vy = &
      -3.d0 * ((1.d0-x(2))**2.d0) *(1.d0-x(1)) &
      - cos(x(1)-1.d0) * cos(x(2)-1.d0);

    !v[0] = -(K[0]*vx+K[1]*vy);
    !v[1] = -(K[2]*vx+K[3]*vy);
    v(1) = -K(1)*vx - K(2)*vy;
    v(2) = -K(3)*vx - K(4)*vy;

    ierr = 0;

  end subroutine VelocityFunction

  subroutine ForcingFunction(tdy,x,f,dummy,ierr)

    implicit none

    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: f
    integer                :: dummy(*)
    PetscErrorCode         :: ierr

    PetscReal :: K(4)
    integer   :: dummy2(1)

    call PermeabilityFunction(tdy,x,K,dummy2,ierr);
    
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

end module steadyf90mod

program main

#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscdm.h>
#include <finclude/tdycore.h>

  use tdycore
  use petscvec
  use petscdm
  use petscksp
  use steadyf90mod

  implicit none

  TDy            :: tdy
  DM             :: dm, dmDist
  PetscInt       :: N, rank, method, successful_exit_code
  PetscBool      :: flg
  PetscInt       :: dim, faces(3)
  PetscReal      :: lower(3), upper(3)
  PetscReal      :: normp, normv
  Mat            :: K
  Vec            :: U,F
  KSP            :: ksp
  PetscErrorCode :: ierr

  call TDyInit(ierr)
  CHKERRA(ierr);
  call TDyCreate(tdy, ierr);
  CHKERRA(ierr);
  method = MPFA_O;
  call TDySetDiscretizationMethod(tdy,method, ierr);
  CHKERRA(ierr);
  call TDySetFromOptions(tdy, ierr);
  CHKERRA(ierr);

  N = 8
  dim = 2;
  successful_exit_code= 0

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-N',N,flg,ierr);
  CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-successful_exit_code',successful_exit_code,flg,ierr);
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
  call DMSetUp(dm,ierr);
  CHKERRA(ierr)
  call DMSetFromOptions(dm, ierr);
  CHKERRA(ierr);
  !call DMViewFromOptions(dm,PETSC_NULL_CHARACTER,'-dm_view',ierr); CHKERRA(ierr)

  call TDySetDM(dm, tdy, ierr);
  CHKERRA(ierr);
  call TDyAllocate(tdy, ierr);
  CHKERRA(ierr);

  call TDySetPermeabilityFunction(tdy,PermeabilityFunction,0,ierr);
  CHKERRA(ierr);
  call TDySetDirichletValueFunction(tdy,PressureFunction,0,ierr);
  CHKERRA(ierr);
  call TDySetForcingFunction(tdy,ForcingFunction,0,ierr);
  CHKERRA(ierr);
  call TDySetDirichletFluxFunction(tdy,VelocityFunction,0,ierr);
  CHKERRA(ierr);

  call TDySetup(tdy, ierr);
  CHKERRA(ierr);


  call DMCreateGlobalVector(dm,U,ierr);
  CHKERRA(ierr);
  call DMCreateGlobalVector(dm,F,ierr);
  CHKERRA(ierr);
  call DMCreateMatrix      (dm,K,ierr);
  CHKERRA(ierr);
  call TDyComputeSystem(tdy,K,F,ierr);
  CHKERRA(ierr);

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

  call TDyComputeErrorNorms(tdy,U,normp,normv,ierr);
  CHKERRA(ierr);
  write(*,*)normp,normv

  call TDyOutputRegression(tdy,U,ierr);
  CHKERRA(ierr);

  call KSPDestroy(ksp,ierr);
  CHKERRQ(ierr);

  call VecDestroy(U,ierr);
  CHKERRQ(ierr);

  call VecDestroy(F,ierr);
  CHKERRQ(ierr);

  call MatDestroy(K,ierr);
  CHKERRQ(ierr);

  call TDyDestroy(tdy,ierr);
  CHKERRQ(ierr);

  call TDyFinalize(ierr)
  CHKERRA(ierr);

  call exit(successful_exit_code)
end
