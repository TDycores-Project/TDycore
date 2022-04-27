module snes_mpfaof90mod

  use tdycore
  use petscdm
#include <petsc/finclude/petsc.h>
#include <finclude/tdycore.h>

  ! DM-related variables for CreateDM.
  character (len=256) :: mesh_filename
  PetscBool           :: mesh_file_flg
  PetscInt            :: nx, ny, nz
  PetscInt            :: dim
  PetscInt            :: dm_plex_extrude_layers

contains

  subroutine PorosityFunction(n,x,theta,ierr)
    implicit none
    PetscInt,                intent(in)  :: n
    PetscReal, dimension(:), intent(in)  :: x
    PetscReal, dimension(:), intent(out) :: theta
    PetscErrorCode,          intent(out) :: ierr

    theta(:) = 0.25d0
    ierr = 0
  end subroutine PorosityFunction

  subroutine Permeability(K)
    implicit none
    PetscReal, intent(out) :: K(9)
    K(1) = 1.0d-12; K(2) = 0.0    ; K(3) = 0.0
    K(4) = 0.0    ; K(5) = 1.0d-12; K(6) = 0.0
    K(7) = 0.0    ; K(8) = 0.0    ; K(9) = 1.0d-12
  end subroutine Permeability

  subroutine PermeabilityFunction(n,x,K,ierr)
    implicit none
    PetscInt,                intent(in)  :: n
    PetscReal, dimension(:), intent(in)  :: x
    PetscReal, dimension(:), intent(out) :: K
    PetscErrorCode,          intent(out) :: ierr
    PetscReal :: K0(9)

    call Permeability(K0)
    K(:) = K0
    ierr = 0
  end subroutine PermeabilityFunction

  subroutine ResidualSaturation(resSat)
    implicit none
    PetscReal resSat

    resSat = 0.115d0
  end subroutine ResidualSaturation

  subroutine is_coordinate_top(x1,x2,x3,is_top)

    implicit none

    PetscReal, intent(in) :: x1,x2,x3
    PetscBool, intent(out) :: is_top

    PetscInt                :: ii
    PetscReal, dimension(5) :: zz_top

    zz_top(1) = 8.2500d0
    zz_top(2) = 7.6250d0
    zz_top(3) = 7.5000d0
    zz_top(4) = 8.5000d0
    zz_top(5) = 9.6250d0

    is_top = PETSC_FALSE
    do ii = 1,5
       if (x1 /= 1.d0 .and. x1 /= 6.d0 .and. x2 /= 1.d0 .and. x2 /= 2.d0 .and. x3 == zz_top(ii)) then
          is_top = PETSC_TRUE
       end if
    end do
  end subroutine is_coordinate_top

  subroutine PorosityFunctionPFLOTRAN(n,x,theta,ierr)
   implicit none
   PetscInt,                intent(in)  :: n
   PetscReal, dimension(:), intent(in)  :: x
   PetscReal, dimension(:), intent(out) :: theta
   PetscErrorCode,          intent(out) :: ierr

   theta(:) = 0.25d0
   ierr  = 0
 end subroutine PorosityFunctionPFLOTRAN

 subroutine Permeability_PFLOTRAN(K)
   implicit none
   PetscReal, intent(out) :: K(9)
   K(1) = 1.0d-12; K(2) = 0.0    ; K(3) = 0.0
   K(4) = 0.0    ; K(5) = 1.0d-12; K(6) = 0.0
   K(7) = 0.0    ; K(8) = 0.0    ; K(9) = 1.0d-13
 end subroutine Permeability_PFLOTRAN

 subroutine PermeabilityFunctionPFLOTRAN(n,x,K,ierr)
   implicit none
   PetscInt,                intent(in)  :: n
   PetscReal, dimension(:), intent(in)  :: x
   PetscReal, dimension(:), intent(out) :: K
   PetscErrorCode         :: ierr
   PetscReal :: K0(9)

   call Permeability_PFLOTRAN(K0)
   K(:) = K0
   ierr = 0
 end subroutine PermeabilityFunctionPFLOTRAN

 subroutine ResidualSaturation_PFLOTRAN(resSat)
   implicit none
   PetscReal resSat
   resSat = 0.115d0
 end subroutine ResidualSaturation_PFLOTRAN

 subroutine PressureFunction(n, x, pressure, ierr)

    implicit none

    PetscInt,                intent(in)  :: n
    PetscReal, dimension(:), intent(in)  :: x
    PetscReal, dimension(:), intent(out) :: pressure
    PetscErrorCode,          intent(out) :: ierr

    PetscInt             :: i
    PetscReal            :: x1, x2, x3
    PetscReal, parameter :: water_height       = 8.d0
    PetscReal, parameter :: pressure_reference = 101325.d0
    PetscReal, parameter :: gravity            = 9.81d0
    PetscReal, parameter :: rho                = 1000.d0
    PetscReal, parameter :: zero = 0.d0
    PetscReal, parameter :: seepage_pressure = 111325.d0
    PetscReal, parameter :: dirichlet_pressure = 70000.d0
    PetscBool :: is_top

    do i = 0, n-1
       x1 = x(3*i+1)
       x2 = x(3*i+2)
       x3 = x(3*i+3)
       
       call is_coordinate_top(x1,x2,x3,is_top)
       if (is_top) then
          pressure = seepage_pressure
       else
          pressure = dirichlet_pressure
       end if
    end do

    ierr = 0

  end subroutine PressureFunction

  subroutine PressureBCTypeFunction(n, x, bc_type, ierr)
    implicit none
    PetscInt,                intent(in)  :: n
    PetscReal, dimension(:), intent(in)  :: x
    PetscInt, dimension(:), intent(out) :: bc_type
    PetscErrorCode,          intent(out) :: ierr

    PetscInt             :: i
    PetscReal            :: x1, x2, x3
    PetscBool            :: is_top

    bc_type = NEUMANN_BC

    do i = 0, n-1
       x1 = x(3*i+1)
       x2 = x(3*i+2)
       x3 = x(3*i+3)

       call is_coordinate_top(x1,x2,x3,is_top)

       if (is_top) then
          bc_type = SEEPAGE_BC
       else
          bc_type = NEUMANN_BC
       end if
       
    end do

    ierr = 0

  end subroutine PressureBCTypeFunction

  subroutine CreateDM(comm, dm, ierr)
    use petscdm
    implicit none

    MPI_Comm :: comm
    DM :: dm, edm
    PetscErrorCode :: ierr

    PetscInt  :: faces(3)
    PetscReal :: lower(3), upper(3)

    if (.not.mesh_file_flg) then
       faces(1) = nx; faces(2) = ny; faces(3) = nz
       lower(:) = 0.d0
       upper(:) = 1.d0

       call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces, lower, upper, &
            PETSC_NULL_INTEGER, PETSC_TRUE, dm, ierr)
       CHKERRA(ierr)
    else
       call DMPlexCreateFromFile(PETSC_COMM_WORLD, mesh_filename, PETSC_TRUE, dm, ierr)
       CHKERRA(ierr)
       call DMGetDimension(dm, dim, ierr)
       CHKERRA(ierr)
       if (dm_plex_extrude_layers > 0) then
          call DMPlexExtrude(dm, PETSC_DETERMINE, -1.d0, PETSC_TRUE, PETSC_NULL_REAL, PETSC_TRUE, edm, ierr)
          CHKERRA(ierr)
          call DMDestroy(dm ,ierr)
          dm = edm
       end if
    endif

  end subroutine CreateDM

end module snes_mpfaof90mod

program main

#include <petsc/finclude/petsc.h>
#include <finclude/tdycore.h>
#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscdmplex.h>
#include <finclude/tdycore.h>

  use tdycore
  use petscvec
  use petscdm
  use petscdmplex
  use petscdmplexdef
  use petscksp
  use petscsnes
  use petscts
  use snes_mpfaof90mod

  implicit none

  TDy                 :: tdy
  DM                  :: dm
  Vec                 :: U
  SNES                :: snes
  PetscFE             :: fe
  PetscViewer         :: viewer
  SNESConvergedReason :: reason

  PetscInt            :: max_steps, step
  PetscInt            :: step_mod

  PetscReal           :: perm(9), resSat
  PetscReal           :: dtime, ic_value

  PetscBool           :: ic_file_flg, flg

  PetscErrorCode      :: ierr
  character (len=256) :: ic_filename
  character(len=256)  :: string
  PetscBool           :: use_seepage_bc

  PetscInt, parameter :: successful_exit_code = 0
  PetscInt, parameter :: unsuccessful_exit_code = -1
  PetscBool           :: pflotran_consistent, use_tdydriver
  PetscInt            :: bc_type

  max_steps = 1;
  dtime = 3600.d0
  ic_value = 102325.d0
  pflotran_consistent = PETSC_FALSE
  bc_type = NEUMANN_BC
  use_tdydriver = PETSC_FALSE

  call TDyInit(ierr); CHKERRA(ierr);
  call TDyCreate(tdy, ierr); CHKERRA(ierr);

  ! numerical method settings
  call TDySetMode(tdy,RICHARDS,ierr)
  call TDySetDiscretization(tdy, MPFA_O, ierr); CHKERRA(ierr);

  ! command line settings
  call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-ic_filename", ic_filename, ic_file_flg,ierr); CHKERRA(ierr);
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-max_steps',max_steps,flg,ierr); CHKERRA(ierr);
  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-use_seepage_bc",use_seepage_bc,flg,ierr); CHKERRA(ierr);
  call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-mesh_filename", mesh_filename, mesh_file_flg,ierr); CHKERRA(ierr)
  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-pflotran_consistent",pflotran_consistent,flg,ierr); CHKERRA(ierr)
  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-use_tdydriver",use_tdydriver,flg,ierr); CHKERRA(ierr)
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-dtime',dtime,flg,ierr); CHKERRA(ierr)

  ! Set a constructor for a DM.
  call TDySetDMConstructor(tdy, CreateDM,ierr); CHKERRA(ierr)

  ! Apply overrides.
  call TDySetFromOptions(tdy,ierr); CHKERRA(ierr)

  call TDyGetDM(tdy, dm, ierr); CHKERRA(ierr)
  call DMGetDimension(dm, dim, ierr);  CHKERRA(ierr)
  call PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "p_", -1, fe, ierr); CHKERRA(ierr)
  call PetscObjectSetName(fe, "p", ierr); CHKERRA(ierr)
  call DMSetField(dm, 0, PETSC_NULL_DMLABEL, fe, ierr); CHKERRA(ierr)
  call DMCreateDS(dm, ierr); CHKERRA(ierr)
  call PetscFEDestroy(fe, ierr); CHKERRA(ierr)
  call DMSetUseNatural(dm, PETSC_TRUE, ierr); CHKERRA(ierr)

  if (pflotran_consistent) then
     call Permeability_PFLOTRAN(perm)
     call ResidualSaturation_PFLOTRAN(resSat) 
  else
     call Permeability(perm)
     call ResidualSaturation(resSat)
  endif
  call TDySetConstantTensorPermeability(tdy, perm, ierr); CHKERRA(ierr)
  call TDySetConstantResidualSaturation(tdy, resSat, ierr); CHKERRA(ierr)

  !call TDySetWaterDensityType(tdy,WATER_DENSITY_EXPONENTIAL, ierr); CHKERRA(ierr)
  call TDySetWaterDensityType(tdy,WATER_DENSITY_CONSTANT, ierr); CHKERRA(ierr)

  call TDySetPorosityFunction(tdy,PorosityFunction,ierr); CHKERRA(ierr)
  call TDySetBoundaryPressureFunction(tdy,PressureFunction,ierr); CHKERRA(ierr)
  if (use_seepage_bc) then
     call TDySetBoundaryPressureTypeFunction(tdy,PressureBCTypeFunction, ierr); CHKERRA(ierr);
  end if

  call TDyMPFAOSetGMatrixMethod(tdy, MPFAO_GMATRIX_TPF, ierr);
  if (use_tdydriver) then
     call TDyDriverInitializeTDy(tdy, ierr); CHKERRA(ierr);
  else
     call TDySetup(tdy,ierr); CHKERRA(ierr)
  end if

  call TDyCreateVectors(tdy, ierr); CHKERRA(ierr)
  call TDyCreateJacobian(tdy, ierr); CHKERRA(ierr)

  ! Set initial condition
  call DMCreateGlobalVector(dm, U, ierr); CHKERRA(ierr);

  ! initial pressure
  if (ic_file_flg) then
    call PetscViewerBinaryOpen(PETSC_COMM_WORLD, ic_filename, FILE_MODE_READ, viewer, ierr); CHKERRA(ierr)
    call VecLoad(U, viewer, ierr); CHKERRA(ierr)
    call PetscViewerDestroy(viewer, ierr); CHKERRA(ierr)
  else
     ic_value = 101325.d0
     call VecSet(U,ic_value,ierr); CHKERRA(ierr);
  end if
  call TDySetInitialCondition(tdy, U, ierr); CHKERRA(ierr);
  call TDyGetInitialCondition(tdy, U, ierr); CHKERRA(ierr);

  ! create SNES
  call SNESCreate(PETSC_COMM_WORLD, snes, ierr); CHKERRA(ierr);
  call TDySetSNESFunction(snes, tdy, ierr); CHKERRA(ierr);
  call TDySetSNESJacobian(snes, tdy, ierr); CHKERRA(ierr);
  call SNESSetFromOptions(snes, ierr); CHKERRA(ierr);

  call TDySetDtimeForSNESSolver(tdy, dtime, ierr); CHKERRA(ierr);

  do step = 1, max_steps

     if (use_tdydriver) then
        call TDyTimeIntegratorSetTimeStep(tdy,dtime, ierr)
        CHKERRA(ierr)

        call TDyTimeIntegratorRunToTime(tdy,dtime * step, ierr)
        CHKERRA(ierr)

        call TDyGetLiquidPressure(tdy, U, ierr);

     else
        call TDyPreSolveSNESSolver(tdy,ierr); CHKERRA(ierr);
        call SNESSolve(snes, PETSC_NULL_VEC, U, ierr); CHKERRA(ierr)
        call SNESGetConvergedReason(snes,reason,ierr); CHKERRA(ierr)
        if (reason<0) then
           call PetscError(PETSC_COMM_WORLD, 0, PETSC_ERR_USER, "SNES did not converge")
        endif
        call TDyPostSolve(tdy,U,ierr);CHKERRA(ierr);
     endif


     step_mod = mod(step,1)
     if (step_mod == 0) then
        write(string,*) step
        string = 'solution_' // trim(adjustl(string)) // '.bin'
        write(*,*)'Writing output: ',trim(string)

        call PetscViewerBinaryOpen(PETSC_COMM_WORLD, trim(string), FILE_MODE_WRITE, viewer, ierr); CHKERRA(ierr)
        call VecView(U, viewer, ierr); CHKERRA(ierr)
        call PetscViewerDestroy(viewer, ierr); CHKERRA(ierr)
     endif
  end do

  call TDyOutputRegression(tdy,U,ierr)
  CHKERRA(ierr)

  call TDyFinalize(ierr); CHKERRA(ierr);

  call exit(successful_exit_code)

end program main
