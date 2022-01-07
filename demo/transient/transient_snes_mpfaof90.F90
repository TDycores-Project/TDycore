module snes_mpfaof90mod

  use tdycore
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

     theta(:) = 0.115d0
     ierr = 0
   end subroutine PorosityFunction

   subroutine Permeability(K)
     implicit none
     PetscReal, intent(out) :: K(9)
      K(1) = 1.0d-10; K(2) = 0.0    ; K(3) = 0.0    ;
      K(4) = 0.0    ; K(5) = 1.0d-10; K(6) = 0.0    ;
      K(7) = 0.0    ; K(8) = 0.0    ; K(9) = 1.0d-10;
   end subroutine

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

  subroutine PorosityFunctionPFLOTRAN(n,x,theta,ierr)
    implicit none
    PetscInt,                intent(in)  :: n
    PetscReal, dimension(:), intent(in)  :: x
    PetscReal, dimension(:), intent(out) :: theta
    PetscErrorCode,          intent(out) :: ierr

    theta(:) = 0.3d0
    ierr  = 0
  end subroutine PorosityFunctionPFLOTRAN

  subroutine Permeability_PFLOTRAN(K)
    implicit none
    PetscReal, intent(out) :: K(9)
    K(1) = 1.0d-12; K(2) = 0.0    ; K(3) = 0.0    ;
    K(4) = 0.0    ; K(5) = 1.0d-12; K(6) = 0.0    ;
    K(7) = 0.0    ; K(8) = 0.0    ; K(9) = 5.0d-13;
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

  subroutine MaterialPropAlpha_PFLOTRAN(alpha)
    implicit none
    PetscReal, intent(out) :: alpha
    alpha = 1.d-4
  end subroutine MaterialPropAlpha_PFLOTRAN

  subroutine MaterialPropM_PFLOTRAN(m)
    implicit none
    PetscReal, intent(out) :: m
    m = 0.3d0
  end subroutine MaterialPropM_PFLOTRAN

  subroutine ResidualSat_PFLOTRAN(resSat)
    implicit none
    PetscReal resSat
    resSat = 0.1d0
  end subroutine ResidualSat_PFLOTRAN

  subroutine PressureFunction(n, x, pressure,ierr)
    implicit none
    PetscInt,                intent(in)  :: n
    PetscReal, dimension(:), intent(in)  :: x
    PetscReal, dimension(:), intent(out) :: pressure
    PetscErrorCode,          intent(out) :: ierr

    PetscInt :: i
    PetscReal :: x1, x2, x3

    do i = 0, n-1
      x1 = x(3*i+1)
      x2 = x(3*i+2)
      x3 = x(3*i+3)
      if (x1 > 0.d0 .and. x1 < 1.d0 .and. x2 > 0.d0 .and. x2 < 1.d0 .and. x3 > 0.99d0) then
        pressure = 110000.d0
      else
        pressure = 100000.d0
      end if
    end do
    ierr = 0
  end subroutine PressureFunction

  subroutine CreateDM(comm, dm, ierr)
    use petscdm
    implicit none

    MPI_Comm :: comm
    DM :: dm, edm
    PetscErrorCode :: ierr

    PetscInt  :: faces(3)
    PetscReal :: lower(3), upper(3)

    if (.not.mesh_file_flg) then
      faces(1) = nx; faces(2) = ny; faces(3) = nz;
      lower(:) = 0.d0;
      upper(:) = 1.d0;

      call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces, lower, upper, &
           PETSC_NULL_INTEGER, PETSC_TRUE, dm, ierr);
      CHKERRA(ierr);
    else
      call DMPlexCreateFromFile(PETSC_COMM_WORLD, mesh_filename, PETSC_TRUE, dm, ierr);
      CHKERRA(ierr);
      call DMGetDimension(dm, dim, ierr);
      CHKERRA(ierr);
      if (dm_plex_extrude_layers > 0) then
        call DMPlexExtrude(dm, PETSC_DETERMINE, -1.d0, PETSC_TRUE, PETSC_NULL_REAL, PETSC_TRUE, edm, ierr);
        CHKERRA(ierr);
        call DMDestroy(dm ,ierr);
        dm = edm
      end if
  endif

  end subroutine

end module snes_mpfaof90mod

program main

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
  use snes_mpfaof90mod

implicit none

  TDy                 :: tdy
  DM                  :: dm, diags_dm
  Vec                 :: U, diags
  PetscSection        :: diags_section
  PetscInt            :: n_diags, i_diag, i_liquid_mass
  character (len=256) :: diag_name
  !TS                 :: ts
  SNES                :: snes
  PetscInt            :: rank, successful_exit_code
  PetscBool           :: flg
  PetscErrorCode      :: ierr
  PetscInt            :: ncell, bc_type
  PetscInt  , pointer :: index(:)
  PetscReal , pointer :: liquid_sat(:), liquid_mass(:)
  PetscReal , pointer :: alpha(:), m(:)
  PetscReal           :: perm(9), resSat
  PetscInt            :: c, cStart, cEnd, j, nvalues,g, max_steps, step
  PetscReal           :: dtime, mass_pre, mass_post, ic_value
  character (len=256) :: ic_filename
  character(len=256)  :: string, bc_type_name
  PetscBool           :: ic_file_flg, pflotran_consistent, use_tdydriver
  PetscViewer         :: viewer
  PetscInt            :: step_mod
  PetscFE             :: fe
  SNESConvergedReason :: reason

  dim = 3
  nx = 1; ny = 1; nz = 15;

  call TDyInit(ierr);
  CHKERRA(ierr);

  ! Register some functions.
  CHKERRA(ierr);

  call TDyCreate(tdy, ierr);
  CHKERRA(ierr);
  call TDySetMode(tdy,RICHARDS,ierr);
  CHKERRA(ierr);
  call TDySetDiscretization(tdy,MPFA_O,ierr);
  CHKERRA(ierr);

  successful_exit_code= 0
  max_steps = 2
  dtime = 1800.d0
  ic_value = 102325.d0
  pflotran_consistent = PETSC_FALSE
  bc_type = MPFAO_NEUMANN_BC
  use_tdydriver = PETSC_FALSE
  dm_plex_extrude_layers=0

  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);
  CHKERRA(ierr)

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-dm_plex_extrude_layers',dm_plex_extrude_layers,flg,ierr)
  CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-max_steps',max_steps,flg,ierr);
  CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-successful_exit_code',successful_exit_code,flg,ierr);
  CHKERRA(ierr)
  call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-mesh_filename", mesh_filename, mesh_file_flg,ierr);
  CHKERRA(ierr);
  call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-ic_filename", ic_filename, ic_file_flg,ierr);
  CHKERRA(ierr);
  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-pflotran_consistent",pflotran_consistent,flg,ierr);
  CHKERRA(ierr)

  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-use_tdydriver",use_tdydriver,flg,ierr);
  CHKERRA(ierr)

  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-dtime',dtime,flg,ierr)
  CHKERRA(ierr)
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-ic_value',ic_value,flg,ierr)
  CHKERRA(ierr)

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nx',nx,flg,ierr)
  CHKERRA(ierr)

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-ny',ny,flg,ierr)
  CHKERRA(ierr)

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nz',nz,flg,ierr)
  CHKERRA(ierr)

  call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-tdy_mpfao_boundary_condition_type',bc_type_name,flg,ierr)
  CHKERRA(ierr)
  if (trim(bc_type_name) == 'MPFAO_DIRICHLET_BC') then
    bc_type = MPFAO_DIRICHLET_BC
  else if (trim(bc_type_name) == 'MPFAO_SEEPAGE_BC') then
    bc_type = MPFAO_SEEPAGE_BC
  endif

  ! Set a constructor for a DM.
  call TDySetDMConstructor(tdy, CreateDM,ierr);
  CHKERRA(ierr);

  ! Apply overrides.
  call TDySetFromOptions(tdy,ierr);
  CHKERRA(ierr);

  ! Set up the discretization.
  call TDyGetDM(tdy, dm, ierr);
  CHKERRA(ierr);
  call DMGetDimension(dm, dim, ierr);
  CHKERRA(ierr)
  call PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "p_", -1, fe, ierr);
  CHKERRA(ierr)
  call PetscObjectSetName(fe, "p", ierr);
  CHKERRA(ierr)
  call DMSetField(dm, 0, PETSC_NULL_DMLABEL, fe, ierr);
  CHKERRA(ierr)
  call DMCreateDS(dm, ierr);
  CHKERRA(ierr)
  call PetscFEDestroy(fe, ierr);
  CHKERRA(ierr)
  call DMSetUseNatural(dm, PETSC_TRUE, ierr);
  CHKERRA(ierr)
  call DMPlexGetHeightStratum(dm,0,cStart,cEnd,ierr);
  CHKERRA(ierr);

  ncell = (cEnd-cStart)
  allocate(liquid_mass (ncell))
  allocate(liquid_sat  (ncell))
  allocate(alpha       (ncell))
  allocate(m           (ncell))
  allocate(index       (ncell))

  ! TODO: We disable the PFlotran stuff for now, because it requires us to
  ! TODO: directly fiddle with characteristic curves.
  if (pflotran_consistent) then
    print *, "pflotran_consistent option is temporarily disabled!"
    stop
    call Permeability_PFLOTRAN(perm);
    call ResidualSat_PFLOTRAN(resSat)
  else
     call Permeability(perm);
     call ResidualSaturation(resSat)
  end if
  call TDySetConstantTensorPermeability(tdy, perm, ierr)
  CHKERRA(ierr)
  call TDySetConstantResidualSaturation(tdy, resSat, ierr)
  CHKERRA(ierr)

  call TDySetWaterDensityType(tdy,WATER_DENSITY_EXPONENTIAL,ierr);
  CHKERRA(ierr)

  if (pflotran_consistent) then
!     call TDySetPorosityFunction(tdy,PorosityFunctionPFLOTRAN,ierr);
!     CHKERRA(ierr);
!
!     do c = 1,ncell
!        index(c) = c-1;
!        call MaterialPropAlpha_PFLOTRAN(alpha(c))
!        call MaterialPropM_PFLOTRAN(m(c))
!     enddo
!
!     call TDySetCharacteristicCurveVanGenuchtenValuesLocal(tdy,ncell,index,m,alpha,ierr)
!     CHKERRA(ierr);
!
!     call TDySetCharacteristicCurveMualemValuesLocal(tdy,ncell,index,m,ierr)
!     CHKERRA(ierr);
  else
     call TDySetPorosityFunction(tdy,PorosityFunction,ierr);
     CHKERRA(ierr);
  end if

  if (bc_type == MPFAO_DIRICHLET_BC .OR. bc_type == MPFAO_SEEPAGE_BC ) then
     call TDySetBoundaryPressureFunction(tdy,PressureFunction,ierr);
     CHKERRA(ierr)
  endif

  if (use_tdydriver) then
     call TDyDriverInitializeTDy(tdy, ierr);
  else
     call TDySetup(tdy,ierr);
     CHKERRA(ierr);
  end if

  call TDyCreateVectors(tdy,ierr); CHKERRA(ierr)
  call TDyCreateJacobian(tdy,ierr); CHKERRA(ierr)

  ! Set initial condition
  call DMCreateGlobalVector(dm,U,ierr);
  CHKERRA(ierr);

  if (ic_file_flg) then
    call PetscViewerBinaryOpen(PETSC_COMM_WORLD, ic_filename, FILE_MODE_READ, viewer, ierr);
    CHKERRA(ierr)
    call VecLoad(U, viewer, ierr);
    CHKERRA(ierr)
    call PetscViewerDestroy(viewer, ierr);
    CHKERRA(ierr)
  else
    call VecSet(U,ic_value,ierr);
    CHKERRA(ierr);
  endif

  call TDySetInitialCondition(tdy,U,ierr);
  CHKERRA(ierr);

  ! Initialize the diagnostics DM and Vec, fetch the section that holds the
  ! layout, and determine which diagnostic index corresponds to the liquid
  ! mass (to check mass conservation).
  call TDyCreateDiagnostics(tdy, diags_dm, ierr)
  CHKERRA(ierr)
  call DMCreateGlobalVector(diags_dm, diags, ierr)
  CHKERRA(ierr)
  call DMGetLocalSection(diags_dm, diags_section, ierr)
  CHKERRA(ierr)
  call PetscSectionGetNumFields(diags_section, n_diags, ierr)
  do i_diag = 1,n_diags
    call PetscSectionGetFieldName(diags_section, i_diag, diag_name, ierr)
    if (diag_name == "liquid_mass") then
      i_liquid_mass = i_diag
    end if
  end do

  ! Set up the SNES solver.
  call TDySetPreviousSolutionForSNESSolver(tdy, U, ierr)
  CHKERRA(ierr);

  call SNESCreate(PETSC_COMM_WORLD,snes,ierr);
  CHKERRA(ierr);

  call TDySetSNESFunction(snes,tdy,ierr);
  CHKERRA(ierr);

  call TDySetSNESJacobian(snes,tdy,ierr);
  CHKERRA(ierr);

  call SNESSetFromOptions(snes,ierr);
  CHKERRA(ierr);

  call TDySetDtimeForSNESSolver(tdy,dtime,ierr);
  CHKERRA(ierr);

  do step = 1,max_steps

    call TDyPreSolveSNESSolver(tdy,ierr);
    CHKERRA(ierr);

    ! Check the total liquid mass before the step. Because the liquid mass is
    ! nonnegative, the sum over all cells is the same as the 1-norm.
    call TDyComputeDiagnostics(tdy, diags_dm, diags, ierr)
    CHKERRA(ierr)
    call VecStrideNorm(diags, i_liquid_mass, NORM_1, mass_pre, ierr)
    CHKERRA(ierr)
    !call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
    !CHKERRA(ierr);
    !mass_pre = 0.d0
    !do g = 1,nvalues
    !mass_pre = mass_pre + liquid_mass(g)
    !enddo

    if (use_tdydriver) then
       call TDyTimeIntegratorSetTimeStep(tdy,1800.d0, ierr);
       CHKERRA(ierr);

       call TDyTimeIntegratorRunToTime(tdy,1800.d0 * step, ierr);
       CHKERRA(ierr);

    else
       call SNESSolve(snes,PETSC_NULL_VEC,U,ierr);
       CHKERRA(ierr);

       call SNESGetConvergedReason(snes,reason,ierr)
       CHKERRA(ierr)
       if (reason<0) then
          call PetscError(PETSC_COMM_WORLD, 0, PETSC_ERR_USER, "SNES did not converge");
       endif
    endif

    call TDyPostSolveSNESSolver(tdy,U,ierr);
    CHKERRA(ierr);

    ! Check the total liquid mass after the step. Because the liquid mass is
    ! nonnegative, the sum over all cells is the same as the 1-norm.
    call TDyComputeDiagnostics(tdy, diags_dm, diags, ierr)
    CHKERRA(ierr)
    call VecStrideNorm(diags, i_liquid_mass, NORM_1, mass_post, ierr)
    CHKERRA(ierr)
    !call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
    !CHKERRA(ierr);
    !mass_post = 0.d0
    !do g = 1,nvalues
    !mass_post = mass_post + liquid_mass(g)
    !enddo
    write(*,*)'Liquid mass pre,post,diff ',step,mass_pre,mass_post,mass_pre-mass_post

    step_mod = mod(step,48);
    write(string,*) step
    string = 'solution_' // trim(adjustl(string)) // '.bin'
    if (step_mod == 0) then
      write(*,*)'Writing output: ',trim(string)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD, trim(string), FILE_MODE_WRITE, viewer, ierr);
      CHKERRA(ierr)
      call VecView(U, viewer, ierr);
      CHKERRA(ierr)
      call PetscViewerDestroy(viewer, ierr);
      CHKERRA(ierr)
    endif

  end do

  !call TDyGetSaturationValuesLocal(tdy,nvalues,liquid_sat,ierr)
  !CHKERRA(ierr);

  if (use_tdydriver) then
     call TDyTimeIntegratorOutputRegression(tdy,ierr);
     CHKERRA(ierr);
  else
     call TDyOutputRegression(tdy,U,ierr);
     CHKERRA(ierr);
  end if

  ! Clean up diagnostic stuff
  call VecDestroy(diags, ierr)
  CHKERRA(ierr)
  call DMDestroy(diags_dm, ierr)
  CHKERRA(ierr)

  call TDyDestroy(tdy,ierr);
  CHKERRA(ierr);

  call TDyFinalize(ierr);
  CHKERRA(ierr);

  deallocate(index)

  call exit(successful_exit_code)

end program
