module snes_mpfaof90mod

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

  subroutine ResidualSaturation(resSat)
    implicit none
    PetscReal resSat

    resSat = 0.115d0
  end subroutine ResidualSaturation

  subroutine PorosityFunctionPFLOTRAN(tdy,x,theta,dummy,ierr)
    implicit none
    TDy                    :: tdy
    PetscReal, intent(in) :: x
    PetscReal, intent(out):: theta
    integer                :: dummy(*)
    PetscErrorCode :: ierr

    theta = 0.3d0
    ierr  = 0
  end subroutine PorosityFunctionPFLOTRAN

  subroutine Permeability_PFLOTRAN(K)
    implicit none
    PetscReal, intent(out) :: K(9)
    K(1) = 1.0d-12; K(2) = 0.0    ; K(3) = 0.0    ;
    K(4) = 0.0    ; K(5) = 1.0d-12; K(6) = 0.0    ;
    K(7) = 0.0    ; K(8) = 0.0    ; K(9) = 5.0d-13;
  end subroutine Permeability_PFLOTRAN

  subroutine PermeabilityFunctionPFLOTRAN(tdy,x,K,dummy,ierr)
    implicit none
    TDy                    :: tdy
    PetscReal, intent(in)  :: x(2)
    PetscReal, intent(out) :: K(9)
    integer                :: dummy(*)
    PetscErrorCode         :: ierr

    call Permeability_PFLOTRAN(K)

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

  subroutine PressureFunction(tdy,x,pressure,dummy,ierr)
    implicit none
    TDy                    :: tdy
    PetscReal, intent(in) :: x(3)
    PetscReal, intent(out):: pressure
    integer                :: dummy(*)
    PetscErrorCode :: ierr

    if (x(1) > 0.d0 .and. x(1) < 1.d0 .and. x(2) > 0.d0 .and. x(2) < 1.d0 .and. x(3) > 0.99d0) then
      pressure = 110000.d0
    else
      pressure = 100000.d0
    end if
    ierr = 0
  end subroutine PressureFunction

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
  DM                  :: dm, edm, dmDist
  Vec                 :: U
  !TS                 :: ts
  SNES                :: snes
  PetscInt            :: rank, successful_exit_code
  PetscBool           :: flg
  PetscInt            :: dim, faces(3)
  PetscReal           :: lower(3), upper(3)
  PetscErrorCode      :: ierr
  PetscInt            :: nx, ny, nz, ncell, bc_type, dm_plex_extrude_layers
  PetscInt  , pointer :: index(:)
  PetscReal , pointer :: residualSat(:), blockPerm(:), liquid_sat(:), liquid_mass(:)
  PetscReal , pointer :: alpha(:), m(:)
  PetscReal           :: perm(9), resSat
  PetscInt            :: c, cStart, cEnd, j, nvalues,g, max_steps, step
  PetscReal           :: dtime, mass_pre, mass_post, ic_value
  character (len=256) :: mesh_filename, ic_filename
  character(len=256)  :: string, bc_type_name
  PetscBool           :: mesh_file_flg, ic_file_flg, pflotran_consistent, use_tdydriver
  PetscViewer         :: viewer
  PetscInt            :: step_mod
  PetscFE             :: fe
  SNESConvergedReason :: reason

  call TDyInit(ierr);
  CHKERRA(ierr);

  ! Register some functions.
  call TDyRegisterPressureFn("p0", PressureFunction, ierr)
  CHKERRA(ierr);

  call TDyCreate(tdy, ierr);
  CHKERRA(ierr);
  call TDySetDiscretizationMethod(tdy,MPFA_O,ierr);
  CHKERRA(ierr);

  nx = 1; ny = 1; nz = 15;
  dim = 3
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
      CHKERRQ(ierr);
      call DMDestroy(dm ,ierr);
      dm = edm
    end if
  endif

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


  call TDySetWaterDensityType(tdy,WATER_DENSITY_EXPONENTIAL,ierr);
  CHKERRA(ierr)

  call DMPlexGetHeightStratum(dm,0,cStart,cEnd,ierr);
  CHKERRA(ierr);

  ncell = (cEnd-cStart)
  allocate(blockPerm   (ncell*dim*dim));
  allocate(residualSat (ncell))
  allocate(liquid_mass (ncell))
  allocate(liquid_sat  (ncell))
  allocate(alpha       (ncell))
  allocate(m           (ncell))
  allocate(index       (ncell))

  if (pflotran_consistent) then
     call Permeability_PFLOTRAN(perm);
     call ResidualSat_PFLOTRAN(resSat)
  else
     call Permeability(perm);
     call ResidualSaturation(resSat)
  end if

  do c = 1,ncell
    index(c) = c-1;
    residualSat(c) = resSat
    do j = 1,dim*dim
      blockPerm((c-1)*dim*dim+j) = perm(j)
    enddo
  enddo

  call TDySetDM(tdy, dm, ierr);
  CHKERRA(ierr);
  call TDySetFromOptions(tdy,ierr);
  CHKERRA(ierr);

  if (pflotran_consistent) then
     call TDySetPorosityFunction(tdy,PorosityFunctionPFLOTRAN,0,ierr);
     CHKERRA(ierr);

     do c = 1,ncell
        index(c) = c-1;
        call MaterialPropAlpha_PFLOTRAN(alpha(c))
        call MaterialPropM_PFLOTRAN(m(c))
     enddo

     call TDySetCharacteristicCurveAlphaValuesLocal(tdy,ncell,index,alpha,ierr)
     CHKERRA(ierr);

     call TDySetCharacteristicCurveMValuesLocal(tdy,ncell,index,m,ierr)
     CHKERRA(ierr);

     call TDySetBlockPermeabilityValuesLocal(tdy,ncell,index,blockPerm,ierr);
     CHKERRA(ierr);

     call TDySetResidualSaturationValuesLocal(tdy,ncell,index,residualSat,ierr);
     CHKERRA(ierr);

  else

     call TDySetPorosityFunction(tdy,PorosityFunction,0,ierr);
     CHKERRA(ierr);

     call TDySetBlockPermeabilityValuesLocal(tdy,ncell,index,blockPerm,ierr);
     CHKERRA(ierr);

     call TDySetResidualSaturationValuesLocal(tdy,ncell,index,residualSat,ierr);
     CHKERRA(ierr);

  end if

  if (bc_type == MPFAO_DIRICHLET_BC .OR. bc_type == MPFAO_SEEPAGE_BC ) then
!     call TDySetBoundaryPressureFn(tdy,PressureFunction,0,ierr);
     call TDySelectBoundaryPressureFn(tdy,"p0",ierr);
     CHKERRQ(ierr)
  endif

  if (use_tdydriver) then
     call TDyDriverInitializeTDy(tdy, ierr);
  else
     call TDySetupNumericalMethods(tdy,ierr);
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

    call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
    CHKERRA(ierr);
    mass_pre = 0.d0
    do g = 1,nvalues
    mass_pre = mass_pre + liquid_mass(g)
    enddo

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
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"SNES did not converge");
       endif
    endif

    call TDyPostSolveSNESSolver(tdy,U,ierr);
    CHKERRA(ierr);

    call TDyGetLiquidMassValuesLocal(tdy,nvalues,liquid_mass,ierr)
    CHKERRA(ierr);
    mass_post = 0.d0
    do g = 1,nvalues
    mass_post = mass_post + liquid_mass(g)
    enddo
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

  call TDyGetSaturationValuesLocal(tdy,nvalues,liquid_sat,ierr)
  CHKERRA(ierr);

  if (use_tdydriver) then
     call TDyTimeIntegratorOutputRegression(tdy,ierr);
     CHKERRA(ierr);
  else
     call TDyOutputRegression(tdy,U,ierr);
     CHKERRA(ierr);
  end if

  call TDyDestroy(tdy,ierr);
  CHKERRA(ierr);

  call TDyFinalize(ierr);
  CHKERRA(ierr);

  deallocate(blockPerm)
  deallocate(residualSat)
  deallocate(index)

  call exit(successful_exit_code)

end program
