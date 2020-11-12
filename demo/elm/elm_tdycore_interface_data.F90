module elm_tdycore_interface_data

#include "petsc/finclude/petscsys.h"
#include "petsc/finclude/petscvec.h"

  use petscsys
  use petscvec

  implicit none

  private

  type, public :: elm_tdycore_idata_type

     ! Time invariant data:

     ! (i) Soil properties -
     ! Seq. vectors
     Vec :: hksat_x_elm_svec
     Vec :: hksat_y_elm_svec
     Vec :: hksat_z_elm_svec
     Vec :: sucsat_elm_svec
     Vec :: watsat_elm_svec
     Vec :: bsw_elm_svec
     Vec :: press_elm_svec
     Vec :: thetares_elm_svec

     ! Local for ELM  - MPI vectors
     Vec :: hksat_x_elm_mvec
     Vec :: hksat_y_elm_mvec
     Vec :: hksat_z_elm_mvec
     Vec :: sucsat_elm_mvec
     Vec :: watsat_elm_mvec
     Vec :: bsw_elm_mvec
     Vec :: press_elm_mvec
     Vec :: thetares_elm_mvec

     ! Local for TDycore - Seq. vec
     Vec :: hksat_x_tdycore_svec
     Vec :: hksat_y_tdycore_svec
     Vec :: hksat_z_tdycore_svec
     Vec :: sucsat_tdycore_svec
     Vec :: watsat_tdycore_svec
     Vec :: bsw_tdycore_svec
     Vec :: press_tdycore_svec
     Vec :: thetares_tdycore_svec

     ! MPI vectors
     Vec :: hksat_x_tdycore_mvec
     Vec :: hksat_y_tdycore_mvec
     Vec :: hksat_z_tdycore_mvec
     Vec :: sucsat_tdycore_mvec
     Vec :: watsat_tdycore_mvec
     Vec :: bsw_tdycore_mvec
     Vec :: thetares_tdycore_mvec

     ! (ii) Mesh property

     ! Area of top face
     Vec :: area_top_face_elm_svec      ! seq vec
     Vec :: area_top_face_tdycore_mvec  ! mpi vec

     ! Time variant data

     ! (i) Sink/Source of water for TDycore's 3D subsurface domain
     Vec :: qflx_elm_mvec     ! mpi vec
     Vec :: qflx_tdycore_svec ! seq vec

     ! (ii) Source of water and temperature of rain for TDycore's 2D surface domain
     Vec :: rain_elm_mvec          ! mpi vec
     Vec :: rain_tdycore_svec      ! seq vec

     ! (iii) Saturation and mass
     Vec :: sat_elm_svec      ! seq vec
     Vec :: sat_tdycore_mvec  ! mpi vec
     Vec :: mass_elm_svec     ! seq vec
     Vec :: mass_tdycore_mvec ! mpi vec

     ! (vi) Ice saturation
     Vec :: sat_ice_elm_svec     ! seq vec
     Vec :: sat_ice_tdycore_mvec ! mpi vec

     ! Number of cells for the 3D subsurface domain
     PetscInt :: nlelm_sub ! num of local clm cells
     PetscInt :: ngelm_sub ! num of ghosted clm cells (ghosted = local+ghosts)
     PetscInt :: nltdy_sub ! num of local pflotran cells
     PetscInt :: ngtdy_sub ! num of ghosted pflotran cells (ghosted = local+ghosts)

     ! Number of cells for the surface of the 3D subsurface domain
     PetscInt :: nlelm_2dsub ! num of local clm cells
     PetscInt :: ngelm_2dsub ! num of ghosted clm cells (ghosted = local+ghosts)
     PetscInt :: nltdy_2dsub ! num of local pflotran cells
     PetscInt :: ngtdy_2dsub ! num of ghosted pflotran cells (ghosted = local+ghosts)

     ! Number of cells for the 2D surface domain
     PetscInt :: nlelm_srf ! num of local clm cells
     PetscInt :: ngelm_srf ! num of ghosted clm cells (ghosted = local+ghosts)
     PetscInt :: nltdy_srf ! num of local pflotran cells
     PetscInt :: ngtdy_srf ! num of ghosted pflotran cells (ghosted = local+ghosts)

     PetscInt :: nzelm_mapped ! num of ELM soil layers that are mapped

  end type elm_tdycore_idata_type

  type(elm_tdycore_idata_type) , public, target , save :: elm_tdycore_idata

  public :: &
       ELMTDycore_InterfaceData_Init, &
       ELMTDycore_InterfaceData_CreateVectors, &
       ELMTDycore_InterfaceData_Deallocate

contains

  ! ************************************************************************** !

  subroutine ELMTDycore_InterfaceData_Init()
    ! 
    ! This routine initialized the data transfer type.
    !

    implicit none

    elm_tdycore_idata%nlelm_sub   = 0
    elm_tdycore_idata%ngelm_sub   = 0
    elm_tdycore_idata%nltdy_sub   = 0
    elm_tdycore_idata%ngtdy_sub   = 0

    elm_tdycore_idata%nlelm_2dsub = 0
    elm_tdycore_idata%ngelm_2dsub = 0
    elm_tdycore_idata%nltdy_2dsub = 0
    elm_tdycore_idata%ngtdy_2dsub = 0

    elm_tdycore_idata%nlelm_srf   = 0
    elm_tdycore_idata%ngelm_srf   = 0
    elm_tdycore_idata%nltdy_srf   = 0
    elm_tdycore_idata%ngtdy_srf   = 0

    elm_tdycore_idata%hksat_x_elm_svec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_y_elm_svec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_z_elm_svec  = PETSC_NULL_VEC
    elm_tdycore_idata%sucsat_elm_svec   = PETSC_NULL_VEC
    elm_tdycore_idata%watsat_elm_svec   = PETSC_NULL_VEC
    elm_tdycore_idata%bsw_elm_svec      = PETSC_NULL_VEC
    elm_tdycore_idata%press_elm_svec    = PETSC_NULL_VEC
    elm_tdycore_idata%thetares_elm_svec = PETSC_NULL_VEC

    elm_tdycore_idata%hksat_x_elm_mvec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_y_elm_mvec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_z_elm_mvec  = PETSC_NULL_VEC
    elm_tdycore_idata%sucsat_elm_mvec   = PETSC_NULL_VEC
    elm_tdycore_idata%watsat_elm_mvec   = PETSC_NULL_VEC
    elm_tdycore_idata%bsw_elm_mvec      = PETSC_NULL_VEC
    elm_tdycore_idata%press_elm_mvec    = PETSC_NULL_VEC
    elm_tdycore_idata%thetares_elm_mvec = PETSC_NULL_VEC

    elm_tdycore_idata%hksat_x_tdycore_svec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_y_tdycore_svec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_z_tdycore_svec  = PETSC_NULL_VEC
    elm_tdycore_idata%sucsat_tdycore_mvec   = PETSC_NULL_VEC
    elm_tdycore_idata%watsat_tdycore_mvec   = PETSC_NULL_VEC
    elm_tdycore_idata%bsw_tdycore_svec      = PETSC_NULL_VEC
    elm_tdycore_idata%press_tdycore_svec    = PETSC_NULL_VEC
    elm_tdycore_idata%thetares_tdycore_svec = PETSC_NULL_VEC

    elm_tdycore_idata%hksat_x_tdycore_mvec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_y_tdycore_mvec  = PETSC_NULL_VEC
    elm_tdycore_idata%hksat_z_tdycore_mvec  = PETSC_NULL_VEC
    elm_tdycore_idata%sucsat_tdycore_mvec   = PETSC_NULL_VEC
    elm_tdycore_idata%watsat_tdycore_mvec   = PETSC_NULL_VEC
    elm_tdycore_idata%bsw_tdycore_mvec      = PETSC_NULL_VEC
    elm_tdycore_idata%thetares_tdycore_mvec = PETSC_NULL_VEC

    elm_tdycore_idata%qflx_elm_mvec        = PETSC_NULL_VEC
    elm_tdycore_idata%qflx_tdycore_svec    = PETSC_NULL_VEC

    elm_tdycore_idata%rain_elm_mvec        = PETSC_NULL_VEC
    elm_tdycore_idata%rain_tdycore_svec    = PETSC_NULL_VEC

    elm_tdycore_idata%sat_elm_svec         = PETSC_NULL_VEC
    elm_tdycore_idata%sat_tdycore_mvec     = PETSC_NULL_VEC
    elm_tdycore_idata%mass_elm_svec        = PETSC_NULL_VEC
    elm_tdycore_idata%mass_tdycore_mvec    = PETSC_NULL_VEC

    elm_tdycore_idata%sat_ice_elm_svec     = PETSC_NULL_VEC
    elm_tdycore_idata%sat_ice_tdycore_mvec = PETSC_NULL_VEC

    elm_tdycore_idata%nzelm_mapped = 0

  end subroutine ELMTDycore_InterfaceData_Init

  ! ************************************************************************** !

  subroutine ELMTDycore_InterfaceData_CreateVectors(mycomm)
    ! 
    ! This routine creates PETSc vectors required for data transfer between
    ! ELM and TDycore.
    !
    implicit none

    PetscErrorCode :: ierr
    PetscMPIInt    :: mycomm, rank

    call MPI_Comm_rank(mycomm,rank, ierr)

    !
    ! For data transfer from ELM to TDycore
    !

    ! Create MPI Vectors for ELM
    call VecCreateMPI(mycomm,elm_tdycore_idata%nlelm_sub,PETSC_DECIDE, &
         elm_tdycore_idata%hksat_x_elm_mvec,ierr)
    call VecSet(elm_tdycore_idata%hksat_x_elm_mvec,0.d0,ierr)

    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%hksat_y_elm_mvec ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%hksat_z_elm_mvec ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%sucsat_elm_mvec  ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%watsat_elm_mvec  ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%bsw_elm_mvec     ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%press_elm_mvec   ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%thetares_elm_mvec,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_elm_mvec,elm_tdycore_idata%qflx_elm_mvec    ,ierr)

    call VecCreateMPI(mycomm,elm_tdycore_idata%nlelm_srf,PETSC_DECIDE,elm_tdycore_idata%rain_elm_mvec,ierr)

    ! Create Seq. Vectors for TDycore
    call VecCreateSeq(PETSC_COMM_SELF,elm_tdycore_idata%ngtdy_sub,&
         elm_tdycore_idata%hksat_x_tdycore_svec,ierr)
    call VecSet(elm_tdycore_idata%hksat_x_tdycore_svec,0.d0,ierr)

    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%hksat_y_tdycore_svec ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%hksat_z_tdycore_svec ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%sucsat_tdycore_svec  ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%watsat_tdycore_svec  ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%bsw_tdycore_svec     ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%press_tdycore_svec   ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%thetares_tdycore_svec   ,ierr)
    call VecDuplicate(elm_tdycore_idata%hksat_x_tdycore_svec,elm_tdycore_idata%qflx_tdycore_svec    ,ierr)

    call VecCreateSeq(PETSC_COMM_SELF,elm_tdycore_idata%ngtdy_srf,elm_tdycore_idata%rain_tdycore_svec,ierr)
    call VecSet(elm_tdycore_idata%rain_tdycore_svec,0.d0,ierr)

    !
    ! For data transfer from TDycore to ELM
    !

    ! Create MPI Vectors for TDycore
    ! 3D Subsurface TDycore ---to--- 3D Subsurface ELM
    call VecCreateMPI(mycomm,elm_tdycore_idata%nltdy_sub,PETSC_DECIDE,elm_tdycore_idata%sat_tdycore_mvec,ierr)
    call VecSet(elm_tdycore_idata%sat_tdycore_mvec,0.d0,ierr)

    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%mass_tdycore_mvec          ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%sat_ice_tdycore_mvec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%area_top_face_tdycore_mvec ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%hksat_x_tdycore_mvec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%hksat_y_tdycore_mvec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%hksat_z_tdycore_mvec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%sucsat_tdycore_mvec        ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%watsat_tdycore_mvec        ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%bsw_tdycore_mvec           ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_tdycore_mvec,elm_tdycore_idata%thetares_tdycore_mvec      ,ierr)

    ! Create Seq. Vectors for ELM
    ! 3D Subsurface TDycore ---to--- 3D Subsurface ELM
    call VecCreateSeq(PETSC_COMM_SELF,elm_tdycore_idata%ngelm_sub,elm_tdycore_idata%sat_elm_svec,ierr)
    call VecSet(elm_tdycore_idata%sat_elm_svec,0.d0,ierr)

    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%mass_elm_svec          ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%sat_ice_elm_svec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%area_top_face_elm_svec ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%hksat_x_elm_svec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%hksat_y_elm_svec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%hksat_z_elm_svec       ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%sucsat_elm_svec        ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%watsat_elm_svec        ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%bsw_elm_svec           ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%press_elm_svec         ,ierr)
    call VecDuplicate(elm_tdycore_idata%sat_elm_svec,elm_tdycore_idata%thetares_elm_svec      ,ierr)

  end subroutine ELMTDycore_InterfaceData_CreateVectors

  ! ************************************************************************** !

  subroutine ELMTDycore_InterfaceData_Deallocate()
    ! 
    ! This routine destroys PETSc vectors that were created for data transfer.
    ! 

    implicit none

    PetscErrorCode :: ierr

    if (elm_tdycore_idata%hksat_x_elm_svec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_x_elm_svec,ierr)

    if (elm_tdycore_idata%hksat_y_elm_svec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_y_elm_svec,ierr)

    if (elm_tdycore_idata%hksat_z_elm_svec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_z_elm_svec,ierr)

    if (elm_tdycore_idata%sucsat_elm_svec   /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sucsat_elm_svec,ierr)

    if (elm_tdycore_idata%watsat_elm_svec   /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%watsat_elm_svec,ierr)

    if (elm_tdycore_idata%thetares_elm_svec /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%thetares_elm_svec,ierr)

    if (elm_tdycore_idata%bsw_elm_svec      /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%bsw_elm_svec,ierr)

    if (elm_tdycore_idata%hksat_x_elm_mvec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_x_elm_mvec,ierr)

    if (elm_tdycore_idata%hksat_y_elm_mvec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_y_elm_mvec,ierr)

    if (elm_tdycore_idata%hksat_z_elm_mvec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_z_elm_mvec,ierr)

    if (elm_tdycore_idata%sucsat_elm_mvec   /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sucsat_elm_mvec,ierr)

    if (elm_tdycore_idata%watsat_elm_mvec   /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%watsat_elm_mvec,ierr)

    if (elm_tdycore_idata%bsw_elm_mvec      /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%bsw_elm_mvec,ierr)

    if (elm_tdycore_idata%thetares_elm_mvec /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%thetares_elm_mvec,ierr)

    if (elm_tdycore_idata%press_elm_mvec    /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%press_elm_mvec,ierr)

    if (elm_tdycore_idata%press_elm_svec    /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%press_elm_svec,ierr)

    if (elm_tdycore_idata%hksat_x_tdycore_svec        /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_x_tdycore_svec,ierr)

    if (elm_tdycore_idata%hksat_y_tdycore_svec        /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_y_tdycore_svec,ierr)

    if (elm_tdycore_idata%hksat_z_tdycore_svec        /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_z_tdycore_svec,ierr)

    if (elm_tdycore_idata%sucsat_tdycore_mvec         /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sucsat_tdycore_mvec,ierr)

    if (elm_tdycore_idata%watsat_tdycore_mvec         /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%watsat_tdycore_mvec,ierr)

    if (elm_tdycore_idata%bsw_tdycore_svec            /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%bsw_tdycore_svec,ierr)

    if (elm_tdycore_idata%hksat_x_tdycore_mvec       /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_x_tdycore_mvec,ierr)

    if (elm_tdycore_idata%hksat_y_tdycore_mvec       /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_y_tdycore_mvec,ierr)

    if (elm_tdycore_idata%hksat_z_tdycore_mvec       /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%hksat_z_tdycore_mvec,ierr)

    if (elm_tdycore_idata%sucsat_tdycore_mvec        /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sucsat_tdycore_mvec,ierr)

    if (elm_tdycore_idata%watsat_tdycore_mvec        /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%watsat_tdycore_mvec,ierr)

    if (elm_tdycore_idata%bsw_tdycore_mvec           /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%bsw_tdycore_mvec,ierr)

    if (elm_tdycore_idata%thetares_tdycore_mvec      /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%thetares_tdycore_mvec,ierr)

    if (elm_tdycore_idata%press_tdycore_svec          /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%press_tdycore_svec,ierr)

    if (elm_tdycore_idata%thetares_tdycore_svec          /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%thetares_tdycore_svec,ierr)

    if (elm_tdycore_idata%qflx_elm_mvec          /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%qflx_elm_mvec,ierr)

    if (elm_tdycore_idata%qflx_tdycore_svec           /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%qflx_tdycore_svec,ierr)

    if (elm_tdycore_idata%rain_elm_mvec          /= PETSC_NULL_VEC) call &
         VecDestroy(elm_tdycore_idata%rain_elm_mvec,ierr)

    if (elm_tdycore_idata%rain_tdycore_svec           /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%rain_tdycore_svec,ierr)

    if (elm_tdycore_idata%sat_elm_svec           /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sat_elm_svec,ierr)

    if (elm_tdycore_idata%sat_tdycore_mvec            /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sat_tdycore_mvec,ierr)

    if (elm_tdycore_idata%mass_elm_svec          /= PETSC_NULL_VEC) call &
         VecDestroy(elm_tdycore_idata%mass_elm_svec,ierr)

    if (elm_tdycore_idata%mass_tdycore_mvec           /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%mass_tdycore_mvec,ierr)

    if (elm_tdycore_idata%sat_ice_elm_svec       /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sat_ice_elm_svec,ierr)

    if (elm_tdycore_idata%sat_ice_tdycore_mvec        /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%sat_ice_tdycore_mvec,ierr)

    if (elm_tdycore_idata%area_top_face_elm_svec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%area_top_face_elm_svec,ierr)

    if (elm_tdycore_idata%area_top_face_tdycore_mvec  /= PETSC_NULL_VEC) &
         call VecDestroy(elm_tdycore_idata%area_top_face_tdycore_mvec,ierr)

  end subroutine ELMTDycore_InterfaceData_Deallocate

end module elm_tdycore_interface_data
