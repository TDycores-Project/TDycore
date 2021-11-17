module tdycoredefdummy
#include <../src/f90-mod/tdycore.h>
#include <petsc/finclude/petscts.h>
end module tdycoredefdummy

module tdycoredef
  use tdycoredefdummy
end module tdycoredef

module tdycore
  use tdycoredef

  interface
     subroutine TDyFinalize(ierr)
       integer ierr
     end subroutine TDyFinalize
  end interface
  interface
     subroutine TDyCreate(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyCreate
  end interface
  interface
     subroutine TDySetMode(a,b,z)
       use tdycoredef
       TDy a
       PetscInt b
       integer z
     end subroutine TDySetMode
  end interface
  interface
     subroutine TDySetDiscretization(a,b,z)
       use tdycoredef
       TDy a
       PetscInt b
       integer z
     end subroutine TDySetDiscretization
  end interface
  interface
     subroutine TDySetFromOptions(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDySetFromOptions
  end interface
  interface
     subroutine TDyDriverInitializeTDy(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyDriverInitializeTDy
  end interface
  interface
     subroutine TDyTimeIntegratorRunToTime(a,b,z)
       use tdycoredef
       TDy a
       PetscReal b
       integer z
     end subroutine TDyTimeIntegratorRunToTime
  end interface
  interface
     subroutine TDyTimeIntegratorSetTimeStep(a,b,z)
       use tdycoredef
       TDy a
       PetscReal b
       integer z
     end subroutine TDyTimeIntegratorSetTimeStep
  end interface
  interface
     subroutine TDyTimeIntegratorOutputRegression(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyTimeIntegratorOutputRegression
  end interface
  interface
     subroutine TDySetup(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDySetup
  end interface
  interface
     subroutine TDyComputeSystem(a,b,c,z)
       use tdycoredef
       use petscvec
       use petscmat
       TDy a
       Mat b
       Vec c
       integer z
     end subroutine TDyComputeSystem
  end interface

!  Interface TDyGetNumCellsLocal
!     subroutine TDyGetNumCellsLocal(a,b,z)
!       use tdycoredef
!       TDy a ! tdy
!       PetscInt b ! PetscInt
!       integer z
!     end subroutine TDyGetNumCellsLocal
!  end interface TDyGetNumCellsLocal
!
!  Interface TDyGetCellNaturalIDsLocal
!     subroutine TDyGetCellNaturalIDsLocal(a,b,c,z)
!       use tdycoredef
!       TDy a ! tdy
!       PetscInt b ! PetscInt
!       PetscInt c (*) ! PetscInt
!       integer z
!     end subroutine TDyGetCellNaturalIDsLocal
!  end interface TDyGetCellNaturalIDsLocal
!
!  Interface TDyGetCellIsLocal
!     subroutine TDyGetCellIsLocal(a,b,c,z)
!       use tdycoredef
!       TDy a
!       PetscInt b
!       PetscInt c (*)
!       integer z
!     end subroutine TDyGetCellIsLocal
!  end interface TDyGetCellIsLocal

  Interface TDyUpdateState
     subroutine TDyUpdateState(a,b,z)
       use tdycoredef
       TDy a ! tdy
       PetscScalar b (*) ! PetscScalar
       integer z
     end subroutine TDyUpdateState
  end interface TDyUpdateState

  interface
     subroutine TDySetIFunction(a,b,z)
       use tdycoredef
       use petscts
       TS a
       TDy b
       integer z
     end subroutine TDySetIFunction
  end interface

  interface
     subroutine TDySetIJacobian(a,b,z)
       use tdycoredef
       use petscts
       TS a
       TDy b
       integer z
     end subroutine TDySetIJacobian
  end interface

  interface
     subroutine TDySetSNESFunction(a,b,z)
       use tdycoredef
       use petscsnes
       SNES a
       TDy b
       integer z
     end subroutine TDySetSNESFunction
  end interface

  interface
     subroutine TDySetSNESJacobian(a,b,z)
       use tdycoredef
       use petscsnes
       SNES a
       TDy b
       integer z
     end subroutine TDySetSNESJacobian
  end interface

  interface
     subroutine TDyCreateVectors(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyCreateVectors
  end interface

  interface
     subroutine TDyCreateJacobian(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyCreateJacobian
  end interface

  interface
     subroutine TDySetDtimeForSNESSolver(a,b,z)
       use tdycoredef
       TDy a
       PetscReal b
       integer z
     end subroutine TDySetDtimeForSNESSolver
  end interface

  interface
     subroutine TDySetInitialCondition(a,b,z)
       use tdycoredef
       use petscvec
       TDy a
       Vec b
       integer z
     end subroutine TDySetInitialCondition
  end interface

  interface
     subroutine TDySetPreviousSolutionForSNESSolver(a,b,z)
       use tdycoredef
       use petscvec
       TDy a
       Vec b
       integer z
     end subroutine TDySetPreviousSolutionForSNESSolver
  end interface

  interface
     subroutine TDyPreSolveSNESSolver(a,z)
       use tdycoredef
       TDy a
       integer z
     end subroutine TDyPreSolveSNESSolver
  end interface

  interface
     subroutine TDyPostSolveSNESSolver(a,b,z)
       use tdycoredef
       use petscvec
       TDy a
       Vec b
       integer z
     end subroutine TDyPostSolveSNESSolver
  end interface

  !> A TDySpatialFunction is a function that computes values f on n points x,
  !> indicating any (non-zero) error status in err.
  abstract interface
    subroutine TDySpatialFunction(n, x, f, ierr)
      use tdycoredef
      PetscInt,                intent(in)  :: n
      PetscReal, dimension(:), intent(in)  :: x
      PetscReal, dimension(:), intent(out) :: f
      PetscErrorCode                       :: ierr
    end subroutine
  end interface

  ! We use GetRegFn to retrieve function pointers from the C registry.
  interface
    function GetRegFn(name, c_func) bind (c, name="TDyGetFunction") result(ierr)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value :: name
      type(c_funptr) :: c_func
      integer(c_int) :: ierr
    end function
  end interface

  abstract interface
    subroutine TDyDMConstructor(comm, dm, ierr)
      use, intrinsic :: iso_c_binding
      use tdycoredef
      use petscdm
      MPI_Comm           :: comm
      DM                 :: dm
      PetscErrorCode     :: ierr
    end subroutine
  end interface

  contains

  subroutine TDyInit(ierr)
#include <petsc/finclude/petscvec.h>
     use petscvec
     implicit none
     integer ierr

     call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
     CHKERRQ(ierr)
     call TDyInitNoArguments(ierr)
  end subroutine TDyInit

  subroutine TDyRegisterFunction(name, func, ierr)
    use, intrinsic :: iso_c_binding
    implicit none
    character(len=*), intent(in)   :: name
    procedure(TDySpatialFunction)  :: func
    PetscErrorCode                 :: ierr

    interface
      function RegisterFn(name, func) bind (c, name="TDyRegisterFunction") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value :: name
        type(c_funptr), value :: func
        integer(c_int) :: ierr
      end function
    end interface

    ierr = RegisterFn(FtoCString(name), c_funloc(func))
  end subroutine

  subroutine TDySetDMConstructor(tdy, dm_ctor, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                 :: tdy
    procedure(TDyDMConstructor) :: dm_ctor
    PetscErrorCode              :: ierr

    interface
      function SetDMConstructor(tdy, dm_ctor) bind (c, name="TDySetDMConstructorF90") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_funptr), value :: dm_ctor
        integer(c_int)        :: ierr
      end function
    end interface

    ierr = SetDMConstructor(c_loc(tdy), c_funloc(dm_ctor))
  end subroutine

  subroutine TDySelectPorosityFunction(tdy, name, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy :: tdy
    character(len=*), intent(in)   :: name
    PetscErrorCode                 :: ierr

    type(c_funptr)                         :: c_func
    procedure(TDySpatialFunction), pointer :: f_func

    ierr = GetRegFn(FtoCString(name), c_func)
    call c_f_procpointer(c_func, f_func)
    call TDySetPorosityFunction(tdy, f_func, ierr)
  end subroutine

  subroutine TDySelectForcingFunction(tdy, name, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy :: tdy
    character(len=*), intent(in)   :: name
    PetscErrorCode                 :: ierr

    type(c_funptr)                         :: c_func
    procedure(TDySpatialFunction), pointer :: f_func

    ierr = GetRegFn(FtoCString(name), c_func)
    call c_f_procpointer(c_func, f_func)
    call TDySetForcingFunction(tdy, f_func, ierr)
  end subroutine

  subroutine TDySetForcingFunction(tdy, f, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                   :: tdy
    procedure(TDySpatialFunction) :: f
    PetscErrorCode                :: ierr

    interface
      function Func(tdy, f) bind (c, name="TDySetForcingFunctionF90") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_funptr), value :: f
        integer(c_int)        :: ierr
      end function
    end interface

    ierr = Func(c_loc(tdy), c_funloc(f))
  end subroutine

! Uncomment this when we are ready to set energy forcing fns in Fortran.
!  subroutine TDySelectEnergyForcingFunction(tdy, name, ierr)
!    use, intrinsic :: iso_c_binding
!    use tdycoredef
!    implicit none
!    TDy :: tdy
!    character(len=*), intent(in)   :: name
!    PetscErrorCode                 :: ierr
!
!    type(c_funptr)                         :: c_func
!    procedure(TDySpatialFunction), pointer :: f_func
!
!    ierr = GetRegFn(FtoCString(name), c_func)
!    call c_f_procpointer(c_func, f_func)
!    call TDySetEnergyForcingFunction(tdy, f_func, ierr)
!  end subroutine

! Uncomment this when we are ready to set permeability functions programmatically.
!  subroutine TDySelectPermeabilityFunction(tdy, name, ierr)
!    use, intrinsic :: iso_c_binding
!    use tdycoredef
!    implicit none
!    TDy :: tdy
!    character(len=*), intent(in)   :: name
!    PetscErrorCode                 :: ierr
!
!    type(c_funptr)                         :: c_func
!    procedure(TDySpatialFunction), pointer :: f_func
!
!    ierr = GetRegFn(FtoCString(name), c_func)
!    call c_f_procpointer(c_func, f_func)
!    call TDySetPermeabilityFunction(tdy, f_func, ierr)
!  end subroutine

  subroutine TDySelectBoundaryPressureFunction(tdy, name, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy :: tdy
    character(len=*), intent(in)   :: name
    PetscErrorCode                 :: ierr

    type(c_funptr)                         :: c_func
    procedure(TDySpatialFunction), pointer :: f_func

    ierr = GetRegFn(FtoCString(name), c_func)
    call c_f_procpointer(c_func, f_func)
    call TDySetBoundaryPressureFunction(tdy, f_func, ierr)
  end subroutine

  subroutine TDySetBoundaryPressureFunction(tdy, f, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                   :: tdy
    procedure(TDySpatialFunction) :: f
    PetscErrorCode                :: ierr
    type(c_ptr)                   :: p_tdy

    interface
      function Func(tdy, f) bind (c, name="TDySetBoundaryPressureFunctionF90") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_funptr), value :: f
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, c_funloc(f))
  end subroutine

  subroutine TDySelectBoundaryVelocityFunction(tdy, name, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy :: tdy
    character(len=*), intent(in)   :: name
    PetscErrorCode                 :: ierr

    type(c_funptr)                         :: c_func
    procedure(TDySpatialFunction), pointer :: f_func

    ierr = GetRegFn(FtoCString(name), c_func)
    call c_f_procpointer(c_func, f_func)
    call TDySetBoundaryVelocityFunction(tdy, f_func, ierr)
  end subroutine

  subroutine TDySetBoundaryVelocityFunction(tdy, f, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                   :: tdy
    procedure(TDySpatialFunction) :: f
    PetscErrorCode                :: ierr
    type(c_ptr)                   :: p_tdy

    interface
      function Func(tdy, f) bind (c, name="TDySetBoundaryVelocityFunctionF90") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_funptr), value :: f
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, c_funloc(f))
  end subroutine

  subroutine TDySetConstantPorosity(tdy, val, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target    :: tdy
    PetscReal      :: val
    PetscErrorCode :: ierr
    type(c_ptr)    :: p_tdy

    interface
      function Func(tdy, val) bind (c, name="TDySetConstantPorosity") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        real(c_double), value :: val
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, val)
  end subroutine

  subroutine TDySetPorosityFunction(tdy, f, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                   :: tdy
    procedure(TDySpatialFunction) :: f
    PetscErrorCode                :: ierr
    type(c_ptr)                   :: p_tdy

    interface
      function Func(tdy, f) bind (c, name="TDySetPorosityFunctionF90") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_funptr), value :: f
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, c_funloc(f))
  end subroutine

  subroutine TDySetConstantTensorPermeability(tdy, val, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                 :: tdy
    PetscReal, target           :: val(9)
    PetscErrorCode              :: ierr
    type(c_ptr)                 :: p_tdy

    interface
      function Func(tdy, val) bind (c, name="TDySetConstantTensorPermeability") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_ptr), value    :: val
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, c_loc(val))
  end subroutine

  subroutine TDySetConstantResidualSaturation(tdy, val, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target    :: tdy
    PetscReal      :: val
    PetscErrorCode :: ierr
    type(c_ptr)    :: p_tdy

    interface
      function Func(tdy, val) bind (c, name="TDySetConstantResidualSaturation") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        real(c_double)        :: val
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, val)
  end subroutine

  subroutine TDySetResidualSaturationFunction(tdy, f, ierr)
    use, intrinsic :: iso_c_binding
    use tdycoredef
    implicit none
    TDy, target                   :: tdy
    procedure(TDySpatialFunction) :: f
    PetscErrorCode                :: ierr
    type(c_ptr)                   :: p_tdy

    interface
      function Func(tdy, f) bind (c, name="TDySetResidualSaturationFunctionF90") result(ierr)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), value    :: tdy
        type(c_funptr), value :: f
        integer(c_int)        :: ierr
      end function
    end interface

    p_tdy = transfer(tdy%v, p_tdy)
    ierr = Func(p_tdy, c_funloc(f))
  end subroutine

  ! Here's a function that converts a Fortran string to a C string and
  ! stashes it in TDycore's Fortran string registry. This allows us to
  ! create more expressive (and standard) Fortran interfaces.
  function FtoCString(f_string) result(c_string)
     use, intrinsic :: iso_c_binding
     implicit none
     character(len=*), target :: f_string
     character(len=:), pointer :: f_ptr
     type(c_ptr) :: c_string

     interface
       function NewCString(f_str_ptr, f_str_len) bind (c, name="NewCString") result(c_string)
         use, intrinsic :: iso_c_binding
         type(c_ptr), value :: f_str_ptr
         integer(c_int), value :: f_str_len
         type(c_ptr) :: c_string
       end function NewCString
     end interface

     f_ptr => f_string
     c_string = NewCString(c_loc(f_ptr), len(f_string))
   end function FtoCString

end module tdycore
