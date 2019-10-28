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
subroutine TDyCreate(a,b,z)
 use petscdm
 use tdycoredef
 DM a
 TDy b
 integer z
end subroutine TDyCreate
        end interface
        interface
subroutine TDySetDiscretizationMethod(a,b,z)
 use tdycoredef
 TDy a
 PetscInt b
 integer z
end subroutine TDySetDiscretizationMethod
        end interface
        interface
subroutine TDySetFromOptions(a,z)
 use tdycoredef
 TDy a
 integer z
end subroutine TDySetFromOptions
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

      Interface TDySetResidualSaturationValuesLocal
        subroutine TDySetResidualSaturationValuesLocal0(a,b,c,d,z)
          use tdycoredef
          TDy a ! Vec
          PetscInt b ! PetscInt
          PetscInt c (*) ! PetscInt
          PetscScalar d (*) ! PetscScalar
          integer z
        end subroutine
        subroutine TDySetResidualSaturationValuesLocal11(a,b,c,d,e,z)
          use tdycoredef
          TDy a ! TDy
          PetscInt b ! PetscInt
          PetscInt c ! PetscInt
          PetscScalar d ! PetscScalar
          integer z
        end subroutine
      end interface TDySetResidualSaturationValuesLocal

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

end module tdycore
