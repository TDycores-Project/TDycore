module tdycoredefdummy
#include <../src/f90-mod/tdycore.h>
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
end module tdycore
