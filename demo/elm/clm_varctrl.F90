module clm_varctl
!
implicit none
  character(len=*), private, parameter :: mod_filename = &
       __FILE__
  integer, public :: iulog = 6        ! "stdout" log file unit number, default is 6
end module clm_varctl
