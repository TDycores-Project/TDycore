#! /bin/sh
  
export TDYCORE=../richards/richards_driver
export TDYPROF=../../tools/tdyprof
export NUM_CELLS_IN_ONE_DIRECTION=16

$TDYCORE \
-dim 3 \
-N $NUM_CELLS_IN_ONE_DIRECTION \
-tdy_mpfao_gmatrix_method          MPFAO_GMATRIX_DEFAULT \
-tdy_mpfao_boundary_condition_type MPFAO_NEUMANN_BC \
-tdy_water_density                 EXPONENTIAL \
-final_time 1. \
-dt_max 1. \
-dt_growth_factor 2. \
-tdy_timers \
-snes_monitor -snes_converged_reason \
-snes_rtol 1e-08 \
-snes_linesearch_type basic \
-ksp_monitor -ksp_type bcgs -pc_type ilu -log_view -tdy_timers \
| tee tdycore.stdout

$TDYPROF tdycore_profile.csv top15 | tee tdyprof.stdout
