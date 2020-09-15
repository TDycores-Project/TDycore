#/bin/sh

mpirun -n 4 ./richards_driver -dim 3 -Nx 2 -Ny 2 -Nz 2 -tdy_water_density exponential -tdy_regression_test -tdy_regression_test_num_cells_per_process 1 -tdy_regression_test_filename richards_driver-prob1-np4 -final_time 3.1536e3 -dt_max 600. -dt_growth_factor 1.5 -time_integration_method SNES
