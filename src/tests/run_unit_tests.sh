#!/usr/bin/env bash

# This runs unit tests using a one-test-per-process protocol.

count=`$1 count`
nproc=`$1 nproc`
i=0
while [ $i -lt $count ]
do
  # Run the test with the appropriate number of MPI processes.
  mpiexec -np $nproc -quiet $1 $i
  status=$?
  if test $status -ne 0
  then
    echo "Unit test $1 FAILED with status $status."
    exit $status
  fi
  ((i++))
done
