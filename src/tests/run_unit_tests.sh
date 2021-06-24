#!/usr/bin/env bash

# This runs unit tests using a one-test-per-process protocol.
num_failures=0
for arg in "$@"
do
  echo "Running $arg"
  count=`$arg count`
  nproc=`$arg nproc`
  i=0
  while [ $i -lt $count ]
  do
    # Run the test with the appropriate number of MPI processes.
    $MPIEXEC -n $nproc $arg $i
    status=$?
    if test $status -ne 0
    then
      echo "Unit test $arg FAILED with status $status."
      echo "You can run this test with the following command:"
      echo "$MPIEXEC -np $nproc $arg $i"
      ((num_failures++))
    fi
    ((i++))
  done
done
exit $num_failures
