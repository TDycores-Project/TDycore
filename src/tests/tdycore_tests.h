#ifndef TDYCORE_TESTS_H
#define TDYCORE_TESTS_H

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmocka.h>


// Runs tests with selected index (or reports the number of available tests).
static int _run_selected_tests(const char* command,
                               int num_tests,
                               const struct CMUnitTest tests[num_tests],
                               int nproc) {
  if (strcasecmp(command, "count") == 0) { // asked for # of tests
    fprintf(stdout, "%d\n", num_tests);
    return 0;
  } else if (strcasecmp(command, "nproc") == 0) { // asked for # of procs
    fprintf(stdout, "%d\n", nproc);
    return 0;
  } else { // asked for a specific test index
    // Try to interpret the argument as an index for the desired test.
    char *endptr;
    long index = strtol(command, &endptr, 10);
    if (*endptr == '\0') { // got a valid index!
      if ((index < 0) || (index >= num_tests)) {
        fprintf(stderr, "Invalid test index: %ld (must be in [0, %d])\n",
            index, num_tests);
        return 1;
      } else {
        const struct CMUnitTest selected_tests[] = { tests[index] };
        return cmocka_run_group_tests(selected_tests, NULL, NULL);
      }
    } else {
      fprintf(stderr, "Invalid command: %s (must be 'nproc', 'count', or index)\n",
          command);
      return 1;
    }
  }
}

// Call this macro instead of cmocka_run_group_tests, with the number of MPI
// processes as the last argument.
#define run_selected_tests(argc, argv, tests, nproc) \
  (argc > 1) ? _run_selected_tests(argv[1], \
                                   sizeof(tests) / sizeof((tests)[0]), \
                                   tests, nproc) \
             : cmocka_run_group_tests(tests, NULL, NULL)

#endif

