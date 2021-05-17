#ifndef TDYCORE_TESTS_H
#define TDYCORE_TESTS_H

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmocka.h>

// Here's a tiny implementation of the non-standard strdup function.
static char *_strdup(const char *s) {
  size_t n = strlen(s);
  char* dup = malloc(sizeof(char)*(n+1));
  strcpy(dup, s);
  return dup;
}

// Runs tests with selected index (or reports the number of available tests).
// If init_function is non-NULL, the function is called with argc and argv, with
// any given test index argument removed.
static int _run_selected_tests(int argc, char **argv,
                               void (*init_function)(int argc, char **argv),
                               int num_tests,
                               const struct CMUnitTest tests[num_tests],
                               int nproc) {
  if (argc == 1) {
    if (init_function != NULL) {
      init_function(argc, argv);
    }
    return cmocka_run_group_tests(tests, NULL, NULL);
  } else {
    const char* command = (const char*)argv[1];
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
          // We have a valid test index. If we have been given an initialization
          // function, construct our own argc and argv with the index argument
          // removed and call the initialization function with them.
          if (init_function != NULL) {
            int my_argc = argc - 1;
            char** my_argv = malloc(my_argc * sizeof(char*));
            my_argv[0] = _strdup(argv[0]);
            for (int i = 2; i < my_argc; ++i) {
              my_argv[i] = _strdup(argv[i-1]);
            }
            init_function(my_argc, my_argv);
          }
          const struct CMUnitTest selected_tests[] = { tests[index] };
          int result = cmocka_run_group_tests(selected_tests, NULL, NULL);
          return result;
        }
      } else {
        fprintf(stderr, "Invalid command: %s (must be 'nproc', 'count', or index)\n",
            command);
        return 1;
      }
    }
  }
}

// Call this macro instead of cmocka_run_group_tests, with the following
// arguments:
// * argc and argv, as supplied to your main function
// * init_function - a void function that takes argc and argv as arguments and
//                   performs any required setup
// * tests - A list of CMUnitTest structs that make up the test suite for your
//           unit test
// * nproc - The number of MPI processes required to execute this unit test
#define run_selected_tests(argc, argv, init_function, tests, nproc) \
  _run_selected_tests(argc, argv, init_function, \
                      sizeof(tests) / sizeof((tests)[0]), \
                      tests, nproc)

#endif

