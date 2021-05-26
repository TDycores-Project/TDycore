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
// If setup is non-NULL, the function is called with argc and argv, with
// any given test index argument removed. If breakdown is non-NULL, it's
// registered to be called on program exit.
static int _run_selected_tests(int argc, char **argv,
                               void (*setup)(int argc, char **argv),
                               int num_tests,
                               const struct CMUnitTest tests[num_tests],
                               void (*breakdown)(void),
                               int nproc) {

  if (argc == 1) {
    if (setup != NULL) {
      setup(argc, argv);
    }
    fprintf(stdout, "Usage:\n");
    fprintf(stdout, "%s count [arg1 [...]]  \treports the number of available unit tests\n", argv[0]);
    fprintf(stdout, "%s nproc [arg1 [...]]  \treports the # of MPI procs supported\n", argv[0]);
    fprintf(stdout, "%s <index> [arg1 [...]]\truns the unit test with the (0-based) index\n", argv[0]);
    return 0;
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
          if (num_tests == 1) {
            fprintf(stderr, "Invalid test index: %ld (must be 0)\n", index);
          } else {
            fprintf(stderr, "Invalid test index: %ld (must be between 0 and %d)\n",
                    index, num_tests-1);
          }
          return 1;
        } else {
          // We have a valid test index. If we have been given a setup function,
          // construct our own argc and argv with the index argument removed and
          // call the function with them.
          int my_argc = -1;
          char** my_argv = NULL;
          if (setup != NULL) {
            my_argc = argc - 1;
            my_argv = malloc(my_argc * sizeof(char*));
            my_argv[0] = _strdup(argv[0]);
            for (int i = 1; i < my_argc; ++i) {
              my_argv[i] = _strdup(argv[i+1]);
            }
            setup(my_argc, my_argv);
          }

          // Select and run the tests.
          if (breakdown != NULL) atexit(breakdown);
          const struct CMUnitTest selected_tests[] = { tests[index] };
          int result = cmocka_run_group_tests(selected_tests, NULL, NULL);

          // Clean up our duplicated argument list if needed.
          if (setup != NULL) {
            for (int i = 0; i < my_argc; ++i) {
              free(my_argv[i]);
            }
            free(my_argv);
          }
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
// * setup - a void function that takes argc and argv as arguments and
//           performs any required setup
// * tests - A list of CMUnitTest structs that make up the test suite for your
//           unit test
// * breakdown - a void function that performs any cleanup needed after the
//               selected tests execute.
// * nproc - The number of MPI processes required to execute this unit test
#define run_selected_tests(argc, argv, setup, tests, breakdown, nproc) \
  _run_selected_tests(argc, argv, setup, \
                      sizeof(tests) / sizeof((tests)[0]), \
                      tests, breakdown, nproc)

#endif

