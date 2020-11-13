TDycore Development
===================

This section contains topics that are useful to anyone developing the TDycore
library or trying to use it in an application.

Initializing the TDycore Library
--------------------------------

Because the TDycore library uses MPI, PETSc, and other subsystems, we have
defined a function to initialize these various subsystems at the beginning of
any TDycore-based driver/program:::

    PetscErrorCode TDyInit(int argc, char* argv[]);

Call this function where you would ordinarily call ``MPI_Init`` or
``PetscInitialize``. It has no effect on subsequent calls. You can check whether
the library has been initialized with a call to::

    PetscBool TDyInitialized(void);

Similarly, we have defined a finalization function to be called at the end of a
TDycore-based program:::

    PetscErrorCode TDyFinalize(void);

Use this instead of ``MPI_Finalize`` or ``PetscFinalize``. This ensures that all
TDycore subsystems properly free their resources.

Fortran 90 Interface
^^^^^^^^^^^^^^^^^^^^

We offer two equivalent subroutines for Fortran 90, similar to their PETSc
counterparts:::

    TDyInit(ierr)
    TDyFinalize(ierr)

Both accept an integer that stores an error code if these subroutines encounter
an issue.

Timers and Profiling
--------------------

We use PETSc's `Logging machinery <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Profiling/index.html>`
to understand the performance of the dycores. In particular, we've provided some
high-level wrappers around the `PetscLogEvent <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Profiling/PetscLogEvent.html>`
object that make it very easy to add timers for functions and blocks of code.

The TDycore Timers Subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The subsystem for timers and profiling is defined almost entirely in ``tdytimers.h``.

When a dycore is initialized, a registry of timers is created when you call
``TDyInit()``. Once this is done, you can manipulate timers with functions and
macros as described below. Make sure you include the ``tdytimers.h`` file
wherever you use these timers.

Function-Level Profiling
^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to use a timer is to instrument a function with a timer using
``TDY_START_FUNCTION_TIMER()`` and ``TDY_STOP_FUNCTION_TIMER()``. These macros
automatically create/retrieve a timer for the function and start/stop it as
you'd expect. You place the ``START`` macro at the top of a function, and the
``STOP`` one at the bottom. For example:::

    void do_some_expensive_things(TDy dy)
    {
      TDY_START_FUNCTION_TIMER()
      ... // Do the expensive things
      TDY_STOP_FUNCTION_TIMER()
    }

There's no need to understand PETSc's logging objects--everything is done for
you. These function-level timers are named after the functions in which they
appear.

Manually-Created Timers
^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you want to time something that happens in the middle of a function.
You can do this by calling the ``TDyGetTimer`` function with the
``TDyStartTimer`` and ``TDyStopTimer`` macros:::

    void do_various_things(TDy dy)
    {
      ... // Stuff happens here

      // Now we want to time a block of code in the middle of the function.
      PetscLogEvent timer = TDyGetTimer("important things");
      TDyStartTimer(timer);
      ... // Important things happen here!
      TDyStopTimer(timer);

      ... // Other stuff happens here
    }

This is a bit more involved--you need to know that timers are ``PetscLogEvent``
objects, for example, and you need to name your timers--but not too difficult.
As you might have guessed, ``TDY_START_FUNCTION_TIMER`` and
``TDY_STOP_FUNCTION_TIMER`` are just wrappers around these constructs.
``TDyStartTimer`` and ``TDyStopTimer`` are themselves just macros that call
``PetscLogEventBegin`` and ``PetscLogEventEnd``, so you can always use those if
you want more control.

Profiling Stages
^^^^^^^^^^^^^^^^

PETSc allows an arbitrary number of logging/profiling "stages" to be defined so
that you can organize your profiling into sections. These stages can be named
for convenience. You can enter and exit a named stage with calls to
``TDyEnterProfilingStage(stageName)`` and ``TDyExitProfilingStage(stageName)``,
where ``stageName`` is a string containing the name of the stage.

TDycore provides these named stages, registering them in ``TDyInit``:

* ``"TDycore Setup"``: for creating meshes, setting up initial conditions,
  calculating time-independent matrices and vectors, etc.
* ``"TDycore Stepping"``, for timestepping
* ``"TDycore I/O"``, for checkpointing, restarting, generating visualizations

You can register your own named stages with ``TDyAddProfilingStage(stageName)``.
All of this machinery is a thin wrapper around PETSc's ``PetscLogStage``
mechanism, which you can use if you prefer. The functions and macros above just
simplify the bookkeeping.

Generating Profiling Logs
^^^^^^^^^^^^^^^^^^^^^^^^^

Adding timers to a code is only part of profiling. You also need to generate
profiling reports for runs of interest! Fortunately, this is easy--just add the
``-tdy_timers`` flag to your command line arguments to generate a performance
log. This log is named ``tdycore_profile.csv``. It's a comma-separated variable
file containing all performance data collected by PETSc. The timers you've
created show up in the profile just like those embedded in the PETSc library.

If you'd rather look at the traditional profiling/log data dumped by PETSc, you
can use the ``-log_view`` flag to have PETSc print that information to the
standard output.

Interpreting Profile Data with TDyProf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you've generated a ``tdycore_profile.csv`` file, you can use a tool called
``tdyprof`` (located in the ``tools/`` subdirectory of the source tree). This
Python script digests the contents of the CSV file you give it and generates
nicely-formatted reports for desired information. Use it thus:::

    tdyprof <profile.csv> <command> [options]

or just type ``tdyprof`` by itself to see its usage information. For example,
to see the top 10 "hotspots" in the performance profile:::

    tdyprof tdycore_profile.csv top10
    tdyprof: showing top 10 hits:
          Stage Name                               Event Name             Time             FLOP
          Main Stage               TDyTimeIntegratorRunToTime         0.139049      1.85991e+07
          Main Stage                                SNESSolve          0.13895      1.85991e+07
          Main Stage                         SNESJacobianEval         0.120045          609812.
          Main Stage              TDyMPFAOSNESJacobian_3DMesh         0.120032          609812.
          Main Stage        TDyMPFAOIJacobian_Vertices_3DMesh         0.118853          606312.
       TDycore Setup                   TDyDriverInitializeTDy        0.0912797           52040.
       TDycore Setup                                 TDySetup        0.0533342           52040.
       TDycore Setup                       TDyMPFAOInitialize        0.0533217           52040.
          Main Stage                         DMPlexDistribute        0.0386017               0.
       TDycore Setup                        TDyCreateJacobian        0.0370897               0.

Generating Scaling Study Plots with TDyPerfPlot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO
