Style Guide
===========

This document describes rules, guidelines, and best practices for writing
code for the TDycore library. The structure of this guide was inspired by
Google's C++ style guide, which appears at

https://google.github.io/styleguide/cppguide.html

.. contents:: Table of Contents

Data Types
==========

Numeric Datatypes
-----------------

TDycore relies on the `PETSC <https://www.mcs.anl.gov/petsc/>` library, and
adopts many of its conventions. In particular, the floating point precision of
TDycore is determined by its underlying PETSc library. The following numeric
data types are supported by PETSc and used by TDycore:

* ``PetscScalar``: A floating point number at the supported level of precision
  (single or double), that is real-valued or complex-valued, depending on how
  PETSc is configured
* ``PetscReal``: A real-valued floating point number at the supported level of
  precision
* ``PetscInt``: An integer whose size is determined by PETSc.

Use these types for numeric calculations within TDycore.

Floating point comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, avoid comparing two floating point numbers for equality. Such
comparisons depend on the representation of the supported precision and the
magnitudes of the quantities under comparison. Instead...
**Question: Any thoughts about what we encourage instead?**

Structs / PODs
--------------

In the TDycore library, a "struct" is a C struct containing data that can
be freely exposed, with no invariants and no restrictions on values. Structs
have no behavior and internal state to manage. They are defined in header files
so that their data members are visible and accessible. Examples of structs in
the TDycore library are the ``MaterialProp`` and ``CharacteristicCurve`` types,
which represent sets of material properties and parameterized curves (a
saturation curve, for example), respectively.

Since a struct is a simple container without behavior (a "Plain Old Datatype",
or "POD"), no associated functions are needed. However, such functions can be
provided if they make the struct/POD more convenient to use.

Classes
-------

A "class" in TDycore is a struct whose definition is **private**--not exposed
within the public interface defined by the set of header files intended for use
by external developers. This means that a developer cannot directly manipulate
the fields within a "class". Instead, one manipulates a class using its
**interface**--a set of functions associated with that class.

Define class bodies in source files only, unless their internal structure is
intended to be explicitly exposed to developers. "Typedef" your class type so
the ``struct`` keyword can be omitted from its type.

The struct and functions defining a class are governed by a few simple
conventions described below. PETSc uses similar conventions.

Class Type (Struct)
^^^^^^^^^^^^^^^^^^^

The struct representing the class type has the same name as the class itself.
The type of the class itself is a pointer to its underlying struct. The
underlying struct is declared with a ``_p`` suffix. Then the class is declared
as a ``typedef`` to a pointer to the struct type. For example, if you want to
declare a "washing machine" class, first declare its underlying struct (without
defining its body):::

    typedef struct WashingMachine_p;

in an appropriate header file. Then define the class itself:::

    typedef WashingMachine_p* WashingMachine;

Class Constructor(s)
^^^^^^^^^^^^^^^^^^^^

Typically, a class has a single constructor function named after the class, with
``New`` at the end. A constructor takes a number of arguments for initializing
the class, plus a final argument that stores a pointer to a newly-allocated
instance of the class. For example, consider the following constructor for our
``WashingMachine`` class:::

    PetscErrorCode WashingMachineNew(PetscInt numCents, WashingMachine* wm);

This constructor creates a ``WashingMachine`` instance that costs the given
number of cents to wash a load of laundry. The ``wm`` argument stores the
new instance. The constructor returns an integer-valued error code described
in the section on functions.

Sometimes it's convenient to provide more than one constructor, or a
constructor that converts another datatype to a given instance of a class.
In these cases, name each constructor so that it briefly conveys its purpose.
For example, a constructor that converts an array of ``real_t`` to a point
might be declared

``point_t* point_from_array(real_t* array);``

A constructor function takes any arguments it needs to completely initialize
an variable of that class type, and returns a pointer to such an initialized
variable. We refer to these variables as objects.

Class Destructor
^^^^^^^^^^^^^^^^

A destructor function frees the resources allocated to a class by its
constructor. Define a single destructor function for each class.
The destructor function is named after the class with a ``Destroy`` suffix,
accepts a pointer to the instance of the class to be destroyed, and returns an
error code indicating whether an error was encountered. For example:::

    PetscErrorCode WashingMachineDestroy(WashingMachine* wm);

Class Functions / Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

Functions associated with a class are sometimes referred to as methods (some
object-oriented programming languages make a bigger distinction between these
concepts). The name of a class method begins with the name of the associated
class, followed by a descriptive name for the method itself. For example:::

    // Washes a load of laundry, changing its state from DIRTY to CLEAN.
    PetscError WashingMachineWash(WashingMachine* wm, Laundry* load);

A method can perform a task involving the instance and other data provided as
arguments, as shown above. In this case, it returns a ``PetscErrorCode``
indicating success or failure. A method can also provide access to data within
the instance of the class, returning that data:::

    // Returns the cost (in cents) of washing a load.
    PetscInt WashingMachineCost(WashingMachine* wm);

If you're familiar with contemporary object-oriented programming languages like
C++ and Java, you can define methods in very similar ways (as long as you don't
wander too far into inheritance and other "polymorphic" techniques. If it's
practical, lead the list of parameters with input values, and place output
parameters at the end.

Polymorphism in C
^^^^^^^^^^^^^^^^^

**Question: do we use this idea? PETSc does, but it might not be needed in
TDycore itself.**

Header Files
============

In general, there should be a header file for each significant type that
possesses behaviors in TDycore. Header file names are all lowercase.
In some cases, a single function (unrelated to a type) may occupy a header
file, and that header file would be named after the function.  In others, a
header file may contain a set of related functions, and its name should
concisely reflect the purpose of those functions.

Self-Contained Headers
----------------------

Header files are self-contained and have a ``.h`` suffix. A "self-contained"
header file can be included in a translation unit without regard for rules
relating to the order of its inclusion, or for other headers that are
"understood" to be included when it is used.

Briefly, a TDycore header file
* is located in the ``include/tdycore`` subdirectory of the TDycore repo
* requires header guards
* should include all the files that it needs
* should not require any particular symbols to be defined

Header File Location
--------------------

To make it easier to deploy TDycore as part of a larger application, we place
most header files in a ``tdycore`` subdirectory within the ``include`` directory
of the repository. There is a high-level header file called ``tdycore.h`` within
the ``include`` directory that includes all the basic headers within this
``tdycore/`` subdirectory.

This means that headers and source files that reference specific TDycore headers
must include the ``tdycore/`` directory as part of the header file's path. For
example, if you want to use TDycore's I/O subsystem in a source file, you would
place the following near the top of the file:::

    #include <tdycore/tdyio.h>

Alternatively, you can rely on the high-level TDycore header to bring in the
I/O subsystem:::

    #include <tdycore.h>

**Question**: does anyone have an opinion on the use of quotes in headers vs the
use of angle brackets?

Header Guards
-------------

A header file uses ``#define`` guards to prevent multiple inclusion. The
format of the guard is ``<HEADER_BASE_NAME>_H``, e.g.::

    #ifndef TDYCORE_H
    #define TDYCORE_H

"C++" guards that use the ``extern "C"`` specification are not necessary for C++
interoperability, since TDycore has a high-level header safe for inclusion in
C++ programs.

Including Headers within TDycore Source Code
--------------------------------------------

Any header files included in a header or source file should be included in the
following order:

1. The header file corresponding to the source file (if applicable)
2. TDycore library headers
3. Third-party library headers
4. System-level headers

Including files in this order makes it obvious when a TDycore header can't be
included without prerequisites.

Public and Private Headers, Structs and Classes
-----------------------------------------------

There are three types of header files in the TDycore library.

1. **Public headers**: these headers form the public application programming
   interface (API) for TDycore, and live at the top level of the ``include/``
   directory of the TDycore source tree. All functions and types contained in
   these headers may be called by software that uses TDycore.

2. **Private headers**: these headers contain implementation details, and are
   not part of the public API for TDycore. As such, they are not supported for
   usage by external software, and their contents may change without warning.

3. **Fortran headers**: these headers expose an interface for using the TDycore
   library within Fortran programs. They form the public Fortran API for
   TDycore.

Recall that a TDycore *struct* is a container for data that has no associated
behavior and may be freely manipulated by developers. Structs are declared and
defined within public header files. **TODO: example?**

In contrast, a *class* is a data structure with behaviors and invariants. It is
implemented by a pointer to a struct whose fields are hidden from developers.
Its behaviors are implemented by a set of functions that form its interface. A
class struct is declared in a public header file, but its body іs defined in a
private header file. Meanwhile, the functions that make up the interface for a
class are declared in public header files and defined in source files.
**TODO: example?**

Functions
---------

Any function that is part of TDycore's API is declared within a public header
file and implemented in a source file. More than one function may "live" in the
same source file. Prepend each public function's declaration with
``PETSC_EXTERN`` to make it available to external callers.

A function may be "inlined" using the ``PETSC_STATIC_INLINE`` macro.
Functions with no arguments are declared with ``void`` in their argument list,
in accordance with the C standard.

Functions that implement functionality internal to TDycore may be declared in
a private header file, or may be declared ``static`` and implemented within a
single source file, if they are used only within that file.

Global variables
----------------

In general, avoid global variables in header files, apart from constants (which
are preferred to macros, since they can be checked by the compiler). Mutable
global variables should be restricted to translation units in which they are
manipulated, and should be declared as ``static``. If you must expose a global
resource, design an appropriate interface so that it can be properly managed.

Other Symbols
-------------

Use inlined functions instead of macros where possible. Similarly, use
constants instead of macros where possible.

Scoping
=======

Internal Functions
------------------

A function that is used only within a single translation unit should be declared
with the ``PETSC_INTERN`` macro. This prevents its name from appearing in the
list of exported symbols for the TDycore library.

Local Variables
---------------

Declare a local variable as close as possible to where it is used, and not at
the beginning of a function body. Declaring variables where they are used makes
it easier to identify issues involving that variable.

Initialize a variable when you declare it wherever practical.

Scoping Operators
-----------------

If a function has a large number of localized variables that perform work,
curly braces can be used to create a local scope containing these variables.
This eases the process of debugging functions by eliminating these variables
from portions of the function that don't use them.

Functions
=========

Functions not associated with classes follow very similar guidelines to
methods: input arguments come before output arguments. A function that performs
an operation instead of returning a value should return a ``PetscErrorCode``
that indicates whether the operation succeeded or failed.

Length of a Function Body
-------------------------

There is no formal limit to the length of a TDycore function implementation.
If breaking up a function into separate functions is practical, feel free to
do so. However, creating lots of ancillary structure just to break up a long
function can be counterproductive. Use your judgement.

A function may be poorly designed if it is difficult to break up. On the other
hand, if the function performs a complicated task with lots of tightly-coupled
steps, attempting to break it up may make it even more confusing.

At the end of the day, arguments about the optimal length of a function are
aesthetic. These arguments often exert strange and unnatural pressures on code
development. At worst, they encourage people to write code with few comments,
lots of side effects, and/or excessive numbers of tightly-coupled
"sub-functions." Your mileage may vary.

Memory Management
=================

For simplicity, TDycore uses PETSc's memory allocation functions:

* `PetscMalloc <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscMalloc.html>`
* `PetscMalloc1 <>https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscMalloc1.html>`
* `PetscNew <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscNew.html>`
* `PetscFree <>https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFree.html>`

Prefer these to the standard C ``malloc`` and ``free`` functions. This gives
PETSc more information about how much memory is used, and how it is used.

Naming
======

Types
-----

Names of structs, classes, and enumerated types follow the "camel case",
consisting of one or more words with no delimiters, each word beginning with a
capital letter followed by lower-case letters. Each type has a ``TDy`` prefix to
indicate that they belong to the TDycore library. Abbreviations are allowed if
their meaning is reasonably clear. For example: ``TDyMesh``, ``TDyRegion``.

Functions
---------

Function and "method" names also use "camel case" with a ``TDy`` prefix, and
should clearly indicate their purpose, with abbreviations allowed when their
meaning is clear. Methods that implement behaviors for classes should begin
with the name of the class, as discussed above.

Variables and Fields
--------------------

Variables (local or global, including fields in structs and classes) follow the
"snake-case" convention, in which names consist of lower-case words separated by
underscores. Exceptions can be made if it makes code clearer. For example,
capital letters and/or abbreviations may help a variable representing a
quantity resemble a mathematical symbol whose role is clear from the context
in which it is used. Use your judgement. Examples of good variable names are
``mat_prop``, ``mesh``, ``model``, ``precond``, ``integ``, and ``xc``.

Constants, Enums, Macros
------------------------

Constants, fields within enumerated types, and preprocessor macros should use
all capital letters with words separated by underscores. If these appear in
header files, they should have descriptive names that are unique within the
library.

Comments and Code Markup
========================

Use C++ style comments (``//``), which have been supported in C since the
C99 standard. C-style comments (``/* */``) may be used sparingly when the C++
style is less convenient.

To formally document a type or a function in a public header file, use Doxygen's
markup:

http://www.doxygen.org/

In header files, describe your class types, structs, and enumerated types
briefly and clearly. Build the Doxygen documentation to get an idea of what
documentation typically looks like. We use Doxygen's ``///`` delimiters for
code comments, and ``@`` for Doxygen-specific commands.

A type should be documented with a description of its purpose and usage, just
above its declaration. Structs should have one-line descriptions above each of
their fields.

Functions and class methods should each have a description (1-2 sentences) above
their declarations in a public header file. In addition, use the following
markup to annotate the function/method signature:

* For each parameter (argument) for the function, an entry like the following:::
    @param [INTENT] PARAM_NAME A description of the parameter

  Here, ``intent`` is ``in``, ``out``, or ``inout``.

* If the return value needs an explanation, use::
    @returns A description of the return value

Typically, you don't need any documentation markup in implementation source
files. Commenting your implementation code is always helpful, of course.

Formatting
==========

The following formatting rules are non-negotiable for source code in TDycore:

* Use 2 spaces per indentation level.
* No tabs are allowed in source files--use only spaces.

The following guidelines are offered for readably-formatted code:

* If a function declaration doesn't fit neatly on a line, break the line after
  an argument and align the following argument with its first. As long as the
  declaration and definition are clearly readable, it's fine.
* Place curly braces that open a new scope at the end of the line for which the
  scope is declared, not on their own line. Closing curly braces go on a line
  by themselves, at the level of indentation outside of their scope.
* If a line is excessively long (in other words, if it doesn't fit on a single
  screen on a luxuriously large monitor), consider breaking it up.
* C preprocessor directives are not indented at all.
* For functions with several parameters, consider linebreaks after each
  parameter, and consider aligning the parameters to improve readability.

