cmake_minimum_required (VERSION 2.6.2)
project (Tdycore C)

if (0)
  # Use Jed Brown's FindPETSc.cmake
  set (CMAKE_MODULE_PATH ${Tdycore_SOURCE_DIR}/config)
  find_package (PETSc)
else ()
  find_path (PETSC_DIR include/petsc.h HINTS ENV PETSC_DIR PATHS $ENV{HOME}/petsc DOC "PETSc top-level directory")
  set (PETSC_ARCH $ENV{PETSC_ARCH} CACHE STRING "PETSc configuration")
  find_path (PETSC_INCLUDE_DIR  petsc.h HINTS "${PETSC_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
  find_path (PETSC_INCLUDE_CONF petscconf.h HINTS "${PETSC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/include" "include" NO_DEFAULT_PATH)
  mark_as_advanced (PETSC_INCLUDE_DIR PETSC_INCLUDE_CONF)
  set (PETSC_INCLUDES ${PETSC_INCLUDE_CONF} ${PETSC_INCLUDE_DIR} CACHE PATH "PETSc include paths" FORCE)
  find_library (PETSC_LIBRARIES NAMES petsc HINTS "${PETSC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/lib" "lib" NO_DEFAULT_PATH)
  find_path (PETSC_CONF_DIR petscrules HINTS "${PETSC_DIR}/${PETSC_ARCH}" PATH_SUFFIXES "lib/petsc/conf" "conf" NO_DEFAULT_PATH)
  mark_as_advanced (PETSC_CONF_DIR)
  include (${PETSC_CONF_DIR}/PETScBuildInternal.cmake)
  mark_as_advanced (PETSC_CLANGUAGE_Cxx)
endif ()

if (PETSC_CLANGUAGE_Cxx)
  enable_language (CXX)
endif ()
enable_language (Fortran)

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${Tdycore_BINARY_DIR}/lib"     CACHE PATH "Output directory for Tdycore archives")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${Tdycore_BINARY_DIR}/lib"     CACHE PATH "Output directory for Tdycore libraries")
set (CMAKE_Fortran_MODULE_DIRECTORY "${Tdycore_BINARY_DIR}/include" CACHE PATH "Output directory for Fortran modules")
mark_as_advanced (CMAKE_ARCHIVE_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_Fortran_MODULE_DIRECTORY)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/tdycore" CACHE PATH "Tdycore install prefix" FORCE)
endif ()
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# XXX @rpath/libtdycore.dylib
set (CMAKE_MACOSX_RPATH NO)
set (CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

include_directories (${PETSC_INCLUDES} ${PETSC_PACKAGE_INCLUDES}
                    "${Tdycore_SOURCE_DIR}/${PETSC_ARCH}/include"
                    "${Tdycore_SOURCE_DIR}/include")

file (GLOB Tdycore_SOURCES_C RELATIVE ${Tdycore_SOURCE_DIR} ${Tdycore_SOURCE_DIR}/src/*.c)
file (GLOB Tdycore_SOURCES_F RELATIVE ${Tdycore_SOURCE_DIR} ${Tdycore_SOURCE_DIR}/src/*.F90)
set  (Tdycore_SOURCES_ALL ${Tdycore_SOURCES_C} ${Tdycore_SOURCES_F})
if (PETSC_CLANGUAGE_Cxx)
  foreach (file ${Tdycore_SOURCES_C})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
  endforeach ()
endif ()

set (BUILD_SHARED_LIBS 1)
set (BUILD_STATIC_LIBS 0)

if (BUILD_SHARED_LIBS)
  add_library (tdycore ${Tdycore_SOURCES_ALL})
  target_link_libraries (tdycore ${PETSC_LIBRARIES} ${PETSC_PACKAGE_LIBS})
  install (TARGETS tdycore LIBRARY DESTINATION lib)
endif ()

if (BUILD_STATIC_LIBS)
  add_library (tdycore-static STATIC ${Tdycore_SOURCES_ALL})
  set_target_properties (tdycore-static PROPERTIES PREFIX "lib" OUTPUT_NAME "tdycore")
  install (TARGETS tdycore-static ARCHIVE DESTINATION lib)
endif ()

install (DIRECTORY include/               DESTINATION include FILES_MATCHING PATTERN "*.h"    )
install (DIRECTORY ${PETSC_ARCH}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h"    )
install (DIRECTORY ${PETSC_ARCH}/include/ DESTINATION include FILES_MATCHING PATTERN "*.mod"  )
install (DIRECTORY lib/tdycore/conf/       DESTINATION lib/tdycore/conf)
install (CODE "execute_process(COMMAND sed -i.arch -e s/$\(PETSC_ARCH\).//g \"${CMAKE_INSTALL_PREFIX}/lib/tdycore/conf/variables\")")
install (CODE "file(REMOVE \"${CMAKE_INSTALL_PREFIX}/lib/tdycore/conf/variables.arch\")")
