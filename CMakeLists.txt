cmake_minimum_required(VERSION 2.8.11)

project(okvis)
# The version number.
set(OKVIS_MAJOR_VERSION 1)
set(OKVIS_MINOR_VERSION 1)
set(OKVIS_PATCH_VERSION 3)
set(OKVIS_VERSION
  ${OKVIS_MAJOR_VERSION}.${OKVIS_MINOR_VERSION}.${OKVIS_PATCH_VERSION})

# options for build configuration
# options for build configuration

option (USE_LOCAL_CERES
        "Use ceres via find_package rather than downloading it as part of okvis" ON)
option (HAS_BUILD_OPENGV
        "Use opengv via find_package rather than downloading it as part of okvis" ON) 

option (BUILD_APPS
        "Builds a demo app (which requires boost)" ON)

option (BUILD_TESTS
        "Builds all gtests" ON)
SET(N_CORES 4 CACHE STRING "Using N number of cores for parallel build")

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/okvis)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
 
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()


set (CMAKE_BUILD_TYPE Release)

# make sure we use Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
else()
  message(STATUS "Building with build type '${CMAKE_BUILD_TYPE}'")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


if(APPLE)
  # The clang compiler (on osx) is somehow much more strict
  # than the compilers on ubuntu and so this does not seem
  # possible on OSX just yet.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++11 -fPIC")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++11 -fPIC")
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()

get_directory_property(defs COMPILE_DEFINITIONS)

# EXTERNAL DEPENDENCIES
include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include_directories(${CMAKE_BINARY_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/lib)

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# libvisensor
find_package(VISensor)

if(VISENSORDRIVER_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  set(LIBVISENSOR TRUE)
  message(STATUS "Found libvisensor. Setting HAVE_LIBVISENSOR flag.")
  include_directories(${VISensorDriver_INCLUDE_DIR})
  # install find script
  install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindVISensor.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/CMake)
else()
  message(STATUS "libvisensor not found")
endif()

#set(OpenCV_DIR /home/pang/software/opencv2413_install/share/OpenCV)

# glog
find_package(Glog REQUIRED)
include_directories(BEFORE ${GLOG_INCLUDE_DIRS})

# ceres
find_package(SuiteSparse REQUIRED QUIET) #will be needed in any case...
find_package(CXSparse QUIET) #optional
if(${USE_LOCAL_CERES})
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  find_package( Ceres 1.9 REQUIRED )
  include_directories(BEFORE ${CERES_INCLUDE_DIRS})
  message(STATUS "Using local Ceres.")
else()
  set(CERES_VERSION 1.9.0)
  set(CERES_TAG 7c57de5080c9f5a4f067e2d20b5f33bad5b1ade6) #version 1.9. Newer ones would need adaptation with ceres::ParameterBlock
  ExternalProject_Add(ceres_external
    GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver
    GIT_TAG ${CERES_TAG}
    UPDATE_COMMAND ""
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CMAKE_ARGS 
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> 
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} 
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_TESTING:BOOL=OFF
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ceres
    BUILD_IN_SOURCE 0
    BUILD_COMMAND make -j${N_CORES}
    INSTALL_COMMAND make install
  )

  add_library(ceres UNKNOWN IMPORTED)
  # Heuristic for determining LIB_SUFFIX from ceres
  set(CERES_LIB_PREFIX "lib")
  if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND
    NOT DEFINED LIB_SUFFIX AND
    NOT CMAKE_CROSSCOMPILING AND
    CMAKE_SIZEOF_VOID_P EQUAL "8" AND
    NOT EXISTS "/etc/debian_version" AND
    NOT EXISTS "/etc/arch-release")
   set(CERES_LIB_PREFIX "lib64")
  endif ()

  # to find it from 3rd party software, since not added to registry:
  set(OKVIS_CERES_CONFIG "${CMAKE_BINARY_DIR}/share/Ceres/")

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set_property(TARGET ceres PROPERTY IMPORTED_LOCATION
               ${CMAKE_BINARY_DIR}/${CERES_LIB_PREFIX}/libceres-debug.a )
  else () 
    set_property(TARGET ceres PROPERTY IMPORTED_LOCATION
               ${CMAKE_BINARY_DIR}/${CERES_LIB_PREFIX}/libceres.a )
  endif ()
   
  find_package(OpenMP QUIET)
  if (OPENMP_FOUND)
    message(STATUS "OpenMP found")
    add_definitions(-DUSE_OPENMP)
    set(CERES_LIBRARIES ceres gomp ${SUITESPARSE_LIBRARIES} ${CXSPARSE_LIBRARIES} ${Eigen_LIBS} ${GLOG_LIBRARY}) #ceres won't export these
  else ()
    message(STATUS "OpenMP NOT found")
    set(CERES_LIBRARIES ceres ${SUITESPARSE_LIBRARIES} ${CXSPARSE_LIBRARIES}  ${Eigen_LIBS} ${GLOG_LIBRARY}) #ceres won't export these
  endif ()
  add_dependencies(${CERES_LIBRARIES} ceres_external)

  # install it once built
  install(FILES ${CMAKE_BINARY_DIR}/${CERES_LIB_PREFIX}/libceres.a 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/include/ceres/ 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/ceres/)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/share/Ceres/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/Ceres/)
endif()

# OpenGV
if(${HAS_BUILD_OPENGV})  # using absolute path
SET(OpenGV_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdpart_install/include)
SET(OpenGV_LIBRARIES ${PROJECT_SOURCE_DIR}/3rdpart_install/lib/libopengv.a)
endif()


# BUILD LOCAL DEPENDENCIES
include_directories(okvis_util/include)
add_subdirectory(okvis_util)

include_directories(okvis_kinematics/include)
add_subdirectory(okvis_kinematics)
add_dependencies(okvis_kinematics okvis_util)

include_directories(okvis_time/include)
add_subdirectory(okvis_time)
add_dependencies(okvis_time okvis_util)

# FLAME
include_directories(mesh_util/include)
add_subdirectory(mesh_util)

include_directories(okvis_cv/include)
add_subdirectory(okvis_cv)
add_dependencies(okvis_cv  okvis_util)

include_directories(okvis_common/include)
add_subdirectory(okvis_common)



include_directories(okvis_ceres/include)
add_subdirectory(okvis_ceres)
add_dependencies(okvis_ceres  okvis_util)

include_directories(okvis_timing/include)
add_subdirectory(okvis_timing)

include_directories(okvis_matcher/include)
add_subdirectory(okvis_matcher)
add_dependencies(okvis_matcher okvis_util)

include_directories(okvis_frontend/include)
add_subdirectory(okvis_frontend)
if(HAS_BUILD_OPENGV)
add_dependencies(okvis_frontend  okvis_util)
else()
add_dependencies(okvis_frontend opengv okvis_util)
endif()

# FLAME
include_directories(mesh_estimate/include)
add_subdirectory(mesh_estimate)

include_directories(okvis_multisensor_processing/include)
add_subdirectory(okvis_multisensor_processing)
add_dependencies(okvis_multisensor_processing okvis_util)


include_directories(pipeline/include)
add_subdirectory(pipeline)
add_dependencies(pipeline okvis_util)

# also build the apps
if(BUILD_APPS)
 find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
 include_directories(BEFORE ${OpenCV_INCLUDE_DIRS}) 

  message(STATUS "Opencv path '${OpenCV_INCLUDE_DIRS}'")

  FIND_PACKAGE(Boost COMPONENTS filesystem system REQUIRED)
  include_directories(${Boost_INCLUDE_DIRS}) 
  add_executable(okvis_app_synchronous okvis_apps/src/okvis_app_synchronous.cpp)
  target_link_libraries(okvis_app_synchronous 
    okvis_util
    okvis_kinematics
    okvis_time
    okvis_cv 
    okvis_common
    okvis_ceres
    okvis_timing
    okvis_matcher
    okvis_frontend 
    okvis_multisensor_processing
    pthread 
    ${OpenCV_LIBRARIES} 
    ${Boost_LIBRARIES}
  )

 add_executable(okvis_app_pipeline okvis_apps/src/okvis_app_pipeline.cpp)
 target_link_libraries(okvis_app_pipeline
         okvis_util
         okvis_kinematics
         okvis_time
         okvis_cv
         okvis_common
         okvis_ceres
         okvis_timing
         okvis_matcher
         okvis_frontend
         pipeline
         pthread
         ${OpenCV_LIBRARIES}
         ${Boost_LIBRARIES}
         )

  if(${VISENSORDRIVER_FOUND})
    target_link_libraries(okvis_app_synchronous ${VISensorDriver_LIBRARY})
  endif()
  install(TARGETS okvis_app_synchronous
    # IMPORTANT: Add the executable to the "export-set"
    EXPORT okvisTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
endif()

# installation is invoked in the individual modules...
export (TARGETS 
    okvis_app_synchronous
    okvis_util
    okvis_kinematics
    okvis_time
    okvis_cv 
    okvis_common
    okvis_ceres
    okvis_timing
    okvis_matcher
    okvis_frontend
    okvis_multisensor_processing
        mesh_util
        mesh_estimate
        pipeline
    FILE "${PROJECT_BINARY_DIR}/okvisTargets.cmake")
export ( PACKAGE okvis )

# Create the okvisConfig.cmake and okvisConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS 
  "${PROJECT_SOURCE_DIR}/okvis_util/include"
  "${PROJECT_SOURCE_DIR}/okvis_kinematics/include"
  "${PROJECT_SOURCE_DIR}/okvis_time/include"
  "${PROJECT_SOURCE_DIR}/okvis_cv/include"
  "${PROJECT_SOURCE_DIR}/okvis_common/include"
  "${PROJECT_SOURCE_DIR}/okvis_ceres/include"
  "${PROJECT_SOURCE_DIR}/okvis_timing/include"
  "${PROJECT_SOURCE_DIR}/okvis_matcher/include"
  "${PROJECT_SOURCE_DIR}/okvis_frontend/include"
  "${PROJECT_SOURCE_DIR}/okvis_multisensor_processing/include"
  "${PROJECT_SOURCE_DIR}/mesh_util/include"
  "${PROJECT_SOURCE_DIR}/mesh_estimate/include"
  "${PROJECT_SOURCE_DIR}/pipeline/include"
  "${PROJECT_BINARY_DIR}"
)
configure_file(cmake/okvisConfig.cmake.in
  "${PROJECT_BINARY_DIR}/okvisConfig.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${OKVIS_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/okvisConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/okvisConfig.cmake" @ONLY)
# ... for both
configure_file(cmake/okvisConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/okvisConfigVersion.cmake" @ONLY)
 
# Install the okvisConfig.cmake and okvisConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/okvisConfig.cmake"
  "${PROJECT_BINARY_DIR}/okvisConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}")
 
# Install the export set for use with the install-tree
install(EXPORT okvisTargets DESTINATION
  "${INSTALL_CMAKE_DIR}")
