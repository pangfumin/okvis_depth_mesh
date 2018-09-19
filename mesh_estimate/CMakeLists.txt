cmake_minimum_required(VERSION 2.8.11)
project(mesh_estimate)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()


# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
FIND_PACKAGE(Boost COMPONENTS filesystem system thread REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Build main library.
set(FLAME_SRCS
        src/mesh_estimator.cpp
        src/flame.cc

        src/optimizers/nltgv2_l1_graph_regularizer.cc

        src/stereo/inverse_depth_filter.cc
        src/stereo/inverse_depth_meas_model.cc
        )

add_library(mesh_estimate  ${FLAME_SRCS})

target_link_libraries(mesh_estimate
        okvis_kinematics
        okvis_ceres
        okvis_cv
        mesh_util
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES})


# export config
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.hpp)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/okvis" COMPONENT dev
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")


# test
enable_testing()

file(GLOB TEST_SOURCES "test/*.cpp")

set(TEST_SOURCES
        test/test_epipolar_geometry.cpp
        test/test_stereo_epipolar_geometry.cpp
        test/test_inverse_depth_filter.cpp)

add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} .)
target_link_libraries(${PROJECT_NAME}_test pthread ${PROJECT_NAME} okvis_util)
set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")

add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)

# Copy test data.
file(COPY ./test/data DESTINATION ${CMAKE_BINARY_DIR})