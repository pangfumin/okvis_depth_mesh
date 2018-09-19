cmake_minimum_required(VERSION 2.8.11)
project(mesh_util)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()


# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
FIND_PACKAGE(Boost COMPONENTS filesystem system thread REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Build main library.
set(MESH_UTIL_SRCS
        src/external/triangle/triangle.cpp

        src/utils/delaunay.cc
        src/utils/frame.cc
        src/utils/image_utils.cc
        #src/utils/keyframe_selector.cc
        src/utils/pyramids.cc
        src/utils/rasterization.cc)

add_library(mesh_util  ${MESH_UTIL_SRCS})

target_link_libraries(mesh_util
        okvis_kinematics
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

