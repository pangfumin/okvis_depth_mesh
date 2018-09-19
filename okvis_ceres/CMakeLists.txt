cmake_minimum_required(VERSION 2.8.11)
project(okvis_ceres)

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR} )
               
# build the library 
add_library(${PROJECT_NAME} 
  src/PoseParameterBlock.cpp
  src/SpeedAndBiasParameterBlock.cpp
  src/HomogeneousPointParameterBlock.cpp
  src/HomogeneousPointLocalParameterization.cpp
  src/PoseLocalParameterization.cpp
  src/ImuError.cpp
  src/PoseError.cpp
  src/RelativePoseError.cpp
  src/SpeedAndBiasError.cpp
  src/IdProvider.cpp
  src/Map.cpp
  src/MarginalizationError.cpp
  src/HomogeneousPointError.cpp
  src/Estimator.cpp
  src/LocalParamizationAdditionalInterfaces.cpp
  include/okvis/Estimator.hpp
  include/okvis/ceres/CeresIterationCallback.hpp
)

# and link it
target_link_libraries(${PROJECT_NAME} 
  PUBLIC okvis_util
  PUBLIC okvis_cv 
  PUBLIC okvis_common
  PUBLIC mesh_util
  PRIVATE ${CERES_LIBRARIES}
  PRIVATE ${OpenCV_LIBRARIES} 
)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")



# test
enable_testing()

file(GLOB TEST_SOURCES "test/*.cpp")

set(TEST_SOURCES
        #test/TestEstimator.cpp
        test/TestHomogeneousPointError.cpp
        test/TestReprojectionError.cpp
        test/TestImuError.cpp
        test/TestMap.cpp
        test/TestMarginalization.cpp)

add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} .)
target_link_libraries(${PROJECT_NAME}_test pthread ${PROJECT_NAME} okvis_util)
set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")

add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)

