cmake_minimum_required(VERSION 2.8.11)
project(okvis_frontend)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()    

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR} ${OpenGV_INCLUDE_DIRS}) 


# build the library
add_library(${PROJECT_NAME}
		src/agast/agast5-8.cc
		src/agast/agast5-8-nms.cc
		src/agast/agast7-12d.cc
		src/agast/agast7-12d-nms.cc
		src/agast/agast7-12s.cc
		src/agast/agast7-12s-nms.cc
		src/agast/ast-detector.cc
		src/agast/non-maximum-suppression.cc
		src/agast/oast9-16.cc
		src/agast/oast9-16-nms.cc
		src/brisk/brisk-descriptor-extractor.cc
		src/brisk/brisk-feature-detector.cc
		src/brisk/brisk-layer.cc
		src/brisk/brisk-scale-space.cc
		src/brisk/brute-force-matcher.cc
		src/brisk/harris-feature-detector.cc
		src/brisk/harris-score-calculator.cc
		src/brisk/harris-score-calculator-float.cc
		src/brisk/harris-scores.cc
		src/brisk/image-down-sampling.cc
		src/brisk/pattern-provider.cc
		src/brisk/vectorized-filters.cc
  src/Frontend.cpp
  src/VioKeyframeWindowMatchingAlgorithm.cpp
  src/stereo_triangulation.cpp
  src/ProbabilisticStereoTriangulator.cpp
  src/FrameNoncentralAbsoluteAdapter.cpp
  src/FrameRelativeAdapter.cpp
  include/okvis/Frontend.hpp
  include/okvis/VioKeyframeWindowMatchingAlgorithm.hpp
  include/okvis/triangulation/stereo_triangulation.hpp
  include/okvis/triangulation/ProbabilisticStereoTriangulator.hpp
  include/opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp
  include/opengv/relative_pose/FrameRelativeAdapter.hpp
  include/opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp)

# and link it                 
target_link_libraries(${PROJECT_NAME}
  PRIVATE ${BRISK_LIBRARIES} 
  PRIVATE ${OpenGV_LIBRARIES} 
  PRIVATE ${CERES_LIBRARIES}
  PRIVATE brisk
  PUBLIC okvis_util
  PUBLIC okvis_cv 
  PUBLIC okvis_ceres 
  PUBLIC okvis_timing 
  PUBLIC okvis_matcher)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
