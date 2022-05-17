cmake_minimum_required(VERSION 3.0)

project(exercises)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen3
include_directories("/usr/include/eigen3")

# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Sophus
#find_package(Sophus REQUIRED)
#include_directories(${Sophus_INCLUDE_DIRS})

# CSparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# -------------------------------------------------------------------

add_executable( directSparseRandomG2O  direct_method_sparse_random_g2o.cpp)
target_link_libraries(directSparseRandomG2O ${OpenCV_LIBS} ${G2O_LIBS} g2o_core g2o_stuff g2o_csparse_extension cxsparse)

add_executable( directSparseFeatureG2O  direct_method_sparse_keypoints_g2o.cpp)
target_link_libraries(directSparseFeatureG2O ${OpenCV_LIBS} ${G2O_LIBS} g2o_core g2o_stuff g2o_csparse_extension cxsparse)


#add_executable( directSparseFeatureCeres direct_method_sparse_keypoints_ceres.cpp )
#target_link_libraries(directSparseFeatureCeres ${OpenCV_LIBS} ${CERES_LIBRARIES})


add_executable( directSemiDenseG2O direct_method_semi_g2o.cpp )
target_link_libraries(directSemiDenseG2O ${OpenCV_LIBS} ${G2O_LIBS} g2o_core g2o_stuff g2o_csparse_extension cxsparse)
