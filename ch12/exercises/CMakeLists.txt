cmake_minimum_required(VERSION 2.8)
project(ch12exercises)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -O3")

############## Dependencies #############
# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
	${OpenCV_LIBS}
	${Sophus_LIBRARIES})

add_executable(DenseInvDepth dense_InvDepth.cpp)
target_link_libraries(DenseInvDepth Sophus::Sophus ${THIRD_PARTY_LIBS})

add_executable(DenseInvDepthAffine dense_InvDepth_Affine.cpp)
target_link_libraries(DenseInvDepthAffine Sophus::Sophus ${THIRD_PARTY_LIBS})


