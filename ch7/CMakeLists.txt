cmake_minimum_required(VERSION 2.8)

project(vo1)

set(CMAKE_BUILD_TYPE "Release")
#add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# Eigen3
include_directories("/usr/include/eigen3")

# g2o
include_directories(${G2O_INCLUDE_DIRS})

add_executable(orbCV orb_cv.cpp)
target_link_libraries(orbCV ${OpenCV_LIBS})

add_executable(orbSelf orb_self.cpp)
target_link_libraries(orbSelf ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} Sophus::Sophus g2o_core g2o_stuff)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} Sophus::Sophus g2o_core g2o_stuff)
