cmake_minimum_required(VERSION 2.8)

project(LM)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O2")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories("/usr/include/eigen3")

add_executable(LMCurveFit LMcurveFit.cpp)
target_link_libraries(LMCurveFit ${OpenCV_LIBS})

add_executable(LMCurveFitVec LMcurveFitVec.cpp)
target_link_libraries(LMCurveFitVec ${OpenCV_LIBS})

