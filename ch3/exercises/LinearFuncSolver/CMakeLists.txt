cmake_minimum_required(VERSION 2.8)

project(linearSolver)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")

include_directories("/usr/include/eigen3")

add_executable(linearSolveEigen EigenLinearSolving.cpp)
