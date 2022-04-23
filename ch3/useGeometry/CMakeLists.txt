cmake_minimum_required( VERSION 2.8 )

project ( useGeometry )

add_executable( usingEigenGeo useGeometry.cpp )

include_directories("/usr/include/eigen3")
