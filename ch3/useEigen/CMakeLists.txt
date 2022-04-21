cmake_minimum_required( VERSION 2.8 )

project( UseEigen )

add_executable( usingEigen eigenMatrix.cpp )

include_directories("/usr/include/eigen3")