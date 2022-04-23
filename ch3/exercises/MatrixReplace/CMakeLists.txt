cmake_minimum_required( VERSION 2.8 )

project( MatrixReplace )

add_executable( replaceMatrixBlock MatrixReplace.cpp )

include_directories("/usr/include/eigen3")