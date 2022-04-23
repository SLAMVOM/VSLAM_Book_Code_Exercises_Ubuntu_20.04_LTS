cmake_minimum_required( VERSION 2.8 )
project( visualizeGeometry )

set(CMAKE_CXX_FLAGS "-std=c++17")

# Eigen header
include_directories( "/usr/include/eigen3" )

# Pangolin dependency
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( visualizeGeometry visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry ${Pangolin_LIBRARIES} )
