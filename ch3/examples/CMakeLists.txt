cmake_minimum_required( VERSION 2.8 )

project ( coordinateTransform )

add_executable ( transformingCoordinate coordinateTransform.cpp )

include_directories("/usr/include/eigen3")

find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )
add_executable ( plotTrajectory plotTrajectory.cpp )
target_link_libraries( plotTrajectory ${Pangolin_LIBRARIES} )
