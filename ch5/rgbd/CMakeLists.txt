cmake_minimum_required(VERSION 2.8)

project(jointMap)

set( CMAKE_CXX_FLAGS "-std=c++14")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS} )

find_package(Pangolin REQUIRED)
find_package(fmt REQUIRED)
set(FMT_LIBRARIES fmt::fmt)
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( jointMap jointMap.cpp )

target_link_libraries( jointMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${FMT_LIBRARIES} fmt Sophus::Sophus )


