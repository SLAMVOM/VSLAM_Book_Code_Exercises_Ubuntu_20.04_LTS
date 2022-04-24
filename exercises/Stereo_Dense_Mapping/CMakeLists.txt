cmake_minimum_required(VERSION 3.0)

project(stereoMapping)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
include_directories("usr/include/eigen3")

# fmt
find_package(fmt REQUIRED)
set(FMT_LIBRARIES fmt::fmt)

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# octomap
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})


add_executable(singleStereoMapping single_pair_mapping.cpp)
target_link_libraries( singleStereoMapping ${OpenCV_LIBS} 
                                           ${Pangolin_LIBRARIES} 
                                           ${OCTOMAP_LIBRARIES} 
                                           ${FMT_LIBRARIES}
                                           fmt 
                                           ${PCL_LIBRARIES} 
                                           )
