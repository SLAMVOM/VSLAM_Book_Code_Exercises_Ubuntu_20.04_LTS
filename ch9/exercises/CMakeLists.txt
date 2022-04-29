cmake_minimum_required(VERSION 3.0)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++14")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

Find_Package(G2O REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)

include_directories("usr/include/eigen3")
include_directories(${G2O_INCLUDE_DIRS})

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_library(bal_common common.cpp)
add_executable(BALCeresBA BAL_BA_ceres.cpp)
add_executable(g2oFullCamModelBA BAL_BA_FullCamModel_g2o.cpp)

target_link_libraries(BALCeresBA ${CERES_LIBRARIES} bal_common)
target_link_libraries(g2oFullCamModelBA Sophus::Sophus ${G2O_LIBS} bal_common)
