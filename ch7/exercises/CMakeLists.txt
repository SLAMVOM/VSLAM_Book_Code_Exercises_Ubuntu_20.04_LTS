cmake_minimum_required(VERSION 2.8)

project(exercises)

set(CMAKE_BUILD_TYPE "Release")
#add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen3
include_directories("/usr/include/eigen3")

# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
# include_directories(${G2O_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable( featuresCV features_cv.cpp )
target_link_libraries(featuresCV ${OpenCV_LIBS})

add_executable( pnpLandmarkGNBA pnp_landmark_only_3d2d_ver2_GN.cpp )
target_link_libraries(pnpLandmarkGNBA ${OpenCV_LIBS} Sophus::Sophus)

add_executable( pnpLandmarkLMBA pnp_landmark_only_3d2d_LM.cpp )
target_link_libraries(pnpLandmarkLMBA ${OpenCV_LIBS} Sophus::Sophus)

add_executable( pnpPoseLandmarkGNBA pnp_pose_landmark_BA_3d2d_GN.cpp )
target_link_libraries(pnpPoseLandmarkGNBA ${OpenCV_LIBS} Sophus::Sophus)

add_executable( pnpPoseLandmarkLMBA pnp_pose_landmark_BA_3d2d_LM.cpp )
target_link_libraries(pnpPoseLandmarkLMBA ${OpenCV_LIBS} Sophus::Sophus)

add_executable( pnpPoseCeres pnp_pose_only_3d2d_Ceres.cpp )
target_link_libraries(pnpPoseCeres ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)

add_executable( pnpPoseCeresV2 pnp_pose_only_3d2d_Ceres_v2.cpp )
target_link_libraries(pnpPoseCeresV2 ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)

add_executable( pnpPoseCeresV2proj pnp_pose_only_3d2d_Ceres_v2_proj.cpp )
target_link_libraries(pnpPoseCeresV2proj ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)

add_executable( pnpPoseLandmarkCeres pnp_pose_landmark_3d2d_Ceres.cpp )
target_link_libraries(pnpPoseLandmarkCeres ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)

add_executable( pnpPoseLandmarkG2OBA  pnp_pose_landmark_g2o_BA_3d2d.cpp)
target_link_libraries(pnpPoseLandmarkG2OBA ${OpenCV_LIBS} Sophus::Sophus ${G2O_LIBS} g2o_core g2o_stuff)

add_executable( icpPoseICPGNTDB icp_pose_TDB_GN_3d3d.cpp )
target_link_libraries(icpPoseICPGNTDB ${OpenCV_LIBS} Sophus::Sophus)

add_executable( icpPoseLandmarkGN icp_pose_landmark_BA_3d3d_GN.cpp )
target_link_libraries(icpPoseLandmarkGN ${OpenCV_LIBS} Sophus::Sophus)

add_executable( icpPoseCeres icp_pose_only_3d3d_Ceres.cpp )
target_link_libraries(icpPoseCeres ${OpenCV_LIBS} Sophus::Sophus ${CERES_LIBRARIES})

add_executable( icpPoseLandmarkCeres icp_pose_landmark_3d3d_Ceres.cpp )
target_link_libraries(icpPoseLandmarkCeres ${OpenCV_LIBS} Sophus::Sophus ${CERES_LIBRARIES})
