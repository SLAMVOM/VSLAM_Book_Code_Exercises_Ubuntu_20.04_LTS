# This exercise is for stereo-vision-based dense mapping and reconstruction.

## The code is mainly based on Sec. 5.4 and Sec. 12.4 in the VSLAM book 2nd Ed.

--------------------------------

Brief description of the task: <br>
Given a pair of stereo images, one from a left camera and the other from the right, we want to:
  - build a disparity map for every pixel
  - construct a dense point cloud based on the disparity map
  - apply a voxel filter (using PCL functions) over the dense point cloud to obtain a .pcd file, which can be visualized in pcl_viewer
  - build mesh (surfel mapping) from the point cloud using pcl package - result can be visualized by Meshlab
  - build an octo-map from the point cloud using octomap package - result can be visualized by octovis

--------------------------------

#### Author: MT
#### Creation Date: 2022-April-23
#### Previous Edit: 2022-April-23
