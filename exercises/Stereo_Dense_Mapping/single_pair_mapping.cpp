// Author: MT
// Creation Date: 2022-April-23
// Previous Edit: 2022-April-23

// Referneces:
//  - Ch 5 stereoVision.cpp
//  - Ch 12 pointcloud_mapping.cpp and surfel_mapping.cpp and octomap_mapping.cpp

#include <vector>
#include <string>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <boost/format.hpp> // for formating stings

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
// Pangolin
#include <pangolin/pangolin.h>
// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/io/ply_io.h>
// Octomap
#include <octomap/octomap.h>

using namespace std;
using namespace Eigen;

// file directory - the pair of images are taken from the KITTI dataset
string left_file = "../left_color.png";
string right_file = "../right_color.png";

// show the point cloud in Pangolin
void showPointCloud( const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud );

// save point cloud to PLY file, a PLY file for inspection in Meshlab or CloudCompare
void WriteCloudToPLYFile(const std::string &filename,
                         const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

// pcl surfel mapping
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudPCL;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

SurfelCloudPtr reconstructSurface(
        const PointCloudPtr &input, float radius, int polynomial_order) {
    pcl::MovingLeastSquares<PointT, SurfelT> mls;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(radius);
    mls.setComputeNormals(true);
    mls.setSqrGaussParam(radius * radius);
    mls.setPolynomialOrder(polynomial_order);
    mls.setInputCloud(input);
    SurfelCloudPtr output(new SurfelCloud);
    mls.process(*output);
    return output;
}

pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels) {
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>);
    tree->setInputCloud(surfels);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3;
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh);

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.05);

    // Set typical values for the parameters
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2* M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(true);

    // Get result
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);

    return triangles;
}


int main(int argc, char **argv) {
    // camera intrinsics for the KITTI dataset
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185/2157;
    // baseline
    double b = 0.573;

    // read the image as grayscale image by using flag 0
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        8, 80, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32); // tuned parameters for good results, SGBM - sensitive to params
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0/16.0f);

    // generating point cloud
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    for (int v = 0; v < left.rows; v++) {
        for (int u = 0; u < left.cols; u++) {
            // medium filter, avoid the points that are either too far or too close to the camera, there are error-prone pts
            if (disparity.at<float>(v, u) <= 8.0 || disparity .at<float>(v, u) >= 96.0 ) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // first 3D are xyz, last val is gray color intensity

            // compute the depth and point positions based on disparity using the stereo camera model
            double x = (u - cx) / fx; // convert from pixel coordinate to normalized image plane coor.
            double y = (v - cy) / fy; // convert from pixel coordinate to normalized image plane coor.
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth; // from normalized image plane coor. to camera frame coordinate
            point[1] = y * depth; // from normalized image plane coor. to camera frame coordinate
            point[2] = depth;

            pointcloud.push_back(point); // record the point position in the camera frame coor. system
        }
    }

    cv::imshow("disparity", disparity / 96.0); // val range [0,1] the 96 here is (max - min disparity), see cv SGBM
    cv::waitKey(0);
    // show the point cloud
    showPointCloud(pointcloud);
    std::cout << "Start saving point cloud to a PLY file" << std::endl;
    WriteCloudToPLYFile("pointcloud.ply", pointcloud);
    std::cout << "Finished saving point cloud to a PLY file\n" << std::endl;


    ///////// Creating a point cloud using pcl, format of point cloud XYZRGB /////////
    
    // creating a new point cloud - for more than one pair of images, see pointcloud_mapping.cpp in VSLAM book Ch12
    PointCloudPCL::Ptr pclCloud(new PointCloudPCL);
    PointCloudPCL::Ptr current(new PointCloudPCL);
    // Read the image as color image for creating point cloud with color
    cv::Mat color = cv::imread(left_file, 1); // flag 1 for cv::IMREAD_COLOR - three channels BGR

    for (int v = 0; v < color.rows; v++) {
        for (int u = 0; u < color.cols; u++) {
            // medium filter, avoid the points that are either too far or too close to the camera, there are error-prone pts
            if (disparity.at<float>(v, u) <= 8.0 || disparity .at<float>(v, u) >= 96.0 ) continue;
            
            // compute the depth and point positions based on disparity using the stereo camera model
            double x = (u - cx) / fx; // convert from pixel coordinate to normalized image plane coor.
            double y = (v - cy) / fy; // convert from pixel coordinate to normalized image plane coor.
            double depth = fx * b / (disparity.at<float>(v, u));

            Eigen::Vector3d pt;
            pt[0] = x * depth; // from normalized image plane coor. to camera frame coordinate
            pt[1] = y * depth; // from normalized image plane coor. to camera frame coordinate
            pt[2] = depth;

            // Note: for more than one pair of images, need to consider the transformation of the landmark to ref frame
            PointT p;
            p.x = pt[0];
            p.y = pt[1];
            p.z = pt[2];
            p.b = color.data[v * color.step + u * color.channels()]; // for grayscale pt cloud, all bgr indices to be the same
            p.g = color.data[v * color.step + u * color.channels() + 1];
            p.r = color.data[v * color.step + u * color.channels() + 2];
            current->points.push_back(p);
        }
    }
    // depth filter and statistical removal
    PointCloudPCL::Ptr tmp(new PointCloudPCL);
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(current);
    statistical_filter.filter(*tmp);
    (*pclCloud) += *tmp;

    pclCloud->is_dense = false; // True if NO points are invalid (e.g.,invalid means having NaN or Inf values in any of point fields)
    std::cout << "There are total " << pclCloud->size() << " points in the pcl point cloud." << std::endl;
    // saving the pcl point cloud before applying voxel filter
    pcl::io::savePCDFileASCII("PCL_pointcloud_without_Voxel.pcd", *pclCloud);

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.05; // the resolution of each voxel, the smaller the more voxels
    voxel_filter.setLeafSize(resolution, resolution, resolution); // resolution of each 3D voxel grid (box)
    PointCloudPCL::Ptr tmp_cloud(new PointCloudPCL); // define a temporary point cloud
    voxel_filter.setInputCloud(pclCloud); // set the point cloud to be filtered
    voxel_filter.filter(*tmp_cloud); // applying the filter over the point cloud and store the output to tmp_cloud
    tmp_cloud->swap(*pclCloud); // swap the point cloud stored in tmp with the cloud stored in pclCloud

    std::cout << "\nAfter filtering, the pcl point cloud has total " << pclCloud->size() << " points." << std::endl;

    pcl::io::savePCDFileBinary("PCL_pointcloud_with_Voxel.pcd", *pclCloud); // save into a bianry format
    std::cout << "Saved PCL point cloud after applying a voxel filter.\n" << std::endl;

    // Compare surface elements
    std::cout << "computing normals ... " << std::endl;
    double mls_radius = 2.0, polynomial_order = 2;
    auto surfels = reconstructSurface(pclCloud, mls_radius, polynomial_order);

    // Compute a greedy surface triangulation
    std::cout << "computing mesh ... " << std::endl;
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels);

    std::cout << "display mesh ... " << std::endl;
    pcl::visualization::PCLVisualizer vis;
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame");
    vis.addPolygonMesh(*mesh, "mesh");
    vis.resetCamera();
    vis.spin();

    pcl::io::savePLYFile("PCL_polygon_mesh.ply", *mesh); // save into a bianry format
    std::cout << "Saved PCL polygon mesh. \n" << std::endl;


    ///////// Generating an octomap based on the point cloud /////////
    std::cout << "Generating octomap" << std::endl;

    octomap::OcTree tree(0.05); // the parameter is the resolutoin

    octomap::Pointcloud cloud_oct; // define a variable to store the point cloud in octomap

    // using the most basic octree map without color information
    // for more than one pair of images, see octomap_mapping.cpp in VSLAM book Ch12
    for (int v = 0; v < color.rows; v++) {
        for (int u = 0; u < color.cols; u++) {
            // medium filter, avoid the points that are either too far or too close to the camera, there are error-prone pts
            if (disparity.at<float>(v, u) <= 8.0 || disparity .at<float>(v, u) >= 96.0 ) continue;
            
            // compute the depth and point positions based on disparity using the stereo camera model
            double x = (u - cx) / fx; // convert from pixel coordinate to normalized image plane coor.
            double y = (v - cy) / fy; // convert from pixel coordinate to normalized image plane coor.
            double depth = fx * b / (disparity.at<float>(v, u));

            Eigen::Vector3d pt;
            pt[0] = x * depth; // from normalized image plane coor. to camera frame coordinate
            pt[1] = y * depth; // from normalized image plane coor. to camera frame coordinate
            pt[2] = depth;
            // storing the point into octomap point cloud
            cloud_oct.push_back(pt[0], pt[1], pt[2]);
        }
    }
    // putting the point cloud into the octomap, giving the origin, so that the projection line can be computed
    tree.insertPointCloud(cloud_oct, octomap::point3d(0.0, 0.0, 0.0));

    // renew the intermediate nodes and save into a .bt file
    tree.updateInnerOccupancy();
    std::cout << "Saving octomap ... " << std::endl;
    tree.writeBinary("octomap.bt");
    return 0;
}


void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {
    
    if (pointcloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768); // width, height
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0.2, -0.5, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    // ModelViewLookAt: from (x1,y1,z1) to look at the origin (x2,y2,z2), (x3,y3,z3) indicates the up direction
    // the inherent axes: z into page, y to up, x to left

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]); // grayscale - three channels with the same intensity
            glVertex3d(p[0], p[1], p[2]);
        }

        glEnd();
        pangolin::FinishFrame();
        usleep(5000); // sleep for 5 ms
    }
    return;
}


void WriteCloudToPLYFile(const std::string &filename,
                         const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << pointcloud.size()
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // Save the structure (i.e., 3D points) as grayscale points
    for (int i = 0; i < pointcloud.size(); i++) {
        of << pointcloud[i][0] << ' ' << pointcloud[i][1] << ' ' << pointcloud[i][2] << ' '
           << (int)(pointcloud[i][3] * 255) << ' ' << (int)(pointcloud[i][3] * 255) << ' '  << (int)(pointcloud[i][3] * 255)
           << '\n';
    }

    of.close();
}
