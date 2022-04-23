#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    // for octomap

#include <Eigen/Geometry>
#include <boost/format.hpp>     // for formating strings

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;   // color and depth images
    vector<Eigen::Isometry3d> poses;    // camera poses

    ifstream fin("./data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("./data/%s/%d.%s"); // image file format
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // using -1 to load the original image
    
        double data[7] = {0};
        for (int j = 0; j < 7; j++) {
            fin >> data[j];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // Calculating point cloud and merge
    // camera intrinsics
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0; // the reason for -ve fy is because the convention of the dataset set the image plane behind focal point
    double depthScale = 5000.0; // a scaling factor for the depth values

    cout << "Converting the images to Octomap ... " << endl;

    // octomap tree
    octomap::OcTree tree(0.01); // the parameter is the resolution

    for (int i = 0; i < 5; i++) {
        cout << "Converting images: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud;  // define a variable to store the point cloud in octomap

        for (int v = 0; v < color.rows; v++) 
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u];   // depth value
                if (d == 0) continue;   // if measurement is zero, then no observation
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                // adding the world coordinate of the point into point cloud
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }
        
        // putting the point cloud into the octomap, giving the origin, so that the projection line can be computed
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }

    // renew the intermediate nodes and save into a .bt file
    tree.updateInnerOccupancy();
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");
    return 0;
}