#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

// file directory
string left_file = "../left.png";
string right_file = "../right.png";

// show the point cloud in Pangolin
void showPointCloud( const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud );

int main(int argc, char **argv) {
    // camera intrinsics
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185/2157;
    // baseline
    double b = 0.573;

    // read the image as grayscale image by using flag 0
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32); // tuned parameters for good results, SGBM - sensitive to params
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0/16.0f);

    // generating point cloud
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    for (int v = 0; v < left.rows; v++){
        for (int u = 0; u < left.cols; u++) {
            // medium filter, avoid the points that are either too far or too close to the camera, there are error-prone pts
            if (disparity.at<float>(v, u) <= 0.0 || disparity .at<float>(v, u) >= 96.0 ) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // first 3 are xyz, last val is intensity

            // compute the depth and point positions based on disparity using the stereo camera model
            double x = (u - cx) / fx; // convert from pixel coordinate to normalized image plane coor
            double y = (v - cy) / fy; // convert from pixel coordinate to normalized image plane coor
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point); // record the point position in the normalized image plane coor. system
        }
    }

    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // show the point cloud
    showPointCloud(pointcloud);
    //cv::waitKey(0);
    return 0;
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin:: CreateWindowAndBind("Point Cloud Viewer", 1024, 768); // width , height
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
        usleep(5000); // sleep 5 ms
    }
    return;
}