/******************************************************************************
* This script implements a direct method in the g2o framework. According to
* the classification in Gao's VSLAM book, the implemented approach is a
* semi-dense direct method.
* 
* The image gradients are calculated by using the Laplace filter in OpenCV.
*
* The depth information value of the first image is given by a disparity map.
******************************************************************************/

/* Note that in the problem, all the transformation matrices are from the first
** frame to the current frame, i.e., T_{ki}, where i is the first image frame, which
** is set to have a pose of identity.
*/

// Author: MT
// Creation Date: 2022-May-17
// Previous Edit: 2022-May-17


#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
// #include "opencv2/xfeatures2d.hpp" // comment this line if the contribute module of opencv is not installed
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>
// #include <pangolin/pangolin.h>
#include <iostream>
#include <iomanip>
// Eigen
#include <Eigen/Core>
// The following libraries are used for g2o - graph optimization
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
// Lie group utility functions
#include "lie_utils.h"

#define DETECTOR    9 // flag for different image feature key point detector: 0 - GFTT; 1 - SIFT; 2 - ORB; 9 - semi-dense direct method

#define GRAD_THRESHOLD    200 // a user-defined image gradient threshold [0-255], larger->less px selected, smaller->more px selected

using namespace std;

// Camera intrinsics for Kitti dataset
double fu = 718.856, fv = 718.856, c_u = 607.1928, c_v = 185.2157;
// baseline [m] between the left and right camera centers
double baseline = 0.573;
// paths
string left_file = "../../left.png";
string disparity_file = "../../disparity.png";
boost::format fmt_others("../../%06d.png"); // a formattable string for file name

// define several types for future use
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

std::vector<Matrix4d, Eigen::aligned_allocator<Matrix4d>> VecPoses;
std::vector<VecVector2d> VecProjections;


// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)]; // img in cv is stored row-by-row in the memory, int() takes the floor integer
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1- yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}


/** The G2O workflow
 * 1. Define vertex class(es)/structure(s)
 * 2. Define edge class(es)/structure(s)
 * 3. Setup and allocate a solver object (with a specified optimization algorithm)
 * 4. Construct the optimization graph by adding vertices and edges to the optimizer
 * 5. Pefrom optimization and return results
 */

// 1. Define vertex class(es)/structure(s)
class VertexPose : public g2o::BaseVertex<6, Matrix4d> { // g2o::BaseVertex<D,T> where D: minimal dimension of the vertex
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // override the reset function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
        virtual void setToOriginImpl() override {
            _estimate = Matrix4d::Identity();
        }

        /// left multiplication on SE3
        virtual void oplusImpl(const double *update) override {
            Vector6d update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            Matrix4d T_update; // initialize a 4x4 Eigen matrix to store the update transformation matrix
            zetaToSE3(update_eigen, T_update); // obtain the transformation from lie algebra tangential vector
            _estimate = T_update * _estimate; // left multiplication to update
        }

        virtual bool read(istream &in) override {return false;}
        virtual bool write(ostream &out) const override {return false;}
}; // VertexPose


// 2. Define edge class(es)/structure(s)
// g2o::BaseUnaryEdge<D, E, VertexXi> where D: error dimension; E: error data type
class EdgeProjection : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const cv::Mat &img1,
                   const cv::Mat &img2,
                   const Eigen::Vector2d &px1,
                   const double depth,
                   const int du, // the horizontal deviation from the central pixel of the feature point
                   const int dv // the vertical deviation from the central pixel of the feature point
                  ) :
                   img1_(img1), img2_(img2), px1_(px1), depth_(depth), du_(du), dv_(dv) {}

    // override the error computation function
    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Matrix4d T_ki = v->estimate();
        double u_1 = px1_[0];    // the col num of the feature
        double v_1 = px1_[1];    // the row num of the feature
        // inverse camera model to project the feature from pixel coordinate to 3D pt under first camera frame
        double X_1 = (u_1 - c_u) * depth_ / fu; // x coordinate of the landmark under first camera frame
        double Y_1 = (v_1 - c_v) * depth_ / fv; // y coordinate of the landmark under first camera frame
        // transform the landmark from the inertial camera frame to the current through the transformation
        Vector4d pt1_homo = {X_1, Y_1, depth_, 1.0}; // homogeneous coordinate of the landmark under the inertial frame
        Vector4d pt2_homo = T_ki * pt1_homo;       // homogeneous coordinate of the landmark under the current image frame
        // reproject the 3D landmark under the current camera frame onto the image plane and convert into pixel coordinate
        double u_2 = pt2_homo[0] * fu / pt2_homo[2] + c_u; // the col pixel coordinate of the feature on the current image
        double v_2 = pt2_homo[1] * fv / pt2_homo[2] + c_v; // the row pixel coordinate of the feature on the current image
        // Note: the photometric error is defined as I_img1(u_1, v_1) - I_img2(u_2, v_2)
        Eigen::Matrix<double, 1, 1> err; // the error term should be in a type of Eigen matrix
        // err << GetPixelValue(img1_, u_1 + du_, v_1 + dv_) - GetPixelValue(img2_, u_2 + du_, v_2 + dv_);
        err << _measurement - GetPixelValue(img2_, u_2 + du_, v_2 + dv_); // equivalent to the line above
        _error = err;
    }

    // override the linearization function
    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Matrix4d T_ki = v->estimate();
        double u_1 = px1_[0];    // the col num of the feature
        double v_1 = px1_[1];    // the row num of the feature
        // inverse camera model to project the feature from pixel coordinate to 3D pt under first camera frame
        double X_1 = (u_1 - c_u) * depth_ / fu; // x coordinate of the landmark under first camera frame
        double Y_1 = (v_1 - c_v) * depth_ / fv; // y coordinate of the landmark under first camera frame
        // transform the landmark from the inertial camera frame to the current through the transformation
        Vector4d pt1_homo = {X_1, Y_1, depth_, 1}; // homogeneous coordinate of the landmark under the inertial frame
        Vector4d pt2_homo = T_ki * pt1_homo;       // homogeneous coordinate of the landmark under the current image frame
        // reproject the 3D landmark under the current camera frame onto the image plane and convert into pixel coordinate
        double u_2 = pt2_homo[0] * fu / pt2_homo[2] + c_u; // the col pixel coordinate of the feature on the current image
        double v_2 = pt2_homo[1] * fv / pt2_homo[2] + c_v; // the row pixel coordinate of the feature on the current image

        // define some intermediate terms to aid calculation
        double X_2 = pt2_homo[0];           // X_2
        double Y_2 = pt2_homo[1];           // Y_2
        double Z_2 = pt2_homo[2];           // Z_2
        double Z_2_sq = Z_2 * Z_2;          // Z_2^2
        double inv_Z = 1.0 / Z_2;           // 1/Z_2
        double inv_Z_sq = inv_Z * inv_Z;    // 1/(Z_2)^2

        // two parts of the jacobian
        Matrix26d J_pixel_xi;
        Eigen::Vector2d J_img_pixel;

        // dy_2 / dXi; here Xi is the transformation parameters
        J_pixel_xi << fu * inv_Z,          0, -fu * X_2 * inv_Z_sq,      -fu * X_2 * Y_2 * inv_Z_sq, fu + fu * X_2 * X_2 * inv_Z_sq, -fu * Y_2 * inv_Z,
                               0, fv * inv_Z, -fv * Y_2 * inv_Z_sq, -fv - fv * Y_2 * Y_2 * inv_Z_sq,      fv * X_2 * Y_2 * inv_Z_sq,  fv * X_2 * inv_Z;

        // dI / dy_2 - image plane gradient
        J_img_pixel = Eigen::Vector2d(
            0.5 * (GetPixelValue(img2_, u_2 + du_ + 1, v_2 + dv_) - GetPixelValue(img2_, u_2 + du_ - 1, v_2 + dv_)),
            0.5 * (GetPixelValue(img2_, u_2 + du_, v_2 + dv_ + 1) - GetPixelValue(img2_, u_2 + du_, v_2 + dv_ - 1))
        );

        // Jacobian = de / dXi = - (dI / dy_2) * (dy_2 / dXi)
        _jacobianOplusXi = - J_img_pixel.transpose() * J_pixel_xi; // [1 x 6]
    }

    virtual bool read(istream &in) override {return false;}

    virtual bool write(ostream &out) const override {return false;}

private:
    const cv::Mat &img1_;
    const cv::Mat &img2_;
    const Eigen::Vector2d px1_;
    const double depth_;
    const int du_; // the horizontal deviation from the central pixel of the feature poi
    const int dv_; // the vertical deviation from the central pixel of the feature point
}; // EdgeProjection


/**
* pose estimation using direct method with multi-layer scaling
* @param img1 - the first image
* @param img2 - the second image
* @param px_ref
* @param depth_ref
* @param T21 - the transformation matrix from frame 1 to 2
*/
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Matrix4d &T21
);


/**
* pose estimation using direct method
* @param img1 - the first image
* @param img2 - the second image
* @param px_ref
* @param depth_ref
* @param T21 - the transformation matrix from frame 1 to 2
*/
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Matrix4d &T21
);


//// main function
int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // randomly pick pixels in the first image and generate corresponding 3d points in the first camera frame
    cv::RNG rng;
    int nPoints = 2000; // randomly select 2000 points
    int boarder = 20;
    bool use_detector = false; // a flag to deterine a feature detector is to be used
    VecVector2d pixels_ref;
    std::vector<double> depth_ref;

    #if (DETECTOR == 0) // 0 - Good feature to tract
        cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(nPoints);
        use_detector = true;
    #elif (DETECTOR == 1) // 1 - SIFT
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(nPoints);
        use_detector = true;
    #elif (DETECTOR == 2) // 2 - ORB - Oriented FAST
        cv::Ptr<cv::ORB> detector = cv::ORB::create(nPoints);
        use_detector = true;
    #elif (DETECTOR == 9)
        // Declare the variables to be used
        int kernel_size = 3;
        int scale = 1; // scale factor for the computed Laplacian values [optional]
        int delta = 0; // delta value to be added to the results before storing into dst image [optional]
        int ddepth = CV_16S; // single-channel 2-byte singed integer
        const char* window_name = "Laplace kernel filtered";
        use_detector = false;
    #else // if any other number specified, using random pixel locations
        use_detector = false;
    #endif


    #if (DETECTOR == 0 || DETECTOR == 1 || DETECTOR == 2) // if a feature detector is specified to be used
        std::vector<cv::KeyPoint> keypoints_1; // a container to store the detected keypoints
        detector->detect(left_img(cv::Range(boarder, left_img.rows - boarder),  // rows first in cv::Mat
                                  cv::Range(boarder, left_img.cols - boarder)), // cols second in cv::Mat
                         keypoints_1);
        for (auto &kp : keypoints_1) {
            int x = kp.pt.x + boarder; // since we sent a sub-image to detector, needs to add back the boarder
            int y = kp.pt.y + boarder; // since we sent a sub-image to detector, needs to add back the boarder
            int disparity = disparity_img.at<uchar>(y, x);
            double depth = fu * baseline / disparity; // converting disparity value to depth
            depth_ref.push_back(depth);
            pixels_ref.push_back(Eigen::Vector2d(x, y));
        }
    #elif ( DETECTOR == 9 ) // if using semi-dense method
        cv::Mat abs_grad_left, img_grad_left;
        cv::Laplacian( left_img, img_grad_left, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT );
        // coverting back to CV_8U - [0, 255]
        cv::convertScaleAbs( img_grad_left, abs_grad_left ); // implemented as: scaling->taking absolute val->casting to uint8
        cv::imshow(window_name, abs_grad_left);
        cv::waitKey(0);

        double focal_times_baseline = fu * baseline;
        int N_r = abs_grad_left.rows - boarder - 1, N_c = abs_grad_left.cols - boarder - 1;

        int max_grad = -1;
        // looping through the gradient image to find out all the pixels that have a image gradient more than GRAD_THRESHOLD
        for (int y = boarder; y < N_r; y++)
            for (int x = boarder; x < N_c; x++) {
                if (abs_grad_left.at<uchar>(y, x) >= GRAD_THRESHOLD) {
                    int disparity = disparity_img.at<uchar>(y, x);
                    double depth = focal_times_baseline / disparity; // converting disparity value to depth
                    depth_ref.push_back(depth);
                    pixels_ref.push_back(Eigen::Vector2d(x, y));
                }
            }
        cout << "\nThere are " << pixels_ref.size() << " pixels with a gradient greater than GRAD_THRESHOLD.\n" << endl; 
    #else  // if neither valid feature detector nor semi-dense method is specified
        // sampling pixel locations in the ref image and loading depth data from disparity map
        for (int i = 0; i < nPoints; i++) {
            int x = rng.uniform(boarder, left_img.cols - boarder); // avoid picking pixels that are close to the boarder
            int y = rng.uniform(boarder, left_img.rows - boarder); // avoid picking pixels that are close to the boarder
            int disparity = disparity_img.at<uchar>(y, x);
            double depth = fu * baseline / disparity; // converting disparity value to depth
            depth_ref.push_back(depth);
            pixels_ref.push_back(Eigen::Vector2d(x, y));
        }
    #endif

    // estimate 01~05.png's pose using the sampled pixel locations
    Matrix4d T_cur_ref = Matrix4d::Identity(); // the reference frame is the first frame

    for (int i = 1; i < 6; i++) {
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // multi-layer direct method
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }

    // print out the stored poses
    std::cout << "\n" << std::endl;
    for (int i = 1; i < 6; i++) {
        std::cout << "Pose " << i <<" is:\n" << VecPoses[i-1] << "\n"<< std::endl;
    }

    return 0;
}


void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Matrix4d &T21
) {

    // 3. Setup and allocate a solver object (with a specified optimization algorithm)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // poses 6D, landmarks 3D
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type, PoseMatrixType defined in "g2o/core/block_solver.h"
    // using one of the gradient methods
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // Graph model
    optimizer.setAlgorithm(solver); // Setup the solver
    optimizer.setVerbose(true);     // Turn on verbose output for debugging

    // 4. Construct the optimization graph by adding vertices and edges to the optimizer
    // 4.1 adding vertiices into the graph
    // In this case, we have only one camera pose to be optimized
    VertexPose *v_cam = new VertexPose();
    v_cam->setEstimate(T21); // the setEstimate() method can be found at: "g2o/core/base_vertex.h"
    v_cam->setId(0); // the _id member is under "g2o/core/parameter.h", and the setId() method is defined at "g2o/core/optimizable_graph.h"

    optimizer.addVertex(v_cam); // adding the vertex into the optimization graph

    // define some parameters to be used during the optimization
    const int iterations = 10;
    const int half_patch_size = 1; // the half size of the square small patch around a pixel location
    int cnt_good = 0;              // number of inliers used in the optimization
    VecVector2d projections; // a storage variable to store the features to be shown

    // traverse through all the feature pixel locations and determine if the fulfill certain filters, yes->add into graph
    for (int i = 0; i < px_ref.size(); i++) {

        projections.push_back(Eigen::Vector2d(0.0, 0.0)); // allocate space for the features to be plotted

        // compute the landmark projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - c_u) / fu, (px_ref[i][1] - c_v) / fv, 1);
        Vector4d point_cur = T21 * Vector4d(point_ref[0], point_ref[1], point_ref[2], 1);

        if (point_cur[2] < 0) // depth invalid
            continue;

        float u = fu * point_cur[0] / point_cur[2] + c_u; // horizontal coordinate on image 2
        float v = fv * point_cur[1] / point_cur[2] + c_v; // vertical coordinate on image 2
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size || v > img2.rows - half_patch_size)
            continue;

        projections.back() = Eigen::Vector2d(u, v); // if a feature passes the filtering condition, treat it as an inlier
        cnt_good++; // keep a record on the number of inliers identified

        // adding the inliers as edges into the graph
        // Here, we calculate the difference of an image patch around a feature pixel, thus each pixel of a patch
        // will form an edge in the graph optimization.
        for (int du = -half_patch_size; du <= half_patch_size; du++)
            for (int dv = -half_patch_size; dv <=half_patch_size; dv++) {
                Eigen::Vector2d px1_pt(px_ref[i][0], px_ref[i][1]);
                EdgeProjection *edge = new EdgeProjection(img1, img2,           // input images
                                                          px1_pt, depth_ref[i], // input point informaiton
                                                          du, dv                // input deviations from the central pixel
                                                          );
                edge->setId(cnt_good);
                edge->setVertex(0, v_cam); // the vertex to be optimize is the camera pose
                edge->setMeasurement(GetPixelValue(img1, px_ref[i][0]+du, px_ref[i][1]+dv));
                edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber()); // using a robust kernel, can comment this line to not use
                optimizer.addEdge(edge);
            }
    }
    VecProjections.push_back(projections); // the points stored are the inlier used in the optimization problem

    // 5. Pefrom optimization and return results
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(false); // true to show intermediate optimization info
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization time used: " << time_used.count() << " seconds." << endl;
    cout << "T_ki pose estimated by g2o =\n" << v_cam->estimate() << endl;
    T21 = v_cam->estimate(); // set the T21 matrix to be the optimized pose


    // plot the projected pixels
    cv::Mat img2_show;
    // cv::cvtColor(img2, img2_show, CV_GRAY2BGR); // initial call of cv::cvtColor from book
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR); // change the data format to OpenCV4 compatible format
    VecVector2d projection_img2 = VecProjections.back();
    for (size_t i = 0; i < px_ref.size(); i++) {
        auto p_ref = px_ref[i];
        auto p_cur = projection_img2[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("current", img2_show);
    cv::waitKey();
} // end of DirectPoseEstimationSingleLayer()


void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Matrix4d &T21) {

    // parameters for multi-layer scaling
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fuG = fu, fvG = fv, cuG = c_u, cvG = c_v; // store the initial values of camera settings at the original scale
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in the current pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fu, fv, c_u, c_v by the same factor as the image in different pyramid levels
        fu = fuG * scales[level];
        fv = fvG * scales[level];
        c_u = cuG * scales[level];
        c_v = cvG * scales[level];
        // Note: each scale level is a stand alone factor graph optimization problem
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
    VecPoses.push_back(T21);
} // end of DirectPoseEstimationMultiLayer()
