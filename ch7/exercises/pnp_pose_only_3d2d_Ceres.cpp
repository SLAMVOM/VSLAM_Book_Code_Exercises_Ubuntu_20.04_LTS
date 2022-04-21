/*********************************************************
* This script includes solving the PnP problem using Ceres
*********************************************************/

#include <iostream>
#include <iomanip>
#include <chrono> // for timing the runnning time of the three methods
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// The following libraries are used if solving the BA by explictly implementating it
#include <Eigen/Core>
#include <sophus/se3.hpp>
// Ceres libraries
#include <ceres/ceres.h>
#include "rotation.h"

using namespace std;
using namespace cv;

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// Converting pixel coordinates to normalized camera frame coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

// Define two types to store data points
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

typedef Eigen::Matrix<double, 6, 1> Vector6d;

// BA by Ceres
void bundleAdjustmentCeres(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);


int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "usage: pnpPoseCeres img1 img2 depth1 depth2" << std::endl;
        return 1;
    }

    // -- Loading images
    // Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    // Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR); // OpenCV 4
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR); // OpenCV 4
    assert(img_1.data & img_2.data && "Cannot load the images!");

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "Number of feature correspondences found in total: " << matches.size() << std::endl;

    // Construct 3D points
    // Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 3
    Mat d1 = imread(argv[3], cv::IMREAD_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 4
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) // 0 indicates no depth information available
            continue;
        float dd = d / 5000.0; // dividing the depth value by a scalar value of 5000
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    cv::solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // using OpenCV's PnP solver, option for EPNP, DLS or other methods
    Mat R;
    cv::Rodrigues(r, R); // r is a rotation vector, so using the Rodrigues formula to convert it into a matrix form
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by solving PnP in OpenCV: " << time_used.count() << " seconds." << endl;

    cout << "R = " << endl << R << endl;
    cout << "t = " << endl << t << endl;

    // Store the feature keypoints into Eigen vector containers
    // Note that in the PnP problem, one set of points are with known depth,
    // while the other set only knows the image plane coordinates
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); i++) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "Calling bundle adjustment by Ceres" << endl;
    Sophus::SE3d pose_ceres;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentCeres(pts_3d_eigen, pts_2d_eigen, K, pose_ceres);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by solving PnP in Ceres: " << time_used.count() << " seconds." << endl;

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    // -- Initialization
    Mat descriptors_1, descriptors_2;
    // using the ORB features in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // Using the brute-force method with hamming distance for feature matching
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    // -- Step 1: Detect Oriented FAST corner keypoint locations
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // -- Step 2: Computing the BRIEF descriptor based on the detected keypoints
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // -- Step 3: Matching the BRIEF descriptors between the two images using the Hamming distance
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // -- Step 4: Filtering the paired feature correspondences
    double min_dist = 10000, max_dist = 0;

    // find out the min and max distance among all the feature paris
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    // when the distance between two descriptors is greater than two times of the min_dist,
    // we classify it as a mismatch. However, the min_dist can be small sometimes,
    // so we set a lower-bound threshold of 30 based on experience.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

// a struct defining the residual terms
// In this question, it is assumed that the 
// camera intrisics are given, thus fixed
struct ReprojectionError {
public:
    ReprojectionError(double observation_x, double observation_y, 
                      double fx, double fy, double cx, double cy,
                      double  pt_x, double pt_y, double pt_z)
        : observed_x(observation_x), observed_y(observation_y),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          pt_x_(pt_x), pt_y_(pt_y), pt_z_(pt_z) {};

    // a template for operator
    template<typename T>
    bool operator()(const T *const camera, T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        
        // Rodrigues' formula
        T p[3];
        
        T p_world[3];
        p_world[0] = T(pt_x_);
        p_world[1] = T(pt_y_);
        p_world[2] = T(pt_z_);
        AngleAxisRotatePoint(camera, p_world, p); // passing in camera, only the first three elements are used
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        predictions[0] = p[0] / p[2] * T(fx_) + T(cx_); // pixel coordinate x
        predictions[1] = p[1] / p[2] * T(fy_) + T(cy_); // pixel coordinate y
        
        
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        
        return true;
    }

    // // INPUT camera : 6 dims array
    // //         [0-2]: angle-axis rotation
    // //         [3-5]: translation
    // // OUTPUT predictions: 2D predictions
    // template<typename T>
    // bool CamProjection(const T *camera, T *predictions) {
    //     // Rodrigues' formula
    //     T p[3];
    //     Eigen::Vector3d point_( T(pt_x_), T(pt_y_), T(pt_z_) );
    //     AngleAxisRotatePoint(camera, point_, p); // passing in camera, only the first three elements are used
    //     // camera[3,4,5] are the translation
    //     p[0] += camera[3];
    //     p[1] += camera[4];
    //     p[2] += camera[5];

    //     predictions[0] = p[0] / p[2] * T(fx_) + T(cx_); // pixel coordinate x
    //     predictions[1] = p[1] / p[2] * T(fy_) + T(cy_); // pixel coordinate y

    //     return true;
    // }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y, 
                                       const double fx_, const double fy_, const double cx_, const double cy_,
                                       const double  pt_x_, const double pt_y_, const double pt_z_) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>( // 2D error; 6D pose
            new ReprojectionError(observed_x, observed_y, fx_, fy_, cx_, cy_, pt_x_, pt_y_, pt_z_)));
    }
    
    private:
        double observed_x;
        double observed_y;
        double fx_, fy_, cx_, cy_;
        double pt_x_, pt_y_, pt_z_;
}; // ReprojectionError

// BA by Ceres
void bundleAdjustmentCeres(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {
    
    // Extract the camera intrinsics
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    // convert Sophus::SE3 to Eigen, then store parameters in a correct order
    Vector6d pose_vec = pose.log(); // Note: in Sophus, translation at front, rotation at the back
    double camera[6] = {pose_vec[3], pose_vec[4], pose_vec[5], pose_vec[0], pose_vec[1], pose_vec[2]};
    
    // create a Ceres problem
    ceres::Problem problem;
    for (size_t i = 0; i < points_3d.size(); i++) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes 2D point's x & y, intrinsics and camera poses as input
        // and outputs a 2-dimensional Residual
        cost_function = ReprojectionError::Create(points_2d[i][0], points_2d[i][1], 
                                                  fx, fy, cx, cy,
                                                  points_3d[i][0], points_3d[i][1], points_3d[i][2]);

        // // If enabled use Huber's robust loss function
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Adding residual block to ceres problem
        problem.AddResidualBlock(cost_function, loss_function, camera); // with robust function
        // problem.AddResidualBlock(cost_function, nullptr, camera); // without robust function
    }

    // setup the ceres solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // solver types: SPARSE_SCHUR, DENSE_SCHUR, DENSE_QR, DENSE_NORMAL_CHOLESKY, SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    pose_vec.setZero();
    pose_vec(0,0) = camera[3];
    pose_vec(1,0) = camera[4];
    pose_vec(2,0) = camera[5];
    pose_vec(3,0) = camera[0];
    pose_vec(4,0) = camera[1];
    pose_vec(5,0) = camera[2];
    pose = Sophus::SE3d::exp(pose_vec);

    cout << "pose by Ceres: \n" << pose.matrix() << endl;
}