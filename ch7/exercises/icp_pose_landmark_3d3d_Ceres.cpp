#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <chrono>
#include <sophus/se3.hpp>

#include <ceres/ceres.h>
#include "rotation.h"

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// Converting pixel coordinates to normalized camera frame coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

// Define two typea to store data points
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

void pose_estimation_3d3d(
    const vector<Point3f> &pts1,
    const vector<Point3f> &pts2,
    Mat &R, Mat &t
);

void bundleAdjustmentCeres(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    // Mat &R, Mat &t,
    Sophus::SE3d &pose,
    double *points_3d_ba
);

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    // -- Loading images
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Number of feature correspondences found in total " << matches.size() << endl;

    // Creating 3D points
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // depth image values are single-channel 16-bits unsigned int
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED); // depth image values are single-channel 16-bits unsigned int

    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    for (DMatch m:matches) { // note that: x for columns and y for rows
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) // no depth information available
            continue;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3d pairs: " << pts1.size() << endl;
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = \n" << R << endl;
    cout << "t = \n" << t << endl;
    cout << "R_inv = \n" << R.t() << endl;
    cout << "t_inv = \n" << -R.t() * t << endl;

    //////////////// Ceres BA ////////////////
    cout << "Calling bundle adjustment by Ceres" << endl;
    
    // Store the feature keypoints into Eigen vector containers
    VecVector3d pts1_3d_eigen;
    VecVector3d pts2_3d_eigen;

    // ravel the pts1_3d_eigen to make a double array for updating the landmarks
    double *pts1_3d_ba;
    pts1_3d_ba = new double[3 * pts1.size()];

    for (size_t i = 0; i < pts1.size(); i++) {
        pts1_3d_eigen.push_back(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        pts2_3d_eigen.push_back(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        
        pts1_3d_ba[3*i+0] = pts1_3d_eigen[i][0];
        pts1_3d_ba[3*i+1] = pts1_3d_eigen[i][1];
        pts1_3d_ba[3*i+2] = pts1_3d_eigen[i][2];
    }
    
    // Sophus::SE3d pose_ceres;
    Eigen::Matrix3d R_eigen;
    R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
               R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
               R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    Sophus::SE3d pose_ceres(R_eigen, t_eigen); /////////
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    bundleAdjustmentCeres(pts1_3d_eigen, pts2_3d_eigen, pose_ceres, pts1_3d_ba);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by solving ICP in Ceres: " << time_used.count() << " seconds." << endl;

    //verify p1 = R * p2 + t, here p1 and p2 are individual feature points
    for (int i = 0; i < 5; i++) { // show the first three points
        Eigen::Matrix<double,4,1> pt_tmp;
        pt_tmp << pts2[i].x, pts2[i].y, pts2[i].z, 1;
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "p1_ba[" << pts1_3d_ba[3*i+0] << ", " << pts1_3d_ba[3*i+1] << ", " << pts1_3d_ba[3*i+2] << "]" << endl;
        cout << "(R * p2 + t) =\n" << 
            R * (Mat_<double>(3,1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << "\n\n" << // using this or below line
            pose_ceres.matrix().inverse() * pt_tmp  // using this or above line
            << endl;
        cout << endl;
    }

    double err_cv = 0.0, err_ceres = 0.0;
    Eigen::Matrix3d R_SVD;
    R_SVD << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
             R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
             R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    Eigen::Matrix<double, 3, 1> t_SVD;
    t_SVD << t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);
    for (int i = 0; i < pts1.size(); i++) {
        Eigen::Matrix<double,4,1> pt_tmp;
        pt_tmp << pts2[i].x, pts2[i].y, pts2[i].z, 1;
        pt_tmp = pose_ceres.matrix().inverse() * pt_tmp;
        err_ceres += ( (pt_tmp[0] - pts1_3d_ba[3*i+0])*(pt_tmp[0] - pts1_3d_ba[3*i+0]) +
                       (pt_tmp[1] - pts1_3d_ba[3*i+1])*(pt_tmp[1] - pts1_3d_ba[3*i+1]) +
                       (pt_tmp[2] - pts1_3d_ba[3*i+2])*(pt_tmp[2] - pts1_3d_ba[3*i+2]));
        
        Eigen::Vector3d cv_tmp;
        cv_tmp << pts2[i].x, pts2[i].y, pts2[i].z;
        cv_tmp = R_SVD * cv_tmp + t_SVD;
        err_cv += ( (cv_tmp[0] - pts1[i].x)*(cv_tmp[0] - pts1[i].x) +
                    (cv_tmp[1] - pts1[i].y)*(cv_tmp[1] - pts1[i].y) +
                    (cv_tmp[2] - pts1[i].z)*(cv_tmp[2] - pts1[i].z));
    }

    cout << "SVD point error: " << err_cv << "\t" << "ceres BA point error: " << err_ceres << endl;

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    // -- Initialization
    Mat descriptors_1, descriptors_2;
    // using the ORB feature in OpenCV
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // brute-force matching of descriptors using hamming distance
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    // -- Step 1: detecting oriented FAST corner keypoint locations
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // -- Step 2: computing BRIEF descriptors based on the detected keypoints
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    
    // -- Step 3: matching the BRIEF descriptors between the two images using hamming distance
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // -- Step 4: filtering the features
    double min_dist = 10000, max_dist = 0;

    // find out the min and max distance among the matched correspondences
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    // when the distance between two descriptors is greater than two times of the min_dist,
    // we classify it as a mismatch. However, the min_dist can be quite small sometimes,
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

// SVD method to compute linear solution for rotation and translation
void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
    Point3f p1, p2; // centroid, or center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N); // point coordinates after subtracting the centroids
    for (int i = 0 ; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W = \n" << W << endl;

    // SVD on W, need to ask for U and V matrices explicitly
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U = \n" << U << endl;
    cout << "V = \n" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) { // if the determinant of the R_ matrix is negative 1, then use its negative as rotation
        R_ = -R_;
    }
    // once obtained the rotation matrix, compute the translation vector
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // Convert from Eigen matrix to cv::Mat
    R = (Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

// a struct defining the residual terms
struct EuclideanError{
public:
    EuclideanError(Eigen::Vector3d pt2) : _pt2(pt2) {}

    // a template for operator
    template<typename T>
    bool operator()(const T *const camera, const T *const pt1, T *residuals) const {
        T predictions[3]; // transformed pt coordinates in the current frame

        // camera[0,1,2] are the angle-axis rotation
        // Rodrigues' formula - defined in "rotation.h"
        AngleAxisRotatePoint(camera, pt1, predictions); // passing in camera, only the first three elements are used
        predictions[0] += camera[3];
        predictions[1] += camera[4];
        predictions[2] += camera[5];

        residuals[0] = predictions[0] - T(_pt2[0]);
        residuals[1] = predictions[1] - T(_pt2[1]);
        residuals[2] = predictions[2] - T(_pt2[2]);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d pt2) {
        return (new ceres::AutoDiffCostFunction<EuclideanError, 3, 6, 3>( // 3D error; 6D pose
            new EuclideanError(pt2)));
    }

    private:
        Eigen::Vector3d _pt2;
}; // EuclideanError

void bundleAdjustmentCeres(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    // Mat &R, Mat &t,
    Sophus::SE3d &pose,
    double *points_3d_ba
) {
    // convert Sophus::SE3 to Eigen, then store parameters in a correct order
    Vector6d pose_vec = pose.log(); // Note: in Sophus, translation at front, rotation at back
    double camera[6] = {pose_vec[3], pose_vec[4], pose_vec[5], pose_vec[0], pose_vec[1], pose_vec[2]};

    // create a Ceres problem
    ceres::Problem problem;
    for (size_t i = 0; i < pts1.size(); i++) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes the point coordinates in the ref and cur frame and camera poses as input
        // and outputs a 3-dimensional Residual
        cost_function = EuclideanError::Create(pts2[i]);

        // create a pointer pointing to the ith landmark's address in the array
        double *pt3d = &(points_3d_ba[i * 3]);

        // If enabled, use Huber's robust loss function
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Adding residual block to ceres problem
        problem.AddResidualBlock(cost_function, loss_function, camera, pt3d); // with robust function
        // problem.AddResidualBlock(cost_function, nullptr, camera, pt3d); // without robust function
    }

    // setup the ceres solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // solver types: SPARSE_SCHUR, DENSE_SCHUR, DENSE_QR, DENSE_NORMAL_CHOLESKY, SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

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