/******************************************************************************
* This script includes solving the ICP BA problem with options to optimize
* pose only, landmark only, or both by using the g2o package that
* formulates the problem as a graph optimization.
******************************************************************************/

/********************************************************************
 * Note that in a two-view ICP setting, if we set one of the frames
 * as an inertia (or world) reference frame, while (assuming) all the
 * landmarks are observed by the two cameras. In such case, if we
 * try to optimize the 6D pose and all the landmarks coordinates in
 * the reference frame together. The problem will be ill-posed,
 * because each landmark can only give three constraints/conditions.
 * At the same time we have 3N variables for all the landmarks plus
 * 6 variables for the camera pose of the moving frame.
 * Therefore, we have (3N + 6) variables but have only 3N conditions.
 * There are less constraints than unknown, thus the problem is
 * ill-posed. We can fix (more than) two landmarks and only optimize
 * the pose and the other landmarks to make the system to give an
 * either unique or overdetermined solution.
********************************************************************/

/********************************************************************
 * Notice the formulation of the problem is transforming the point
 * cloud in the second (moving) frame into the first (world) frame,
 * then using the point cloud in the ref frame to subtract the
 * transformed cloud to obtain the error terms. If we want to
 * also optimize the landmark point coordinates, then the things to
 * be updated are: the 6 DoF transformation from the second frame to
 * the world reference frame, T_{wc}; and the 3D landmark coordinates
 * of the points in the FIRST world reference frame.
 * Again, the points to be changed are those in the reference frame!
********************************************************************/

// Author: MT
// Creation Date: 2022-April-27
// Previous Edit: 2022-April-27

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
// The following libraries are used if solving the BA by g2o - graph optimization
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;
using namespace cv;

//////////////// Set the following flag to different values to perform different optimization tasks
#define OPTIMIZATION_MODE   0   // 0 - pose only; 1 - landmarks only; 2 - both pose and landmarks

// Searching for feature matches between image pair
void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// Converting pixel coordinates to normalized image plane coordinates
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat&K); // return the first two dimensions since the last is always 1

typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// Pose estimation using the SVD method
void pose_estimation_3d3d(
    const vector<Point3f> &pts1,
    const vector<Point3f> &pts2,
    Mat &R, Mat &t
);

// BA by g2o
void BAPoseLandmarkG2O (
    const VecVector3d &points1_3d,
    const VecVector3d &points2_3d,
    Sophus::SE3d &pose,
    VecVector3d &points1_3d_ba // the set of landmarks will be updated if the OPTIMIZATION_MODE flag is 1 or 2
);

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }

    // -- Loading images, one can also use the flag 1 to load color images
#if CV_MAJOR_VERSION < 4
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); // OpenCV 3
#else
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);    // OpenCV 4
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);    // OpenCV 4
#endif
    assert(img_1.data && img_2.data && "Cannot load the image!");

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "Number of feature correspondences found in total: " << matches.size() << std::endl;

    // Construct 3D points
    // when loading the depth image, one can also use the flag -1 to specify the "UNCHANGED" loading mode
#if CV_MAJOR_VERSION < 4
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 3
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 3
#else
    Mat depth1 = imread(argv[3], cv::IMREAD_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 4
    Mat depth2 = imread(argv[4], cv::IMREAD_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 4
#endif

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // the camera intrinsics for the camera
    std::vector<Point3f> pts1, pts2;
    
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

    //////////////// G2O BA ////////////////
    cout << "Calling bundle adjustment by G2O" << endl;
    
    // Store the feature keypoints into Eigen vector containers
    VecVector3d pts1_3d_eigen, pts1_3d_ba;
    VecVector3d pts2_3d_eigen;
    
    for (size_t i = 0; i < pts1.size(); i++) {
        pts1_3d_eigen.push_back(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        pts2_3d_eigen.push_back(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    }
    // a deep copy of pts1_3d_eigen for bundle adjustment that also updates landmarks
    pts1_3d_ba = pts1_3d_eigen;

    // Sophus::SE3d pose_g2o; // using an identity matrix as initial guess, or comment this line to following block of code
    Eigen::Matrix3d R_eigen; ///////// uncomment the followng block of code, or use the line above of this block of code
    R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
               R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
               R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    Sophus::SE3d pose_g2o(R_eigen, t_eigen); /////////
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    BAPoseLandmarkG2O(pts1_3d_eigen, pts2_3d_eigen, pose_g2o, pts1_3d_ba);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by solving ICP in G2O: " << time_used.count() << " seconds." << endl;

    //verify p1 = R * p2 + t, here p1 and p2 are individual feature points
    for (int i = 0; i < 5; i++) { // show the first three points
        Eigen::Matrix<double,4,1> pt_tmp;
        pt_tmp << pts2[i].x, pts2[i].y, pts2[i].z, 1;
        cout << "p1 =  " << pts1[i] << endl;
        cout << "p2 =  " << pts2[i] << endl;
        cout << "p1_ba=" << pts1_3d_ba[i].transpose() << endl;
        cout << "(R * p2 + t) =\n" << 
            R * (Mat_<double>(3,1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << "\n\n" << // using this or below line
            (pose_g2o.matrix() * pt_tmp).head<3>()  // using this or above line
            << endl;
        cout << endl;
    }

    // compare the total cost between the transformation
    double err_cv = 0.0, err_g2o = 0.0;
    Eigen::Matrix3d R_SVD;
    R_SVD << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
             R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
             R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    Eigen::Matrix<double, 3, 1> t_SVD;
    t_SVD << t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);
    for (int i = 0; i < pts1.size(); i++) {
        Eigen::Matrix<double,4,1> pt_tmp;
        pt_tmp << pts2[i].x, pts2[i].y, pts2[i].z, 1;
        pt_tmp = pose_g2o.matrix() * pt_tmp;
        err_g2o   += ( (pt_tmp[0] - pts1_3d_ba[i][0])*(pt_tmp[0] - pts1_3d_ba[i][0]) +
                       (pt_tmp[1] - pts1_3d_ba[i][1])*(pt_tmp[1] - pts1_3d_ba[i][1]) +
                       (pt_tmp[2] - pts1_3d_ba[i][2])*(pt_tmp[2] - pts1_3d_ba[i][2]));

        Eigen::Vector3d cv_tmp;
        cv_tmp << pts2[i].x, pts2[i].y, pts2[i].z;
        cv_tmp = R_SVD * cv_tmp + t_SVD;
        err_cv += ( (cv_tmp[0] - pts1[i].x)*(cv_tmp[0] - pts1[i].x) +
                    (cv_tmp[1] - pts1[i].y)*(cv_tmp[1] - pts1[i].y) +
                    (cv_tmp[2] - pts1[i].z)*(cv_tmp[2] - pts1[i].z));
    }

    cout << "SVD point error: " << err_cv << "\t" << "G2O BA point error: " << err_g2o << endl;


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

/** The G2O workflow
 * 1. Define vertex class(es)/structure(s)
 * 2. Define edge class(es)/structure(s)
 * 3. Setup and allocate a solver object (with a specified optimization algorithm)
 * 4. Construct the optimization graph by adding vertices and edges to the optimizer
 * 5. Pefrom optimization and return results
 */

// 1. Define the vertex class to be used
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> { // g2o::BaseVertex<D,T> where D: minimal dimension of the vertex
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // override the resetn function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
        virtual void setToOriginImpl() override {
            _estimate = Sophus::SE3d();
        }

        /// left multiplication on SE3
        virtual void oplusImpl(const double *update) override {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate; // left perturbation
        }

        virtual bool read(istream &in) override {return false;}
        virtual bool write(ostream &out) const override {return false;}
}; // VertexPose


class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> { // g2o::BaseVertex<D,T> where D: minimal dimension of the vertex
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexPoint() {}

        // override the reset function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
        virtual void setToOriginImpl() override {
            _estimate = Eigen::Vector3d(0, 0, 0);
        }

        // override the Oplus operator to be addition to the old values
        virtual void oplusImpl(const double *update) override {
            _estimate += Eigen::Vector3d( update[0], update[1], update[2] ); // coor. update of 3D landmarks is simple addition
        }

        virtual bool read(istream &in) override {return false;}
        virtual bool write(ostream &out) const override {return false;}
}; // VertexPoint


// 2. Define edge class(es)/structure(s)
// g2o::BaseBinaryEdge<D,E, VertexXi, VertexXj> where D: error dimension; E: error data type
class EdgeProjection : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexPose, VertexPoint> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeProjection() {}

        // override the error computation function - Notice: this method should be changed accordingly to fit the problem setup
        virtual void computeError() override {
            const VertexPose  *v0 = static_cast<VertexPose *>(_vertices[0]); // the _vertices attribute is inherited from g2o::HyperGraph::Edge defined in "g2o/core/hyper_graph.h"
            const VertexPoint *v1 = static_cast<VertexPoint*>(_vertices[1]); 
            Sophus::SE3d T = v0->estimate();
            Eigen::Vector3d pos_pt_ref = T * _measurement; // point coordinates under the world reference frame [3 x 1]
            // Note: The problem setup here transforms the measurement to find the difference (follows the VSLAM book), which may NOT be a common practice
            // Here, measurements refer to the points in the second moving camera frame
            _error = v1->estimate() - pos_pt_ref; // the _error protected attribute is defined under g2o::BaseEdge in "g2o/core/base_edge.h"
        }

        // override the linearizaition funciton - the jacobian is defined in this method
        // override the linearizaition funciton
        virtual void linearizeOplus() override {
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
            const VertexPoint *v1 = static_cast<VertexPoint *>(_vertices[1]);
            Sophus::SE3d T = v0->estimate(); // in the problem setup, this is T_wc
            Eigen::Vector3d pos_ref = T * _measurement; // transform the point to the reference world frame
            double X = pos_ref[0];
            double Y = pos_ref[1];
            double Z = pos_ref[2];
            double Z2 = Z * Z;
            double inv_z = 1.0 / Z;
            double inv_z2 = inv_z * inv_z;
    
            _jacobianOplusXi // define the jacobian matrix for pose explicitly - [3 x 6]
                <<  -1,  0,  0,           0, -pos_ref[2],  pos_ref[1], // first row
                     0, -1,  0,  pos_ref[2],           0, -pos_ref[0], // second row
                     0,  0, -1, -pos_ref[1],  pos_ref[0],           0; // third row
                    
            _jacobianOplusXj // define the jacobian matrix for landmark explicitly - [3 x 3]
                << 1, 0, 0,
                   0, 1, 0,
                   0, 0, 1; // in this problem setup, the Jacobian of each err term w.r.t. the ref frame pt is the identity
        }

        virtual bool read(istream &in) override {return false;}

        virtual bool write(ostream &out) const override {return false;}
}; // EdgeProjection

// BA by g2o
void BAPoseLandmarkG2O (
    const VecVector3d &points1_3d,
    const VecVector3d &points2_3d,
    Sophus::SE3d &pose,
    VecVector3d &points1_3d_ba // the set of landmarks will be updated if the OPTIMIZATION_MODE flag is 1 or 2
) {
    const int iterations = 50; // maximum number of iterations
    int N = (int)points1_3d.size(); // number of landmarks involved in the optimization problem
    // omit updating the last several landmarks to avoid an ill-posed problem
    int omit_num = 3 + 1; // will ignore the last (omit_num-1) landmarks : DO NOT CHANGE THE +1 part
    
    cout << "pose inputted to g2o =\n" << pose.matrix() << endl;

    // 3. Allocate and setup a solver object (with a specified optimization algorithm)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // poses 6D, landmarks 3D
    //// using one of the below linear solvers: Dense (default), CSparse
    // typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type - Dense
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type - CSparse
    //// using one of the gradient methods
    // auto solver = new g2o::OptimizationAlgorithmGaussNewton( // Gauss-Newton
    auto solver = new g2o::OptimizationAlgorithmLevenberg( // Levenberg-Marquardt
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // Graph model
    optimizer.setAlgorithm(solver); // Setup the solver
    optimizer.setVerbose(true);     // Turn on verbose output for debugging

    // create vectors to store the created vertices
    std::vector<VertexPose *> vertex_poses;
    std::vector<VertexPoint *> vertex_points;

    // 4. Construct the optimization graph by adding vertices and edges to the optimizer
    // 4.1 adding vertiices into the graph
    // In this case, we have only one camera pose to be optimized
    VertexPose *v_cam = new VertexPose();
    v_cam->setEstimate(pose); // the setEstimate() method can be found at: "g2o/core/base_vertex.h"
    v_cam->setId(0); // the _id member is under "g2o/core/parameter.h", and the setId() method is defined at "g2o/core/optimizable_graph.h"

    if (OPTIMIZATION_MODE == 1) { // when mode is 1, we only update landmarks
        v_cam->setFixed(true); // the setFixed() method is defined in "g2o/core/optimizable_graph.h"
    }

    optimizer.addVertex(v_cam);
    vertex_poses.push_back(v_cam);

    // landmark vertices
    for (int i = 0; i < points1_3d.size(); i++) {
        VertexPoint *v_pt = new VertexPoint();
        Eigen::Vector3d point = points1_3d_ba[i];
        v_pt->setId(i+vertex_poses.size()); // the Id values follows the last one from camera pose
        v_pt->setEstimate(Eigen::Vector3d(point[0], point[1], point[2]));

        if (OPTIMIZATION_MODE == 0) { // when mode is 0, we only update poses
            v_pt->setFixed(true); 
        } else { // for OPTIMIZATION_MODE of 1 and 2
            if (OPTIMIZATION_MODE == 2) { // when mode is 2, we will marginalize the landmarks: the non-mariginalized vertices are processed, then the marginalized ones
                v_pt->setMarginalized(true); // BA in g2o needs to manually set vertices to be marginalized
            }
            // To avoid an ill-posed problem, will fix the last (omit_num-1) landmarks, i.e., not update those landmarks
            if (i > (int)points1_3d.size() - omit_num) {
                v_pt->setFixed(true); 
            }
        }

        optimizer.addVertex(v_pt);
        vertex_points.push_back(v_pt);
    }

    // 4.2 adding edges into the graph
    int index = 1; // an index for edge
    for (int i = 0; i < points1_3d.size(); i++) {
        auto pt_2 = points2_3d[i];
        // auto pt_1 = points1_3d_ba[i];
        EdgeProjection *edge = new EdgeProjection();
        edge->setId(index);
        edge->setVertex(0, vertex_poses[0]); // in this problem, we have only one camera pose to consider
        edge->setVertex(1, vertex_points[i]);
        edge->setMeasurement(pt_2);
        edge->setInformation(Eigen::Matrix3d::Identity()); // assuming the same identity uncertainty for all points
        // edge->setRobustKernel(new g2o::RobustKernelHuber()); // using a robust kernel, can comment this line to not use
        optimizer.addEdge(edge);
        index++;
    }

    // 5. Pefrom optimization and return results
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization time used: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex_poses[0]->estimate().matrix() << endl;
    pose = vertex_poses[0]->estimate(); // update the pose estimate

    // update the landmark estimates if the landmarks were optimized
    if (OPTIMIZATION_MODE != 0) {
        for (int i = 0; i <= (int)points1_3d.size() - omit_num; i++) {
            auto pt_ver = vertex_points[i];
            for (int k = 0; k < 3; k++) {points1_3d_ba[i][k] = pt_ver->estimate()[k];}
        }
    }

}


