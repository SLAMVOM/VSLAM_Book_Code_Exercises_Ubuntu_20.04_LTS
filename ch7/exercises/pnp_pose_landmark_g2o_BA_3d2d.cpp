/******************************************************************************
* This script includes solving the PnP BA problem with options to optimize
* pose only, landmark only, or both by using the g2o package that
* formulates the problem as a graph optimization.
******************************************************************************/

/********************************************************************
 * Note that in a two-view PnP setting, if we set one of the frames
 * as an inertia (or world) reference frame, while (assuming) all the
 * landmarks are observed by the two cameras. In such case, if we
 * try to optimize the 6D pose and all the landmarks coordinates in
 * the reference frame together. The problem will be ill-posed,
 * because each observation can only give two constraints/conditions.
 * At the same time we have 3N variables for all the landmarks plus
 * 6 variables for the camera pose of the moving frame.
 * Therefore, we have (3N + 6) variables but have only 2N conditions.
 * There are less conditions than unknowns, thus the problem is
 * ill-posed. We can fix (more than) N/3+2 landmarks and only update
 * the pose and the other landmarks to make the system to give an
 * either unique or overdetermined solution.
********************************************************************/

// Author: MT
// Creation Date: 2022-April-26
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
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;
using namespace cv;

//////////////// Set the following flag to a different value to perform a different optimization task
#define OPTIMIZATION_MODE   2   // 0 - pose only; 1 - landmarks only; 2 - both pose and landmarks

// Searching for feature matches between image pair
void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// Converting pixel coordinates to normalized image plane coordinates
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat&K); // return the first two dimensions since the last is always 1

// Define two types to store data points
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// BA by g2o
void BAPoseLandmarkG2O (
    const VecVector3d &points1_3d,
    const VecVector2d &points1_2d,
    const VecVector2d &points2_2d,
    const Mat &K,
    Sophus::SE3d &pose,
    VecVector3d &points1_3d_ba // the set of landmarks will be updated if the OPTIMIZATION_MODE flag is 1 or 2
);


int main(int argc, char **argv) {
    if (argc!= 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
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
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 3
#else
    Mat d1 = imread(argv[3], cv::IMREAD_UNCHANGED);  // depth image is single-channel 16-bits unsigned values - OpenCV 4
#endif

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // the camera intrinsics for the camera
    std::vector<Point3f> pts1_3d;
    std::vector<Point2f> pts1_2d, pts2_2d;
    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) // 0 indicates no depth information available
            continue;
        float dd = d / 5000.0; // dividing the depth value by a scalar value of 5000, specific to the camera
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts1_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts1_2d.push_back(keypoints_1[m.queryIdx].pt);
        pts2_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    std::cout << "3d-2d pairs: " << pts1_3d.size() << std::endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    cv::solvePnP(pts1_3d, pts2_2d, K, Mat(), r, t, false); // using OpevCV's PnP solver, options for EPNP, DLS or other methods
    Mat R;
    cv::Rodrigues(r, R); // r is a rotation vector, so using the Rodrigues formula to convert it into a matrix form
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cout << "Time spent by solving PnP in OpenCV: " << time_used.count() << " seconds." << std::endl;

    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << "t = " << std::endl << t << std::endl;

    // Store the feature keypoints into Eigen vector containers
    // Note that in the PnP problem, one set of points are with known depths,
    // while the other set only knows the pixels coordinates
    VecVector3d pts1_3d_eigen;
    VecVector2d pts1_2d_eigen, pts2_2d_eigen;
    for (size_t i = 0; i < pts1_3d.size(); i++) {
        pts1_3d_eigen.push_back(Eigen::Vector3d(pts1_3d[i].x, pts1_3d[i].y, pts1_3d[i].z));
        pts1_2d_eigen.push_back(Eigen::Vector2d(pts1_2d[i].x, pts1_2d[i].y));
        pts2_2d_eigen.push_back(Eigen::Vector2d(pts2_2d[i].x, pts2_2d[i].y));
    }
    
    // a deep copy of pts_3d_eigen for bundle adjustment that also updates landmarks
    VecVector3d pts1_3d_ba = pts1_3d_eigen;

    // G2O Graph optimization
    std::cout << "Calling bundle adjustment by g2o" << std::endl;
    Sophus::SE3d pose_g2o; // using an identity matrix as initial guess, or comment this line to following block of code
    // Eigen::Matrix3d R_eigen; ///////// uncomment the followng block of code, or use the line above of this block of code
    // R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
    //            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
    //            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    // Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    // Sophus::SE3d pose_g2o(R_eigen, t_eigen); /////////
    t1 = chrono::steady_clock::now();
    BAPoseLandmarkG2O(pts1_3d_eigen, pts1_2d_eigen, pts2_2d_eigen, K, pose_g2o, pts1_3d_ba);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cout << "Time used by solving PnP in g2o: " << time_used.count() << " seconds." << std::endl;

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

/** The G2O workflow
 * 1. Define vertex class(es)/structure(s)
 * 2. Define edge class(es)/structure(s)
 * 3. Setup and allocate a solver object (with a specified optimization algorithm)
 * 4. Construct the optimization graph by adding vertices and edges to the optimizer
 * 5. Pefrom optimization and return results
 */


// 1. Define the vertex class to be used in the g2o BA
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> { // g2o::BaseVertex<D,T> where D: minimal dimension of the vertex
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // override the reset function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
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
class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexPoint> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeProjection(const Eigen::Matrix3d &K) : _K(K) {}

        // override the error computation function
        virtual void computeError() override {
            const VertexPose  *v0 = static_cast<VertexPose *>(_vertices[0]); // the _vertices attribute is inherited from g2o::HyperGraph::Edge defined in "g2o/core/hyper_graph.h"
            const VertexPoint *v1 = static_cast<VertexPoint*>(_vertices[1]); 
            Sophus::SE3d T = v0->estimate();
            Eigen::Vector3d pos_pixel = _K * (T * (v1->estimate())); // pixel coordinates [3 x 1]
            pos_pixel /= pos_pixel[2]; // converted the 3D landmark from camera frame coordinates to pixel coordinates
            // Note: the reprojection error is defined as measured - estimated
            _error = _measurement - pos_pixel.head<2>(); // the _error protected attribute is defined under g2o::BaseEdge in "g2o/core/base_edge.h"
        }

        // override the linearizaition funciton
        virtual void linearizeOplus() override {
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
            const VertexPoint *v1 = static_cast<VertexPoint *>(_vertices[1]);
            Sophus::SE3d T = v0->estimate();
            Eigen::Vector3d pos_cam = T * (v1->estimate()); // transform the point to current frame
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double cx = _K(0, 2);
            double cy = _K(1, 2);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Z2 = Z * Z;
            double inv_z = 1.0 / Z;
            double inv_z2 = inv_z * inv_z;
    
            _jacobianOplusXi // define the jacobian matrix for pose explicitly - [2 x 6]
                <<  -fx/Z,     0, fx*X/Z2,    fx*X*Y/Z2, -fx-fx*X*X/Z2,    fx*Y/Z,
                        0, -fy/Z, fy*Y/Z2, fy+fy*Y*Y/Z2,    -fy*X*Y/Z2,   -fy*X/Z;
            _jacobianOplusXj // define the jacobian matrix for landmark explicitly
                << -fx * inv_z,           0, fx * X * inv_z2,
                             0, -fy * inv_z, fy * Y * inv_z2;
            _jacobianOplusXj = _jacobianOplusXj * T.rotationMatrix(); // [2 x 3]

        }

        virtual bool read(istream &in) override {return false;}

        virtual bool write(ostream &out) const override {return false;}

        private:
            Eigen::Matrix3d _K;
}; // EdgeProjection


void BAPoseLandmarkG2O (
    const VecVector3d &points1_3d,
    const VecVector2d &points1_2d,
    const VecVector2d &points2_2d,
    const Mat &K,
    Sophus::SE3d &pose,
    VecVector3d &points1_3d_ba // the set of landmarks will be updated if the OPTIMIZATION_MODE flag is 1 or 2
) {
    // 3. Allocate and setup a solver object (with a specified optimization algorithm)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // poses 6D, landmarks 3D
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type
    // using one of the gradient methods
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
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
            // To avoid an ill-posed problem, set half of the landmarks to be fixed, i.e., only optimize half of the landmarks
            if (i % 2 == 1) {
                v_pt->setFixed(true); 
            }
        }

        optimizer.addVertex(v_pt);
        vertex_points.push_back(v_pt); 
    }

    // define the camera intrinsics matrix, K 
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // 4.2 adding edges into the graph
    int index = 1; // an index for edge
    for (int i = 0; i < points1_3d.size(); i++) {
        auto p2d = points2_2d[i];
        auto p3d = points1_3d_ba[i];
        EdgeProjection *edge = new EdgeProjection(K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_poses[0]); // in this problem, we have only one camera pose to consider
        edge->setVertex(1, vertex_points[i]);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        // edge->setRobustKernel(new g2o::RobustKernelHuber()); // using a robust kernel, can comment this line to not use
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(50);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization time used: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex_poses[0]->estimate().matrix() << endl;
    pose = vertex_poses[0]->estimate(); // update the pose estimate
    
    // update the landmark estimates if the landmarks were optimized
    if (OPTIMIZATION_MODE != 0) {
        for (int i = 0; i < points1_3d.size(); i++) {
            if (i % 2 == 0) { // only half the landmarks are optimized to avoid an ill-posed problem setup
                auto pt_ver = vertex_points[i];
                for (int k = 0; k < 3; k++) {points1_3d_ba[i][k] = pt_ver->estimate()[k];}
            }
        }
    }

    // compare some landmarks before and after BA
    cout << "point coor before (front) and after (latter) BA: \n" << endl;
    double err = 0.0;
    for (int j = 0; j < points1_3d.size(); j++) {
        if (j < 6){
            cout << "[" << points1_3d[j].transpose() << "]\t[" << points1_3d_ba[j].transpose() << "]\n";
        }
        err += (points1_3d[j][0]-points1_3d_ba[j][0]) * (points1_3d[j][0]-points1_3d_ba[j][0]) +
               (points1_3d[j][1]-points1_3d_ba[j][1]) * (points1_3d[j][1]-points1_3d_ba[j][1]) + 
               (points1_3d[j][2]-points1_3d_ba[j][2]) * (points1_3d[j][2]-points1_3d_ba[j][2]);
    }
    cout << "\nSquared difference between all the initial and optimized landmark coordinates: " << err << '\n' << endl;
}




