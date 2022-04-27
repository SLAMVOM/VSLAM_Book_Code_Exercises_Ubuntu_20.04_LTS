/*************************************************************************
* This script includes solving the PnP problem using three different ways:
* 1. Using OpenCV's functions
* 2. Explictly implementing the BA using Gauss Newton
* 3. Using g2o package to formula the problem as a graph optimization
*************************************************************************/

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
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>


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

// BA by g2o
void bundleAdjustmentG2O (
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);


// BA by gauss-newton
void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);

int main(int argc, char **argv) {
    if (argc!= 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    // -- Loading images
    // Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    // Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR); // OpenCV 4
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR); // OpenCV 4
    assert(img_1.data && img_2.data && "Cannot load the images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Number of feature correspondences found in total: " << matches.size() << endl;

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
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // using OpenCV's PnP solver, option for EPNP, DLS or other methods
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

    // Gauss-Newton
    cout << "Calling bundle adjustment by Gauss Newton" << endl;
    Sophus::SE3d pose_gn;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 -t1);
    cout << "Time spent by solving PnP in Gauss Newton: "<< time_used.count() << " seconds." << endl;

    // Graph optimization
    cout << "Calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by solving PnP in g2o: " << time_used.count() << " seconds." << endl;

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

void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute the cost
        for (int i = 0; i < points_3d.size(); i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J <<-fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b); // using Eigen's Robust Cholesky decomposition solver to solve for optimal update

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // once the cost starts increasing, stop incrementing the update
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update the estimation
        pose = Sophus::SE3d::exp(dx) * pose; // here is a matrix exponential
        lastCost = cost;

        cout << "Iteration " << iter << " cost = " << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) {
            // if the norm of update for variables is smaller than 1e-6 -> convergence
            break;
        }
    }
    cout << "pose by Gauss-Newton: \n" << pose.matrix() << endl;
}

// define the vertex and edges used in the g2o BA
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override { // set the initial guess of the pose
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
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d); // pixel coordinates [3 x 1]
        pos_pixel /= pos_pixel[2]; // normalize the coordinates by dividing the third dimension
        // Note: the reprojection error is defined as measured - estimated
        _error = _measurement - pos_pixel.head<2>(); // the _error protected attribute is defined under g2o::BaseEdge in "g2o/core/base_edge.h"
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi // define the jacobian matrix explicitly
            << -fx/Z,     0, fx*X/Z2,    fx*X*Y/Z2, -fx-fx*X*X/Z2,    fx*Y/Z,
                   0, -fy/Z, fy*Y/Z2, fy+fy*Y*Y/Z2,    -fy*X*Y/Z2,   -fy*X/Z;
    }

    virtual bool read(istream &in) override {return false;}

    virtual bool write(ostream &out) const override {return false;}

    private:
        Eigen::Vector3d _pos3d;
        Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {

    // construct graph optimization, first setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // pose is 6d, landmarks are 3d
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type
    // choose one of the gradient methods: Gauss-Newton, Levenberg-Marquardt, DogLet
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // Graph model
    optimizer.setAlgorithm(solver); // Set up the solver
    optimizer.setVerbose(true); // Turn on verbose output for debugging

    // vertex
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); i++) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization time used: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}
