/******************************************************************************
* This script includes solving the PnP BA problem, 
* only landmark optimization by using using the g2o package
* to formula the problem as a graph optimization
******************************************************************************/

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

#define G2OBA 1 // 1 for BA including landmark optimizations
                // 0 for camera pose optimization only

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


#if G2OBA
void G2OBAFunc(
    const vector<cv::Point3f> &points1_3d, // 3d correspondences in the first frame (with depth info)
    const vector<cv::Point2f> &points1_2d, // image pixel coordinates of correspondences in first frame
    const vector<cv::Point2f> &points2_2d, // image pixel coordinates of correspondences in second frame
    const Mat &K,
    const Mat &R,
    const Mat &t
);
#else
// BA by g2o for only the poses
void bundleAdjustmentG2O (
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);
#endif


int main(int argc, char **argv) {
    if (argc!= 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    // -- Loading images
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Cannot load the images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Number of feature correspondences found in total: " << matches.size() << endl;

    // Construct 3D points
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);  // depth image is single-channel 16-bits unsigned values
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    #if G2OBA
        vector<Point2f> pts1_2d; // pixel coordinates in the first frame, only needed when pts coordinates to be optimized
    #endif

    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) // 0 indicates no depth information available
            continue;
        float dd = d / 5000.0; // dividing the depth value by a scalar value of 5000
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd)); // 3D coor in first cam frame, triangulation to get 3D coor
        pts_2d.push_back(keypoints_2[m.trainIdx].pt); // pixel coor of paired correspondences in the second frame

        #if G2OBA
            pts1_2d.push_back( keypoints_1[m.queryIdx].pt ); // pixel coor of correspondences in the first frame
        #endif
    }

    // using the first frame as the world frame, R = I, t = 0 in the first frame
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    // // Using OpenCV's implemented method to solve the PnP problem
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    // arguments: 3D coor in first cam, pixel coor in second cam, intrinsics, distortion, rot vec, trans vec, false - not using r and t as init val
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // using OpenCV's PnP solver, option for EPNP, DLS or other methods
    Mat R;
    cv::Rodrigues(r, R); // r is a rotation vector, so using the Rodrigues formula to convert it into a matrix form
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by solving PnP in OpenCV: " << time_used.count() << " seconds." << endl;

    cout << "R = " << endl << R << endl;
    cout << "t = " << endl << t << endl;

    #if G2OBA
        cout << "Optimizing landmark coordinates by g2o" << endl;
    #else
        // Store the feature keypoints into Eigen vector containers
        // Note that in the PnP problem, one set of points are with known depth,
        // while the other set only knows the image plane coordinates
        VecVector3d pts_3d_eigen;
        VecVector2d pts_2d_eigen;
        for (size_t i = 0; i < pts_3d.size(); i++) {
            pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
            pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
        }
    #endif

    // Graph optimization
    cout << "Calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();

    #if G2OBA
        G2OBAFunc( pts_3d, pts1_2d, pts_2d, K, R, t );
    #else
        bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    #endif

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


// define the vertex and edges used in the g2o BA
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        virtual void setToOriginImpl() override {
            _estimate = Sophus::SE3d();
        }

        /// left multiplication on SE3
        virtual void oplusImpl(const double *update) override {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate; // left perturbation
        }

        virtual bool read(istream &in) override {}
        virtual bool write(ostream &out) const override {}
};


class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // VertexPoint() {}

        virtual void setToOriginImpl() override {
            _estimate = Eigen::Vector3d(0, 0, 0);
        }

        virtual void oplusImpl(const double *update) override {
            _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
        }

        virtual bool read(istream &in) override {}

        virtual bool write(ostream &out) const override {}
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
        _error = _measurement - pos_pixel.head<2>();
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
            << -fx/Z, 0, fx*X/Z2, fx*X*Y/Z2, -fx-fx*X*X/Z2, fx*Y/Z,
            0, -fy/Z, fy*Y/Z2, fy+fy*Y*Y/Z2, -fy*X*Y/Z2, -fy*X/Z;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

    private:
        Eigen::Vector3d _pos3d;
        Eigen::Matrix3d _K;
};


#if G2OBA
void G2OBAFunc( // function reference: cnblogs.com/newneul/p/8545450.html
    const vector<cv::Point3f> &points1_3d, // 3d correspondences in the first frame (with depth info)
    const vector<cv::Point2f> &points1_2d, // image pixel coordinates of correspondences in first frame
    const vector<cv::Point2f> &points2_2d, // image pixel coordinates of correspondences in second frame
    const Mat &K,                          // the intrinsic values are not updated, so set as const
    const Mat &R,                          // initial guess of the rotation matrix, R21
    const Mat &t                           // initial guess of the translation vec
){
    // try changing to 1
    #define PoseVertexNum 2 // define the number of pose nodes, there are two camera frames here

    // setup the optimizer
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block; // pose is 6d, landmarks are 3d
    // std::unique_ptr<Block::LinearSolverType> linearSolver = g2o::make_unique<g2o::LinearSolverCSparse<Block::PoseMatrixType> >(); // using the CSparse linear solver
    // std::unique_ptr<Block::LinearSolverType> linearSolver ( new g2o::LinearSolverCSparse<Block::PoseMatrixType>() ); // using the CSparse linear solver
    // std::unique_ptr<Block> solver_ptr ( new Block(std::move(linearSolver)) );
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );
    
    
    typedef g2o::LinearSolverDense<Block::PoseMatrixType> linearSolver; // a linear solver type
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<Block>(g2o::make_unique<linearSolver>()));

    g2o::SparseOptimizer optimizer; // set sparse optimizer
    optimizer.setAlgorithm(solver); // set the optimization algorithm


    // Adding vertices and edges into the optimizer
    // Ading vertex
    // (1) adding pose vertex, where the first frame is treated as the world frame (also a cam frame), thus not optimized
    // int poseVertexIndex = 0;
    // Eigen:: Matrix3d R2Init; // the initial guess of R21
    // R2Init << 
    //         R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    //         R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    //         R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    // for( ; poseVertexIndex < PoseVertexNum; poseVertexIndex++) {
    //     auto pose = new g2o::VertexSE3Expmap(); // camera pose
    //     pose->setId( poseVertexIndex ); // set the vertex label
    //     pose->setFixed( poseVertexIndex == 0 ); // if it is the first frame, then keep the value fixed
    //     if( poseVertexIndex == 1 )    // in the second frame, let the initialization of R,t be the PnP given by OpenCV
    //         pose->setEstimate(
    //                 g2o::SE3Quat ( R2Init,
    //                                Eigen::Vector3d( t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0) )
    //                              )
    //                          );
    //     else
    //         pose->setEstimate( g2o::SE3Quat() );
    //         optimizer.addVertex( pose ); // adding the pose vertex into the optimizer
    // }
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);


    // (2) Adding landmark vertices into the optimizer    
    // for(int i = 0; i < points1_3d.size(); i++) {
    //     auto point = new g2o::VertexPointXYZ(); // 3d landmark point class by g2o
    //     point->setId( landmarkVertexIndex + i ); // index the landmark point
    //     point->setMarginalized( true ); // setting the marginalization
    //     point->setEstimate( Eigen::Vector3d(points1_3d[i].x, points1_3d[i].y, points1_3d[i].z) ); // set the initial values, which are the 3d coordinates in the first frame
    //     optimizer.addVertex( point ); // adding vertex into the optimizer
    // }
    int landmarkVertexIndex = PoseVertexNum;
    vector<VertexPoint *> vertex_points;
    for(int i = 0; i < points1_3d.size(); i++) {
        VertexPoint *vertex_point = new VertexPoint(); // point vertex_point
        vertex_point->setId(landmarkVertexIndex + i); // index the landmark point
        vertex_point->setEstimate( Eigen::Vector3d(points1_3d[i].x, points1_3d[i].y, points1_3d[i].z) ); // set the initial values, which are the 3d coordinates in the first frame
        vertex_point->setMarginalized( true ); // setting the marginalization
        optimizer.addVertex( vertex_point ); // adding vertex into the optimizer
        vertex_points.push_back(vertex_point);
    }

    // Adding camera parameters, K, to be optimized (the last element is 0, assume fx=fy, then optimize pose)
    // g2o::CameraParameters *camera = new g2o::CameraParameters(
    //     K.at<double>(0,0), Eigen::Vector2d( K.at<double>(0,2), K.at<double>(1,2)), 0
    // );
    // camera->setId(0);
    // optimizer.addParameter( camera );
    Eigen::Matrix3d K_eigen;
    K_eigen << 
                K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
                K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
                K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);


    // Adding edges
    // connecting landmark vertices to first-frame pose vertex (we don't optimize the R and t for the first frame, only optimize the R21 and t_2^{21})
    // for(int i = 0; i < points1_2d.size(); i++) {
    //     auto edge = new g2o::EdgeProjectXYZ2UV; // set the edge that connects to the first frame
    //     // binary edge
    //     edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(landmarkVertexIndex + i)) ) ; // the coordinates under the world coor system
    //     edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap *> (optimizer.vertex(0)) );
    //     edge->setMeasurement( Eigen::Vector2d(points1_2d[i].x, points1_2d[i].y) ); // set the measurement as the values in the normalized plane coordinates in the first frame
    //     edge->setParameterId(0, 0); // set the camera parameter(since we only input a cam param with id=0, corresponds to camera->setId(0) above, note the first parameter should be 0)
    //     edge->setInformation( Eigen::Matrix2d::Identity() ); // 2 x 2 identity information matrix
    //     optimizer.addEdge(edge);
    // }
    // // connecting the landmark vertices to the second-frame poes vertex
    // for(int i = 0; i < points1_2d.size(); i++) {
    //     auto edge = new g2o::EdgeProjectXYZ2UV(); // set the edge that connects to the second frame
    //     edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex( landmarkVertexIndex + i)) ); // the coordinates under the first cam frame (world frame)
    //     edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1)) ); // connects to the second cam vertex
    //     edge->setMeasurement( Eigen::Vector2d(points2_2d[i].x, points2_2d[i].y) ); // set the measurement as the values in the normalized plane coordinates in the second frame
    //     edge->setInformation( Eigen::Matrix2d::Identity() ); // 2 x 2 identity information matrix, that the the error weighting is 1:1
    //     edge->setParameterId(0,0);
    //     optimizer.addEdge(edge);
    // }
    for(int i = 0; i < points1_2d.size(); i++) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose);
        edge->setVertex(1, vertex_points[i]);
        edge->setMeasurement(Eigen::Vector2d(points1_2d[i].x, points1_2d[i].y));// set the measurement as the values in the normalized plane coordinates in the first frame
        edge->setInformation( Eigen::Matrix2d::Identity() ); // 2 x 2 identity information matrix, that the the error weighting is 1:1
        optimizer.addEdge(edge);
    }
        for(int i = 0; i < points1_2d.size(); i++) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose);
        edge->setVertex(1, vertex_points[i]);
        edge->setMeasurement(Eigen::Vector2d(points2_2d[i].x, points2_2d[i].y));// set the measurement as the values in the normalized plane coordinates in the first frame
        edge->setInformation( Eigen::Matrix2d::Identity() ); // 2 x 2 identity information matrix, that the the error weighting is 1:1
        optimizer.addEdge(edge);
    }


    // Running the optimization
    cout << "--------- Starting optimization ---------" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true); // set the verbose setting to true for output
    optimizer.initializeOptimization(); // initialize the optimizer
    optimizer.optimize(100); // maximum 100 iterations for the optimization
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    cout << "--------- Finishing optimization ---------" << endl;
    chrono::duration<double>time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization spent time: " << time_used.count() << "sec." << endl;
    cout << endl << "Outputing optimization results:" << endl;
    // Output the pose vertex by calliing estimate(), the output is SE3(). Note: Eigen::Isometry3d is a 4x4 transformation matrix
    // cout << "T = " << endl << Eigen::Isometry3d(dynamic_cast< g2o::VertexSE3Expmap * >(optimizer.vertex(1))->estimate()).matrix() << endl;
    // cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
}
#else
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
#endif