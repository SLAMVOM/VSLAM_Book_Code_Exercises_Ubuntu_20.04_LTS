/********************************************************************
 * Note that in a two-view BA setting, if we set one of the frames
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
 * unique or overdetermined solution.
********************************************************************/

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

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// Converting pixel coordinates to normalized camera frame coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

void pose_estimation_3d3d_SVD(
    const vector<Point3f> &pts1,
    const vector<Point3f> &pts2,
    Mat &R, Mat &t
);

typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustment_GN(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    // Mat &R, Mat &t // either using rotation matrix and translation vector for transformation
    Sophus::SE3d &pose, // or using Sophus for transformation
    VecVector3d &points_3d_ba
);

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    // -- Loading images
    // Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    // Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR); // OpenCV 4
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR); // OpenCV 4


    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Number of feature correspondences found in total " << matches.size() << endl;

    // Creating 3D points
    // Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // depth image values are single-channel 16-bits unsigned int - OpenCV 3
    // Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED); // depth image values are single-channel 16-bits unsigned int - OpenCV 3
    Mat depth1 = imread(argv[3], cv::IMREAD_UNCHANGED); // depth image values are single-channel 16-bits unsigned int - OpenCV 4
    Mat depth2 = imread(argv[4], cv::IMREAD_UNCHANGED); // depth image values are single-channel 16-bits unsigned int - OpenCV 4

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
    pose_estimation_3d3d_SVD(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = \n" << R << endl;
    cout << "t = \n" << t << endl;
    cout << "R_inv = \n" << R.t() << endl;
    cout << "t_inv = \n" << -R.t() * t << endl;

    cout << "Calling bundle adjustment using Gauss-Newton" << endl;

    // Mat R_GN, t_GN; // if using rotation matrix and translation vector for transformation
    // bundleAdjustment_GN(pts1, pts2, R_GN, t_GN); // if using rotation matrix and translation vector for transformation


    // save the landmarks into VecVector3d type
    VecVector3d pts1_eigen, pts2_eigen;
    for (size_t i = 0; i < pts1.size(); i++){
        pts1_eigen.push_back(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        pts2_eigen.push_back(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    }

    // a deep copy of the pts1_eigen
    VecVector3d pts_3d_ba = pts1_eigen;

    Sophus::SE3d pose_gn; // if using Sophus SE3 for poses
    // Eigen::Matrix3d R_eigen; // given the pose estimate from the OpenCV as initial guess
    // R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
    //            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
    //            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    // Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    // Sophus::SE3d pose_gn(R_eigen, t_eigen); /////////
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    bundleAdjustment_GN(pts1_eigen, pts2_eigen, pose_gn, pts_3d_ba); // if using Sophus SE3 for poses
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by looping method for iters: " << time_used.count() << " seconds." << endl;


    //verify p1 = R * p2 + t, here p1 and p2 are individual feature points
    for (int i = 0; i < 5; i++) { // show the first three points
        Eigen::Matrix<double,4,1> pt_tmp;
        pt_tmp << pts2[i].x, pts2[i].y, pts2[i].z, 1;
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "p1_ba[" << pts_3d_ba[i][0] << ", " << pts_3d_ba[i][1] << ", " << pts_3d_ba[i][2] << "]" << endl;
        cout << "(R * p2 + t) =\n" <<
            R * (Mat_<double>(3,1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << "\n\n" << // using this or below line
            (pose_gn.matrix() * pt_tmp).head(3)  // using this or above line
            << endl;
        cout << endl;
    }

    // compare the total cost between the transformation
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
        pt_tmp = pose_gn.matrix() * pt_tmp;
        err_ceres += ( (pt_tmp[0] - pts_3d_ba[i][0])*(pt_tmp[0] - pts_3d_ba[i][0]) +
                       (pt_tmp[1] - pts_3d_ba[i][1])*(pt_tmp[1] - pts_3d_ba[i][1]) +
                       (pt_tmp[2] - pts_3d_ba[i][2])*(pt_tmp[2] - pts_3d_ba[i][2]));

        Eigen::Vector3d cv_tmp;
        cv_tmp << pts2[i].x, pts2[i].y, pts2[i].z;
        cv_tmp = R_SVD * cv_tmp + t_SVD;
        err_cv += ( (cv_tmp[0] - pts1[i].x)*(cv_tmp[0] - pts1[i].x) +
                    (cv_tmp[1] - pts1[i].y)*(cv_tmp[1] - pts1[i].y) +
                    (cv_tmp[2] - pts1[i].z)*(cv_tmp[2] - pts1[i].z));
    }

    cout << "SVD point error: " << err_cv << "\t" << "Gauss-Newton BA point error: " << err_ceres << endl;

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
void pose_estimation_3d3d_SVD(const vector<Point3f> &pts1,
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

void bundleAdjustment_GN(const VecVector3d &pts1,
                         const VecVector3d &pts2,
                        //  Mat &R, Mat &t // either using rotation matrix and translation vector for transformation
                         Sophus::SE3d &pose, // or using Sophus for transformation
                         VecVector3d &points_3d_ba)
                         {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 50; // maximum number of iterations
    double cost = 0, lastCost = 0;

    cout << "Initial pose: \n" << pose.matrix() << endl;

    int N = (int)pts1.size();

    // omit updating the last several landmarks
    int omit_num = N+1; // will ignore the last (omit_num-1) landmarks : DO NOT CHANGE THE +1 part

    // first layer for loop for iterations
    for (int iter = 0; iter < iterations; iter++) {

        // define some variables for the least square terms
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_all;
        Eigen::Matrix<double, Eigen::Dynamic, 1> b_all;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J_all;
        Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero(); // Jacobian matrix for the pose variables
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        J_all = J*1.0; // initialize J_all as J
        b_all = b*1.0; // initialize b_all as b

        // Extract the rotation matrix from the pose
        // Eigen::Matrix<double, 3, 3> R_tmp = pose.so3().matrix(); // wrong, becuase this is the Jac wrt the landmark in current frame
        Eigen::Matrix<double, 3, 3> R_tmp = Eigen::Matrix3d::Identity(); // the Jac wrt the world (reference) frame pt should be Identity

        // define several variables to store the intermediate Jacobians
        std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>> vec_jac_pts1; // Jacobian for landmarks
        std::vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> vec_jac_pose; // Jacobian for cam poses
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vec_b_pts1; // RHS for each pt

        cost = 0.0;
        // second layer for loop to compute cost accumulated through each individual landmark
        for (int i = 0; i < pts1.size(); i++) {

            Eigen::Vector3d p_trans = pose * pts2[i]; // compute the landmark coor under the second frame
            Eigen::Vector3d e = points_3d_ba[i] - p_trans; // compute the error for each landmark

            cost += e.squaredNorm(); // accumulate the error

            J << -1,  0,  0,           0, -p_trans[2],  p_trans[1], // first row
                  0, -1,  0,  p_trans[2],           0, -p_trans[0], // second row
                  0,  0, -1, -p_trans[1],  p_trans[0],           0; // third row

            if (i > (int)pts1.size() - omit_num) { // omit the last (omit_num-1) landmarks, Note: .size() returns an unsigned int
                J_all.conservativeResize(J_all.rows()+3, J_all.cols()); // [3n x (6+3(n-omit_num+1))]
            } else {
                if (i == 0) {
                    J_all.conservativeResize(J_all.rows(), J_all.cols()+3); // [3 x 9] to include jacobian for landmark

                    b_all = -J_all.transpose() * e; // [(6+3)x1] = [9 x 1]

                } else {
                    // extend the J_all matrix to include the current Jacobian, J
                    J_all.conservativeResize(J_all.rows()+3, J_all.cols()+3); // [3n x (6+3(n-omit_num+1))]
                    // J_all.block<3,3>(i*3, i*3+6) = -R_tmp;
                    // J_all.block<3,6>(i*3, 0) = J;

                    b_all.conservativeResize(b_all.rows()+3, b_all.cols()); // [(6+3(n-omit_num+1) x 1]
                    // b_all.block<6,1>(0, 0) += -J.transpose() * e; // the first [6 x 1] block is for pose variables
                    // b_all.block<3,1>(6+3*i, 0) = R_tmp.transpose() * e; // note: there are two negative signs, thus positive
                }
                vec_jac_pts1.push_back(R_tmp); // store the Jacobain of error wrt landmark, each term is [3 x 3]
                vec_b_pts1.push_back(-R_tmp.transpose() * e); // store the RHS of the Eq., each term is [3 x 1]
            }

            vec_jac_pose.push_back(J); // store the Jacobian into the storage variable, each term is [3 x 6]

            b += -J.transpose() * e; // [6 x 1]
        }

        // set the terms to zeros
        H_all.setZero();
        J_all.setZero();
        b_all.setZero();

        // allocate the cooresponding blocks
        b_all.block<6,1>(0,0) = b;
        for (int i = 0; i < pts1.size(); i++) {
            if (i <= (int)pts1.size()-omit_num) {
                J_all.block<3,6>(i*3, 0) = vec_jac_pose[i];
                J_all.block<3,3>(i*3, 6+3*i) = vec_jac_pts1[i];
                b_all.block<3,1>(6+3*i,0) = vec_b_pts1[i];
            } else {
                J_all.block<3,6>(i*3, 0) = vec_jac_pose[i];
            }
        }

        // compute the Hessian matrix: H = J^T J
        H_all = J_all.transpose() * J_all; // [(6+3(n-omit_num+1)) x ((6+3(n-omit_num+1))]

        Eigen::VectorXd dx;
        dx = H_all.ldlt().solve(b_all); // [(6+3(n-omit_num+1)) x 1], the first 6 elements are for pose variables

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        // if (iter > 0 && cost >= lastCost) {
        //     // cost increases, ending the iterative process
        //     cout << "cost: " << cost << ", last cost: " << lastCost << endl;
        //     break;
        // }

        // update the pose estimate
        pose = Sophus::SE3d::exp(dx.head(6)) * pose;

        // update the landmark coordinates
        for (int j = 0; j <= (int)points_3d_ba.size()-omit_num; j++) {
            points_3d_ba[j][0] += dx[6+3*j];
            points_3d_ba[j][1] += dx[7+3*j];
            points_3d_ba[j][2] += dx[8+3*j];
        }

        lastCost = cost;

        cout << "iteration " << iter << " cost = " << cost << endl;
        if (dx.norm() < 1e-6) {
            cout << "Optimization converges." << endl;
            break;
        }

        cout << "Pose by ICP-BA with GN: \n" << pose.matrix() << endl;
    }
}
