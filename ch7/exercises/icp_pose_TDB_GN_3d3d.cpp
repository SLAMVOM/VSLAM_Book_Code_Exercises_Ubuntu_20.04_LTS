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
    Sophus::SE3d &pose // or using Sophus for transformation
);

// Method introduced in Barfoot's book
void bundleAdjustment_GN_TB(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    Sophus::SE3d &pose // or using Sophus for transformation
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

    Sophus::SE3d pose_gn; // if using Sophus SE3 for poses
    // Eigen::Matrix3d R_eigen; // given the pose estimate from the OpenCV as initial guess
    // R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
    //            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
    //            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    // Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    // Sophus::SE3d pose_gn(R_eigen, t_eigen); /////////
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    bundleAdjustment_GN(pts1_eigen, pts2_eigen, pose_gn); // if using Sophus SE3 for poses
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by looping method for iters: " << time_used.count() << " seconds." << endl;


    // Using method introduced in Barfoot's book
    Sophus::SE3d pose_gn_TB; // if using Sophus SE3 for poses
    // Eigen::Matrix3d R_eigen; // given the pose estimate from the OpenCV as initial guess
    // R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
    //            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
    //            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    // Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    // Sophus::SE3d pose_gn(R_eigen, t_eigen); /////////
    t1 = chrono::steady_clock::now();
    bundleAdjustment_GN_TB(pts1_eigen, pts2_eigen, pose_gn_TB); // if using Sophus SE3 for poses
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time spent by TDB method for iters: " << time_used.count() << " seconds." << endl;


    //verify p1 = R * p2 + t, here p1 and p2 are individual feature points
    for (int i = 0; i < 5; i++) { // show the first three points
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "(R * p2 + t) =\n" << 
            R * (Mat_<double>(3,1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
            << endl;
        cout << endl;
    }

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
                         Sophus::SE3d &pose // or using Sophus for transformation
                         ) 
                         {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 50; // maximum number of iterations
    double cost = 0, lastCost = 0;

    cout << "Initial pose: \n" << pose.matrix() << endl;
    
    // first layer for loop for iterations
    for (int iter = 0; iter < iterations; iter++) {
        
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // second layer for loop to compute cost accumulated through each individual landmark
        for (int i = 0; i < pts1.size(); i++) {
            Eigen::Vector3d p_trans = pose * pts1[i]; // compute the landmark coor under the second frame
            Eigen::Vector3d e = pts2[i] - p_trans; // compute the error for each landmark
            
            cost += e.squaredNorm(); // accumulate the error
            Eigen::Matrix<double, 3, 6> J;
            J << -1,  0,  0,           0, -p_trans[2],  p_trans[1], // first row
                  0, -1,  0,  p_trans[2],           0, -p_trans[0], // second row
                  0,  0, -1, -p_trans[1],  p_trans[0],           0; // third row


            H += J.transpose() * J; // [6 x 6]
            b += -J.transpose() * e; // [6 x 1]
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increases, ending the iterative process
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update the pose estimate
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost = " << cost << endl;
        if (dx.norm() < 1e-6) {
            cout << "Optimization converges." << endl;
            break;
        }

        cout << "Pose by ICP GN: \n" << pose.matrix() << endl;
    }
}


void bundleAdjustment_GN_TB(const VecVector3d &pts1,
                         const VecVector3d &pts2,
                         Sophus::SE3d &pose // or using Sophus for transformation
                         ) 
                         {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 50; // maximum number of iterations
    // double cost = 0, lastCost = 0;

    cout << "\nMethod by Prof. Barfoot --------------------- \n" << endl;
    
    Eigen::Vector3d pts1_cen = Eigen::Vector3d::Zero();
    Eigen::Vector3d pts2_cen = Eigen::Vector3d::Zero();
    double count_pt = 0.0;
    
    // Calculate the centroids
    for (int i_n = 0; i_n < pts1.size(); i_n++) {
        pts1_cen += pts1[i_n];
        pts2_cen += pts2[i_n];
        count_pt += 1.0;
    }
    pts1_cen = pts1_cen / count_pt; // centroid of first pt cloud
    pts2_cen = pts2_cen / count_pt; // centroid of second pt cloud

    // Calculate the big M matrix and the W matrix
    Eigen::Matrix<double,3 ,3> W = Eigen::Matrix<double, 3, 3>::Zero();
    // define the left and right matrices of the big M matrix
    Eigen::Matrix<double, 6, 6> M_l, M_r, M_big;

    M_l <<             1,            0,            0,   0,  0,  0,
                       0,            1,            0,   0,  0,  0,
                       0,            0,            1,   0,  0,  0,
                       0,  -pts1_cen[2], pts1_cen[1],   1,  0,  0,
            pts1_cen[2],            0,  -pts1_cen[0],   0,  1,  0,
             -pts1_cen[1], pts1_cen[0],            0,   0,  0,  1;

    M_r <<  1, 0, 0,            0,  pts1_cen[2],    -pts1_cen[1],
            0, 1, 0,  -pts1_cen[2],             0,   pts1_cen[0],
            0, 0, 1, pts1_cen[1],   -pts1_cen[0],              0,
            0, 0, 0,            1,             0,              0,
            0, 0, 0,            0,             1,              0,
            0, 0, 0,            0,             0,              1;

    Eigen::Matrix<double, 6, 6> M_m = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 3, 3> M_mid_temp = Eigen::Matrix<double, 3, 3>::Zero();
    for (int i = 0; i < pts1.size(); i++) {
        Eigen::Vector3d temp_vec;
        Eigen::Matrix<double, 3, 3> temp_Mat = Eigen::Matrix<double, 3, 3>::Zero();
        temp_vec = pts1[i] - pts1_cen;
        temp_Mat <<             0,  -temp_vec[2],   temp_vec[1],
                      temp_vec[2],             0,  -temp_vec[0],
                     -temp_vec[1],   temp_vec[0],             0;
        M_mid_temp += temp_Mat * temp_Mat;

        // compute the W matrix
        W += (pts2[i] - pts2_cen) * temp_vec.transpose();
    }
    M_m.block<3,3>(3,3) = -M_mid_temp / count_pt; // notice the negative sign
    M_m.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

    M_big = M_l * M_m * M_r; // compute the big M matrix
    Eigen::Matrix<double, 6, 6> M_big_inv = M_big.inverse();
    
    // compute the W matrix, which is an average matrix
    W = W / count_pt; // [3x3]

    // first layer for loop for iterations
    for (int iter = 0; iter < iterations; iter++) {
        // Extract the current estimate of rotation matrix and translation vector for convenience
        Eigen::Vector3d t_cur = pose.matrix().block<3, 1>(0, 3);
        Eigen::Matrix<double, 3, 3> C_cur = Eigen::Matrix<double, 3, 3>::Zero();
        C_cur = pose.matrix().block<3, 3>(0, 0);

        // Construct the adjoint transformation matrix
        Eigen::Matrix<double, 6, 6> T_adj = Eigen::Matrix<double, 6, 6>::Zero();
        T_adj.block<3, 3>(0, 0) = C_cur;
        T_adj.block<3, 3>(3, 3) = C_cur;
        
        Eigen::Matrix<double, 3, 3> temp_Mat;
        temp_Mat <<                   0, -pose.matrix()(2,3),  pose.matrix()(1,3),
                     pose.matrix()(2,3),                   0, -pose.matrix()(0,3),
                    -pose.matrix()(1,3),  pose.matrix()(0,3),                   0;
        temp_Mat = temp_Mat * C_cur;
        T_adj.block<3, 3>(0, 3) = temp_Mat;

        // Construct the [6x1] a vector, which includes the process of building the [3x1] b vector
        temp_Mat = Eigen::Matrix<double, 3, 3>::Zero(); // reset the temp matrix
        double b1, b2, b3;
        temp_Mat << 0, 0,  0,
                    0, 0, -1,
                    0, 1,  0;
        temp_Mat = temp_Mat * C_cur * W.transpose(); // [3x3]
        b1 = temp_Mat(0,0) + temp_Mat(1,1) + temp_Mat(2,2); // first term of b vector

        temp_Mat << 0,  0, 1,
                    0,  0, 0,
                    -1, 0, 0;
        temp_Mat = temp_Mat * C_cur * W.transpose(); // [3x3]
        b2 = temp_Mat(0,0) + temp_Mat(1,1) + temp_Mat(2,2); // second term of b vector

        temp_Mat << 0, -1, 0,
                    1,  0, 0,
                    0,  0, 0;
        temp_Mat = temp_Mat * C_cur * W.transpose(); // [3x3]
        b3 = temp_Mat(0,0) + temp_Mat(1,1) + temp_Mat(2,2); // third term of b vector
        Eigen::Vector3d b_vec = Eigen::Vector3d(b1, b2, b3);


        temp_Mat = Eigen::Matrix<double, 3, 3>::Zero(); // reset the temp matrix
        temp_Mat << 0, -pts2_cen[2], pts2_cen[1],
                    pts2_cen[2], 0 ,-pts2_cen[0],
                    -pts2_cen[1], pts2_cen[0], 0;

        Vector6d a_vec = Vector6d::Zero(); // define the [6x1] a_vec vector
        a_vec.block<3,1>(0,0) = pts2_cen - C_cur * pts1_cen - t_cur;
        a_vec.block<3,1>(3,0) = b_vec - temp_Mat * (C_cur * pts1_cen + t_cur);


        // Calculate the optimal update, [6x1] dx
        Vector6d dx; // [6 x 1]
        dx = T_adj * M_big_inv * T_adj.transpose() * a_vec;


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
        pose = Sophus::SE3d::exp(dx) * pose;
        // lastCost = cost;

        // cout << "iteration " << iter << " cost = " << cost << endl;
        if (dx.norm() < 1e-6) {
            cout << "Optimization converges." << endl;
            break;
        }

        cout << "Pose by ICP GN TDB: \n" << pose.matrix() << endl;
    }
}