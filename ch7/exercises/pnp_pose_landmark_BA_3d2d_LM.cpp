/*************************************************************************
* This script includes solving the PnP BA problem, 
* including landmark optimization by explictly implementing using 
* Levenberg-Marquardt. Note: there are various versions of LM algorithms.
* The one implemented here is just one of the many workable versions.
Reference: 
Gavin (2020) "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems"
*************************************************************************/

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
 * unique or overdetermined solution.
********************************************************************/

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

// BA by gauss-newton
void bundleAdjustmentLM(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose,
    VecVector3d &points_3d_ba
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

    // // Using OpenCV's implemented method to solve the PnP problem
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

    // a deep copy of pts_3d_eigen for bundle adjustment that also updates landmarks
    VecVector3d pts_3d_ba = pts_3d_eigen;

    // Levenber-Marquardt
    cout << "Calling bundle adjustment by Levenberg-Marquardt" << endl;
    Sophus::SE3d pose_LM; // using an identity matrix as initial guess
    // Eigen::Matrix3d R_eigen;
    // R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
    //            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
    //            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);/////////
    // Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2)); /////////
    // Sophus::SE3d pose_LM(R_eigen, t_eigen); /////////
    t1 = chrono::steady_clock::now();
    bundleAdjustmentLM(pts_3d_eigen, pts_2d_eigen, K, pose_LM, pts_3d_ba);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 -t1);
    cout << "Time spent by solving PnP in Levenber-Marquardt: "<< time_used.count() << " seconds." << endl;

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

void bundleAdjustmentLM(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose,
    VecVector3d &points_3d_ba) {

    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 50;
    double cost = 0.0, lastCost = 0.0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    // Levenber-Marquardt parameters:
    // Gavin (2020) "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems"
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D; // the diaonal matrix in L-M
    double L_upper = 11.0, L_lower = 9.0; // hyperparameters
    double lambda_init = 1e-2; // initial value of the L-M damping parameter
    double epsilon = 1e-1; // update acceptance threshold
    double lambda = lambda_init;
    double rho;
    double cost_tmp = 0.0;
    Eigen::Vector3d dx_tmp; // a temporary update of landmark pts for L-M
    Sophus::SE3d pose_tmp; // a temporary updated SE3 pose


    int N = (int)points_3d.size(); // number of landmarks or features

    // does not update the last several 3D landmarks in the reference world frame
    int omit_num = (int)(N/3+2) + 1; // will ignore the last (omit_num-1) landmakrs, omit_num = N + 1 to only optimize pose!

    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J_all; // the jacobian matrix is of [2n x (6+3(n-omit_num+1))]
        Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero(); // Jacobian matrix for the pose variables

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_all; // H_all = J^T * J, [(6+3(n-omit_num+1)) x (6+3(n-omit_num+1))]
        
        Eigen::Matrix<double, Eigen::Dynamic, 1> b_all; // initialize the RHS of Gauss-Newton [(6+3(n-omit_num+1)) x 1]: -J * e
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero() ; // the RHS term in Gauss-Newton that associated with the pose variables

        // define vector containers to store the jacobian of e wrt the pts and poses
        std::vector<Eigen::Matrix<double, 2, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 3>>> vec_jac_pts; // Jacobian for pts
        std::vector<Eigen::Matrix<double, 2, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 6>>> vec_jac_pose; // Jacobian for poses
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vec_b_pts; // RHS for each pt

        J_all = J*1.0; // initialize J_all as J, [2 x 6]
        b_all = b*1.0; // initialize b_all as b, [6 x 1]

        cost = 0.0;
        // compute the cost
        for (int i = 0; i < points_3d.size(); i++) {
            // compute the current p^{i}_{c2}
            Eigen::Vector3d pc = pose * points_3d_ba[i]; // [3 x 1]

            double inv_z = 1.0 / pc[2]; // the third element as normalizing scalar
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy); // projected pixel coor. on frame 2

            Eigen::Vector2d e = points_2d[i] - proj; // calculate the reprojection error, observed - projected

            cost += e.squaredNorm(); // the overall cost is the sum of reprojection errors of all landmarks
            Eigen::Matrix<double, 2, 6> J; // the jacobian of e wrt the pose variables
            Eigen::Matrix<double, 2, 3> J_pts; // the jacobian of e wrt the landmark coordinates
            
            // compute the jacobian of e wrt the pose variables
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

            if (i > (int)points_3d.size()-omit_num) { // omit the last (omit_num-1) landmarks, Note: .size() returns an unsigned int
                J_all.conservativeResize(J_all.rows()+2, J_all.cols()); // [2n x (6+2(n-omit_num+1))]
            } else { // for the first (N - omit_num +1) landmarks 
                
                // compute the jacobian of e wrt the current landmark coordinates
                J_pts <<-fx * inv_z,
                        0,
                        fx * pc[0] * inv_z2,
                        0,
                        -fy * inv_z,
                        fy * pc[1] * inv_z2; // [2 x 3]
                J_pts = J_pts * pose.rotationMatrix(); // [2 x 3]
                
                if (i == 0) { // the first landmark to be optimized
                    J_all.conservativeResize(J_all.rows(), J_all.cols()+3); // [2 x 9] to include jacobian for landmark
                    b_all = -J_all.transpose() * e; // [(6+3)x1] = [9 x 1]

                } else { // for the rest landmarks
                    // extend the J_all matrix to include the current Jacobian, J
                    J_all.conservativeResize(J_all.rows()+2, J_all.cols()+3); // [2n x (6+3(n-omit_num+1))]

                    b_all.conservativeResize(b_all.rows()+3, b_all.cols()); // [(6+3(n-omit_num+1) x 1]
                }
                vec_jac_pts.push_back(J_pts); // store the Jac w.r.t. landmark point
                vec_b_pts.push_back(-J_pts.transpose() * e); // store the RHS for the landmark
            }

            vec_jac_pose.push_back(J); // store the Jac w.r.t. to the pose variables

            b += -J.transpose() * e; // [6x1] accumulate the RHS for pose variables
        }

        // first set all entries in J_all, b_all, H_all to zeros and then use a for loop to construct J_all, b_all
        J_all.setZero();
        H_all.setZero();
        b_all.setZero();

        // allocate the corresponding blocks
        b_all.block<6,1>(0,0) = b; // b for pose

        for (int i = 0; i < points_3d.size(); i++) {
            if (i <= (int)points_3d.size()-omit_num) { // for the first (N - omit_num +1) landmarks 
                J_all.block<2,6>(i*2, 0) = vec_jac_pose[i];
                J_all.block<2,3>(i*2, 6+3*i) = vec_jac_pts[i];
                b_all.block<3,1>(6+3*i, 0) = vec_b_pts[i];
            } else { // for the last omit_num - 1 landmarks
                J_all.block<2,6>(i*2, 0) = vec_jac_pose[i];
            }
        }

        // compute the Hessian matrix: H_all = J_all^T J_all
        H_all = J_all.transpose() * J_all; // [(6+3(n-omit_num+1)) x (6+3(n-omit_num+1))]

        // L-M steps
        D = H_all.diagonal().asDiagonal(); // construct the digonal matrix D
        H_all += lambda * D; // [(6+3(n-omit_num+1)) x (6+3(n-omit_num+1))]

        // solve for the optimal update
        Eigen::Matrix<double, Eigen::Dynamic, 1> dx; // [(6+3(n-omit_num+1)) x 1], the first 6 elements are for pose variables
        dx = H_all.ldlt().solve(b_all); // using Eigen's Robust Cholesky decomposition solver to solve for optimal update

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        // if (iter > 0 && cost >= lastCost) {
        //     // once the cost starts increasing, stop incrementing the update
        //     cout << "cost: " << cost << ", last cost: " << lastCost << endl;
        //     break;
        // }


        // L-M update of lambda
        cost_tmp = 0.0; // reset the temporary cost
        // compuate the temporary pose update
        pose_tmp = Sophus::SE3d::exp(dx.head(6)) * pose;
        for (int i = 0; i <= (int)points_3d.size()-omit_num; i++) {
            Eigen::Vector3d dx_tmp(dx[6+3*i], dx[7+3*i], dx[8+3*i]);
            // compute the current p^{i}_{c2}
            Eigen::Vector3d pc_tmp = pose_tmp * (points_3d_ba[i] + dx_tmp); // [3 x 1]
            double inv_z = 1.0 / pc_tmp[2]; // the third element as normalizing scalar
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc_tmp[0] / pc_tmp[2] + cx, fy * pc_tmp[1] / pc_tmp[2] + cy); // projected pixel coor. on frame 2

            Eigen::Vector2d e = points_2d[i] - proj; // calculate the reprojection error, observed - projected

            cost_tmp += e.squaredNorm(); // the overall cost is the sum of reprojection errors of the updated landmarks
        }

        // compute the rho value for the LM update
        rho = (cost - cost_tmp) / (dx.transpose() * (lambda * D * dx + b_all)); // [#]
        if (rho > epsilon) { // accept the update if the update results in a rho greater than the threshold
            pose = Sophus::SE3d::exp(dx.head(6)) * pose; // update the pose
            // update the landmarks using a for loop
            for (int j = 0; j <= (int)points_3d.size()-omit_num; j++) {
                points_3d_ba[j][0] += dx[6+3*j];
                points_3d_ba[j][1] += dx[7+3*j];
                points_3d_ba[j][2] += dx[8+3*j];
            }
            lambda = std::max(lambda / L_lower, 1e-7); // update the damping param lambda
        } else { // reject the update and change the value of lambda if rho is below the threshold
            lambda = std::min(lambda * L_upper, 1e7);
        }

        // // update the pose estimation
        // pose = Sophus::SE3d::exp(dx.head(6)) * pose; // here is a matrix exponential
        lastCost = cost;
        cout << "Iteration " << iter << " cost = " << std::setprecision(12) << cost << endl;

        // // update the landmarks using a for loop
        // for (int j = 0; j <= (int)points_3d.size()-omit_num; j++) {
        //     points_3d_ba[j][0] += dx[6+3*j];
        //     points_3d_ba[j][1] += dx[7+3*j];
        //     points_3d_ba[j][2] += dx[8+3*j];
        // }

        if (dx.norm() < 1e-6) {
            // if the norm of update for variables is smaller than 1e-6 -> convergence
            cout << "Optimization converges." << endl;
            break;
        }
    }
    cout << "pose by PnP Levenberg-Marquardt: \n" << pose.matrix() << endl;
}