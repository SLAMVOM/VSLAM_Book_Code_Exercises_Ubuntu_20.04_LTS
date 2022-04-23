#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

// #include <boost/timer.hpp>

// loading Sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// loading Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/***********************************************************************************************
* This program illustrates a dense depth estimation under a known trajectory using a mono camera
* Using epipolar line search + NCC matching for depth map construction
* *********************************************************************************************/

// --------------------------------------------------------------------------------------------
// parameters
const int boarder = 20;     // image boarder width
const int width = 640;      // image width
const int height = 480;     // image height
const double fx = 481.2f;   // camera intrinsics
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;  // half width of an NCC window
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // area of NCC window
const double min_cov = 0.1;     // convergence threshold: smallest covariance
const double max_cov = 10;      // divergence threshold: largest covatiance

//-----------------------------------------------------------------------------------------------
// Necessary functions
/// loading data from REMODE dataset
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
* update depth estimate according to new images
* @param ref            reference image
* @param curr           current image
* @param T_C_R          transformation from reference to current image
* @param depth          depth
* @param depth_cov      covariance of depth
* @return               a boolean indicating whether the update succeeds
*/
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
* epipolar line search
* @param ref            reference image
* @param curr           current image
* @param T_C_R          transformation from reference to current image
* @param pt_ref         a point in the reference image
* @param depth_mu       mean depth value
* @param depth_cov      covariance of depth
* @param pt_curr        current point
* @param epipolar_direction  epipolar line direction
* @return               success or fail (boolean)
*/
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
* update depth filter
* @param pt_ref         reference image point
* @param pt_curr        current image point
* @param T_C_R          transformation from reference to current image
* @param epipolar_direction     epipolar line direction
* @param depth          mean depth value
* @param depth_cov2     covariance of depth
* @return               success or fail (boolean)
*/
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
* Calculating NCC score
* @param ref            reference image
* @param curr           current image
* @param pt_ref         reference point
* @param pt_curr        current point
* @return               NCC score
*/
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// Bilinear Interpolation of pixel values
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------------------------------
// Some ultility functions
// showing the estiated depth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// pixel coordinate to normalized image plane coordinate
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// camera coordinate to pixel coordinate, note the difference between this function can px2cam
// In px2cam function above, the point is only transformed to the normalized image plane, not the camera frame 
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// verify if a point falls within an image region
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1,0) >= boarder
           && pt(0, 0) + boarder <= width && pt(1, 0) + boarder <= height;
}

// showing the epipolar line match
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// showing the epipolar line
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

// evaluate depth estimates
void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// -------------------------------------------------------------------


int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // reading data from the dataset
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // The first image
    Mat ref = imread(color_image_files[0], 0);              // flag = 0 for grayscale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;                                // initial depth value
    double init_cov2 = 3.0;                                 // initial covariance value
    Mat depth(height, width, CV_64F, init_depth);           // depth map
    Mat depth_cov2(height, width, CV_64F, init_cov2);       // depth map covariance

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);     // load as grayscale image
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // transformation relationship: T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaluateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) 
{
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // data format: image name, tx, ty, tz, qx, qy, qz, qw, note: the transformation is TWC, not TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;    // convert unit from cm to m
        }
    
    return true;
}

// update the whole depth map
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = boarder; x < width - boarder; x++)
        for (int y = boarder; y < height - boarder; y++) {
            // traverse every single pixel
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // either depth is converged or diverged
                continue;
            // searching for correspondences of (x,y) along the epipolar line
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            if (ret == false) // matching fails
                continue;

            // uncomment the following line to show epipolar match
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // if matching succeeds, updating the depth map
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
    }
    return true;
}

// epipolar line search
// details see Section 12.2 in VSLAM book ver2
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) {
    
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;      // p vector in the reference frame 

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref);  // projection of pixel according to mean depth
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;  // using three stdev as threshold
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));     // projected pixel based on minimum depth
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));     // projected pixel based on maximum depth

    Vector2d epipolar_line = px_max_curr - px_min_curr;         // epipolar line (in line segment format
    epipolar_direction = epipolar_line;                         // epipolar line direction
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();            // half length of the epipolar line
    if (half_length > 100) half_length = 100;                   // limiting the searching range
    
    // uncomment the following line of code to show epipolar line (line segment)
    //showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // searching along the epipolar line, using the mean depth point as center, each left and right extent for half len
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7){  // l += sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;   // the (query) point to be matched against the ref pt
        if (!inside(px_curr))
            continue;
        // Calculate the query point's NCC with respect to the reference point
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)       // only accept the match when the best NCC score over a certain threshold
        return false;
    pt_curr = best_px_curr;
    return true;
}

// function to calculate the NCC between one image patch to another
double NCC(
    const Mat &ref, const Mat &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // zero mean normalized cross correlation
    // first calculate the mean value
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr;     // mean values of the reference and current frames
    for (int x = -ncc_window_size; x <= ncc_window_size; x++) // x for columns
        for (int y = -ncc_window_size; y <=ncc_window_size; y++) { // y for rows
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // Calculate Zero mean NCC
    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        denominator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        denominator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(denominator1 * denominator2 + 1e-10);   // adding a small value to avoid dividing by zero
}

bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) {
    // using triangulation to calculate depth
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref); // [3x1]
    f_ref.normalize();  // after normalization, this is a unit vector pointing to the direction of the point
    Vector3d f_curr = px2cam(pt_curr); // [3x1]
    f_curr.normalize();

    // Equations
    // d_ref * f_ref = d_cur * (R_RC * f_cur) + t_RC
    // f2 = R_RC * f_cur
    // multiply f_ref^T at the both sizes in the first eqn. 
    // multiply f_2^T at the both sizes in the second eqn.
    // convert into the following equation matrix
    // => [ f_ref^T f_ref, -f_ref^T f2 ]  [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref,   -f2^T f2    ]  [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;               // result in the reference frame, [3x1]
    Vector3d xn = t + ans[1] * f2;              // result in the current frame, [3x1]
    Vector3d p_esti = (xm + xn) / 2.0;          // location of P, taking the average of both, [3x1]
    double depth_estimation = p_esti.norm();    // depth value

    // Calculate the uncertainty (assuming one pixel of error)
    // Equations see Seciont 12.2 in VSLAM book ed.2
    Vector3d p = f_ref * depth_estimation; // f_ref is a unit direction vector, and multiplying depth will connect the point to the optical center
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // fusion of Gaussian distributions
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// showing the depth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

// evaluate the depth estimates
void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;     // average error
    double ave_depth_error_sq = 0;  // squared error
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

// showing the matched points on the reference and current image
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    // cv::cvtColor(ref, ref_show, CV_GRAY2BGR);   // convert from single band grayscale to three bands grayscale - OpenCV 3
    // cv::cvtColor(curr, curr_show, CV_GRAY2BGR); // convert from single band grayscale to three bands grayscale - OpenCV 3
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);   // covert from single-band grayscale to three-bands grayscale - OpenCV 4
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR); // covert from single-band grayscale to three-bands grayscale - OpenCV 4
    

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

// showing the epipolar line on the current image
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {
    
    Mat ref_show, curr_show;
    // cv::cvtColor(ref, ref_show, CV_GRAY2BGR);   // covert from single-band grayscale to three-bands grayscale - OpenCV 3
    // cv::cvtColor(curr, curr_show, CV_GRAY2BGR); // covert from single-band grayscale to three-bands grayscale - OpenCV 3
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);   // covert from single-band grayscale to three-bands grayscale - OpenCV 4
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR); // covert from single-band grayscale to three-bands grayscale - OpenCV 4
    

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
