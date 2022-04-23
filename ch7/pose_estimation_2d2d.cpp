#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

/**********************************************************************************
* This program performs pose estimation using matched 2D-2D feature correspondences
**********************************************************************************/
void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

void pose_estimation_2d2d(
    std::vector<KeyPoint> keypoints_1,
    std::vector<KeyPoint> keypoints_2,
    std::vector<DMatch> matches,
    Mat &R, Mat &t);

// Convert pixel coordinates to camera frame normalized homogeneous coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    cout.precision(8);

    if (argc != 3) {
        cout << "usage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }

    // -- Loading the input images
    // Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    // Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR); // OpenCV 4
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR); // OpenCV 3
    assert(img_1.data && img_2.data && "Cannot load the images.");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Number of point correspondences found in total: " << matches.size() << endl;

    // -- Estimate the camera transformation between two frames using two images
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // -- Verify E = t^ R * scale
    Mat t_x = 
        (Mat_<double>(3, 3) <<                  0,  -t.at<double>(2,0),     t.at<double>(1,0),
                                t.at<double>(2,0),                   0,    -t.at<double>(0,0),
                               -t.at<double>(1,0),   t.at<double>(0,0),                    0);

    cout << "t^R = " << endl << t_x * R << endl;

    // -- Verify Epipolar constraints
    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // provided camera intrinsics
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constaint = " << d << endl;
    }
    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    // -- initialization
    Mat descriptors_1, descriptors_2;
    // using the built-in ORB functions in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // Using the brute-force method with hamming distance to find the feature correspondences
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    // -- Step 1: Detecting the Oriented FAST (corner) keypoints
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // -- Step 2: Generating BRIEF descriptors based on the detected keypoints
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // -- Step 3: Matching the features from the two images using the Hamming distance
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING);
    matcher->match(descriptors_1, descriptors_2, match);

    // -- Step 4: Filtering the matched feature correspondences
    double min_dist = 10000, max_dist = 0;

    // finding out the min and max distances between the matched features, 
    // i.e., distances between the most similar and most dislike feature correspondences
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance; // note: match is in DMatch class
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    // When the distance between two descriptors is greater than two times of the min_dist,
    // it is recognized as a mismatch. However, sometimes the min_dist is quite small, so we set a
    // lower-bound threshold based on experience.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2.0 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}


Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), // fu * x + cu = u -> (u - cu) / fu = x
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)  // fv * y + cv = v -> (v - cv) / fv = y
    );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {
    // camera intrinsics, given by TUM Freiburg2
    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // -- Converting the matched feature keypoints into vector<Point2f> type
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt); // using the first image as anchor/query point
        points2.push_back(keypoints_2[matches[i].trainIdx].pt); // using the second img as destination image
    }

    // -- Calculate the fundamental matrix, F
    Mat fundamental_matrix;
    // fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT); // OpenCV 3
    fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT); // OpenCV 4
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    // -- Calculate the essential matrix, E
    Point2d principal_point(325.1, 249.7); // camera's principal point, calibrated in TUM dataset
    double focal_length = 521; // camera's focal length, calibrated in TUM dataset
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    // -- Calculate the homography matrix
    // In this example, the scene is not planar, and the calculated homography matrix will not result in good performance
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    // -- Recover the rotation and translation from the essential matrix
    // this function is provided in OpenCV 3
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}