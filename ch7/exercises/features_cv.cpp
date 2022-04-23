/*****************************************
* This script is to compute image features 
* using various feature detectors in OpenCV 

* OpenCV version: 3.4.16
******************************************/
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <oepncv2/xfeatures2d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: features_cv img1 img2" << endl;
        return 1;
    }

    // Loading the input images
    // Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    // Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); // OpenCV 3
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR); // OpenCV 4
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR); // OpenCV 4
    assert(img_1.data != nullptr && img_2.data != nullptr && "Cannot load the img"); // check if the input image is empty

    // -- Initialization
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;

    /*************************************
    * SIFT
    *************************************/
    // -- Step 1: Detect keypoints using SIFT detector, may or may not compute the descriptors
    int num_pts = 1000;
    cv::Ptr<SIFT> detector = cv::SIFT::create(num_pts);
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detect(img_1, keypoints_1);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    // -- Illustrate the detected SIFT keypoints
    cv::Mat img_out_SIFT;
    cv::drawKeypoints(img_1, keypoints_1, img_out_SIFT);
    cout << "Number of SIFT keypoints detected: " << keypoints_1.size() << endl;
    cout << "Time used for detecting 1000 SIFT keypoints: " << time_used.count() << " seconds." << endl;
    imshow("SIFT keypoints", img_out_SIFT);


    /*************************************
    * ORB - Oriented FAST
    *************************************/
    // -- Step 1: Detect keypoints using SIFT detector, may or may not compute the descriptors
    num_pts = 1000;
    cv::Ptr<ORB> detector_ORB = cv::ORB::create(num_pts);
    t1 = chrono::steady_clock::now();
    // detector_ORB->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector_ORB->detect(img_1, keypoints_1);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // detector_ORB->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    // -- Illustrate the detected SIFT keypoints
    cv::Mat img_out_ORB;
    cv::drawKeypoints(img_1, keypoints_1, img_out_ORB);
    cout << "Number of ORB keypoints detected: " << keypoints_1.size() << endl;
    cout << "Time used for detecting 1000 ORB keypoints: " << time_used.count() << " seconds." << endl;
    imshow("ORB keypoints", img_out_ORB);

    // check the matching time between images
    // using brute-force method with Hamming distance for feature matching
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    detector_ORB->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector_ORB->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "\nTime used for detecting features in both img1 and img2: " << time_used.count() << " seconds." << endl;

    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time used for matching the ORB features in the two images: " << time_used.count() << " seconds.\n" << endl;
       


    /*************************************
    * GFTTDetector
    *************************************/
    // -- Step 1: Detect keypoints using SIFT detector, may or may not compute the descriptors
    num_pts = 1000;
    cv::Ptr<GFTTDetector> detector_GFTT = cv::GFTTDetector::create(num_pts);
    t1 = chrono::steady_clock::now();
    detector_GFTT->detect(img_1, keypoints_1);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    // -- Illustrate the detected SIFT keypoints
    cv::Mat img_out_GFTT;
    cv::drawKeypoints(img_1, keypoints_1, img_out_GFTT);
    cout << "Number of GFTT keypoints detected: " << keypoints_1.size() << endl;
    cout << "Time used for detecting 1000 GFTT keypoints: " << time_used.count() << " seconds." << endl;
    imshow("GFTT keypoints", img_out_GFTT);


    waitKey();
    return 0;
}