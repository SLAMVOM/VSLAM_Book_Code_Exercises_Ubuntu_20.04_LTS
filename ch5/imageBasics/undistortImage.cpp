#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
string image_file = "../distorted.png"; // the path to the distorted image file

int main (int argc, char **argv) {
    // This program implements the image undistortion explictly without using opencv functions
    // radial and tangential correction parameters
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // camera intrinsics
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;


    // Note that the loaded image is a distorted image
    cv::Mat image = cv::imread(image_file, 0); // flag 0 to read image in grayscale, CV_8UC1 data type

    // check if it is an empty image
    if (image.data == nullptr) {
        cerr << "Image is not loaded, either file does not exist or is an empty image." << endl;
        return 0;
    }  

    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // a variable to store the undistorted img


    // compute the pixel values of the undistorted image
    for (int v; v < rows; v++) { // image rows, i.e., y direction
        for (int u = 0; u < cols; u++) { // image cols, i.e., x direction
            // note that the (v,u) pixel coordinate is for the undistorted image
            // the calculated (v_dis,x_dis) coordinate is for the loaded (distrorted) image
            double x = (u - cx) / fx, y = (v - cy) / fy; // covert to the normalized image plane coordinate
            double r = sqrt(x * x + y * y); // calcuate the radius in the polar cooridnate system
            
            // according to the radial-tangential model, compute the corresponding distorted pixel
            // coordinate for each of the pixel in the undistorted image
            double x_distorted = x * (1 + k1*r*r + k2*r*r*r*r) + 2*p1*x*y + p2 * (r*r + 2*x*x);
            double y_distorted = y * (1 + k1*r*r + k2*r*r*r*r) + p1 * (r*r + 2*y*y) + 2*p2*x*y;
            double u_distorted = fx * x_distorted + cx; // convert from normalized image plane to pixel coordinate
            double v_distorted = fy * y_distorted + cy;

            // assign pixel value (nearest value)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            }
            else {
                image_undistort.at<uchar>(v, u) = 0; // assign a zero intensity value if outside of image region
            }
        }
    }

    //show the undistorted image
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey();
    return 0;
}