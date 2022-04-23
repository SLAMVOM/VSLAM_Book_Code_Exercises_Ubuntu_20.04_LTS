#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    // Read image in argv[1]
    cv::Mat image;
    image = cv::imread(argv[1]); // call cv::image to read the image from file

    // check if the data are correctly loaded
    if (image.data == nullptr) { // if the data is nullptr, then possibly the file does not exist
        cerr << "File " << argv[1] << " does not exist." << endl;
        return 0;
    }

    // if the image is loaded, print some basic information
    cout << "Image cols: " << image.cols << ", rows: " << image.rows
         << ", channels: " << image.channels() << endl;
    cv::imshow("image", image); // using cnv::imshow to show the image
    cv::waitKey(0); // display the image and wait for a keyboard input

    // check image type
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        // if the input is neither an RGB no a grayscale image, the input is in invalid type
        cout << "Image type incorrect, please input a RGB or grayscale image." << endl;
        return 0;
    }

    // traverse the image, please note that the following traverse method can be used to
    // access random pixel
    // Using std::chrono to time the algorithms
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (size_t y=0; y < image.rows; y++) {
        //using cv::Mat::ptr to get the pointer of each image row
        unsigned char *row_ptr = image.ptr<unsigned char>(y); // row_ptr is the pointer to the start of the yth row
        for (size_t x = 0; x < image.cols; x ++) {
            // access the (x,y) pixel location
            unsigned char *data_ptr = &row_ptr[x * image.channels()]; // data_ptr points to the red chann of (x,y)th px
            // output the intensity value in each channel of the current px location, grayscale has only one channel
            for (int c = 0; c != image.channels(); c++) {
                unsigned char data = data_ptr[c]; // `data` is the intensity value at the c channel of (x,y) px 
            }
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast < chrono::duration< double >> (t2 - t1);
    cout << "Time used to traverse the image " << time_used.count() << " seconds." << endl;


    // About cv::Mat copy
    // operator = will not copy the image data, but only referencing the image (shallow copy)
    cv::Mat image_another = image;
    // modification on `image_another` will also change the values in `image`
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0); // set the top-left 100*100 pixel values to zeros
    cv::imshow("image", image);
    cv::waitKey(0);

    // Using cv::Mat::clone to actually clone the data (deep copy)
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255); // set the top-left 100*100 pixel values to 255
    cv::imshow("image", image);
    cv::waitKey(0);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    // There are many other image processing operations such as clippling, rotating, and scaling.
    // Please refer to OpenCV's official documentation for further information.


    return 0;

}