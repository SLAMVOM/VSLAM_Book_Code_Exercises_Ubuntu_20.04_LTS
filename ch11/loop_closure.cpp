#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/******************************************************************************
* This script illustrates how to evaluate using the previous learned dictionary
******************************************************************************/

int main(int argc, char **argv) {
    // read the images and database
    cout << "reading database" << endl;
    DBoW3::Vocabulary vocab("./vocabulary.yml.gz");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz"); // using a larger vocabulary
    if (vocab.empty()) {
        cerr << "Vocabulary does not exist." << endl;
        return 1;
    }
    cout << "reading images ..." << endl;
    vector<Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "../data/" + to_string(i + 1) + ".png";
        images.push_back(imread(path));
    }

    // NOTE: when the vocabulary is generated based on the images to be evaluated, this may lead to overfit.
    // detect ORB features
    cout << "detecting ORB features ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for (Mat &image:images) {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // one can compare the images directly or may compare an image to the database images:
    cout << "comparing images with images " << endl;
    for (int i = 0; i < images.size(); i++) {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); j++) {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);
            cout << "image " << i << " vs image " << j << " : " << score << endl;
        }
        cout << endl;
    }

    // compare against database
    cout << "comparing images with database " << endl;
    DBoW3::Database db(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); i++)
        db.add(descriptors[i]);
    cout << "database info: " << db << endl;
    for (int i = 0; i < descriptors.size(); i++) {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4);       // max results=4
        cout << "searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "done." << endl;
}