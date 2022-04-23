#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

struct Frame; // Frame defined in frame.h
struct MapPoint; // MapPoint defined in mappoint.h

/**
 * 2D features 
 * the feature will be correlated to one of the map points after triangulation
*/
struct Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr;

        std::weak_ptr<Frame> frame_;            // the frame that owns the feature
        cv::KeyPoint position_;                 // 2D keypoint location
        std::weak_ptr<MapPoint> map_point_;     // the corresponding map point of the feature

        bool is_outlier_ = false;               // whether the feature is an outlier
        bool is_on_left_image_ = true;          // whether the feature is on the left image, false for right image

    public:
        Feature() {}

        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp) {}



};
} // namespace myslam

#endif // MYSLAM_FEATURE_H
